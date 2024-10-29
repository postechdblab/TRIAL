import abc
import logging
import os
import queue
import random
import shutil
import threading
from contextlib import contextmanager
from typing import *

import faiss
import hkkang_utils.file as file_utils
import numpy as np
import torch
import ujson
from omegaconf import DictConfig

from eagle.dataset.corpus import Corpus
from eagle.index.codecs.residual import ResidualCodec
from eagle.index.codecs.residual_embeddings import BaseResidualEmbeddings
from eagle.model.base_model import BaseModel
from eagle.model.registry import MODEL_REGISTRY
from eagle.tokenization.tokenizers import Tokenizers

TYPICAL_DOCLEN = 120
HELDOUT_FRACTION = 0.05
MIN_HELDOUT_NUM = 50_000

logger = logging.getLogger("BaseIndexer")


# Decorator to check self.is_main_thread and call the function only when it is true
def main_thread_only(func):
    def wrapper(self, *args, **kwargs):
        if self.is_main_thread:
            return func(self, *args, **kwargs)
        else:
            return None

    return wrapper


class BaseIndexer:
    def __init__(self, cfg: DictConfig, rank: int = 0, world_size: int = 1) -> None:
        self.cfg = cfg.indexing
        self.rank = rank
        self.world_size = world_size
        self.corpus: Corpus = Corpus(cfg)
        self.tokenizers = Tokenizers(
            cfg.tokenizers.query, cfg.tokenizers.document, cfg.model.backbone_name
        )
        # Set model
        assert cfg.model.ckpt_path, "model ckpt_path is not provided."
        model_module: BaseModel = MODEL_REGISTRY[cfg.model.name]
        self.model = model_module(cfg=cfg.model, tokenizers=self.tokenizers).to(
            self.device
        )
        # Sample embeddings
        self.sample_embs = None
        # Stats
        self.num_embeddings_est = None
        self.num_partitions = None
        self.avg_doclen_est = None

    def __call__(self) -> None:
        # Check if directory exists
        self._check_index_already_exists()
        self._distributed_barrier()

        # Plan indexing
        self._log_info_main_thread_only("Begin Index planning")
        self.plan()
        self._distributed_barrier()

        # Train kmeans
        self._log_info_main_thread_only(
            "Begin Kmeans clustering with sampled embeddings"
        )
        self.train_kmeans()
        self._distributed_barrier()

        # Encode all documents
        self._log_info_main_thread_only(
            "Start clustering embeddings with the trained centriods"
        )
        self.encoding()
        self._distributed_barrier()

        # Finalize
        self._log_info_main_thread_only(
            "Aggregate indices generated from multiple processes and save metadata"
        )
        self.finalize()

    @property
    def use_gpu(self) -> bool:
        return torch.cuda.is_available()

    @property
    def device(self) -> torch.device:
        return torch.device(f"cuda:{self.rank}" if self.use_gpu else "cpu")

    @property
    def dir_path(self) -> str:
        return os.path.join(
            self.cfg.dir_path, self.corpus.cfg.dataset.name, self.cfg.tag
        )

    @property
    def plan_path(self) -> str:
        return os.path.join(self.dir_path, "plan.json")

    @property
    def num_doc(self) -> int:
        return len(self.corpus)

    @property
    def num_options(self) -> int:
        return 2**self.cfg.nbits

    @property
    def num_chunks(self) -> int:
        return self.corpus.get_num_chunks(world_size=self.world_size)

    @property
    def is_main_thread(self) -> bool:
        return self.rank in [0, -1]

    @main_thread_only
    def _log_info_main_thread_only(self, *args, **kwargs) -> None:
        self._log_info(*args, **kwargs)

    @main_thread_only
    def _check_index_already_exists(self) -> bool:
        if os.path.exists(self.dir_path):
            self._log_info(f"Index directory already exists: {self.dir_path}")
            if not self.cfg.override:
                exit(0)
            # Remove the existing directory
            self._log_info(f"Removing the existing index directory: {self.dir_path}")
            shutil.rmtree(self.dir_path)
            self._log_info(f"Removed the existing index directory: {self.dir_path}")
        os.makedirs(self.dir_path)
        return True

    @main_thread_only
    def _save_plan(self) -> None:
        # Create indexing plan
        d = {"config": dict(self.cfg)}
        d["num_chunks"] = self.num_chunks
        d["num_partitions"] = self.num_partitions
        d["num_embeddings_est"] = self.num_embeddings_est
        d["avg_doclen_est"] = self.avg_doclen_est

        # Save the plan
        self._log_info(f"Saving the indexing plan to {self.plan_path}..")
        file_utils.write_json_file(d, self.plan_path)

    def _distributed_barrier(self) -> None:
        if self.rank >= 0 and self.world_size > 1:
            torch.distributed.barrier()

    def _log_info(self, *args, **kwargs) -> None:
        args = (f"[Rank {self.rank}] " + args[0], *args[1:])
        logger.info(*args, **kwargs)

    def _get_kmeans_heldout_size(self, sample_num: int) -> int:
        return int(min(HELDOUT_FRACTION * sample_num, MIN_HELDOUT_NUM))

    def _compute_avg_residual(
        self, centroids: torch.Tensor, heldout: torch.Tensor
    ) -> None:
        compressor = ResidualCodec(cfg=self.cfg, centroids=centroids, avg_residual=None)

        heldout_reconstruct = compressor.compress_into_codes(heldout, out_device="cuda")
        heldout_reconstruct = compressor.lookup_centroids(
            heldout_reconstruct, out_device="cuda"
        )
        heldout_avg_residual = heldout.cuda() - heldout_reconstruct

        avg_residual = torch.abs(heldout_avg_residual).mean(dim=0).cpu()
        self._log_info(f"{[round(x, 3) for x in avg_residual.squeeze().tolist()]}")

        quantiles = torch.arange(
            0, self.num_options, device=heldout_avg_residual.device
        ) * (1 / self.num_options)
        bucket_cutoffs_quantiles, bucket_weights_quantiles = quantiles[
            1:
        ], quantiles + (0.5 / self.num_options)

        bucket_cutoffs = heldout_avg_residual.float().quantile(bucket_cutoffs_quantiles)
        bucket_weights = heldout_avg_residual.float().quantile(bucket_weights_quantiles)

        self._log_info(
            f"Got bucket_cutoffs_quantiles = {bucket_cutoffs_quantiles} and bucket_weights_quantiles = {bucket_weights_quantiles}"
        )
        self._log_info(
            f"Got bucket_cutoffs = {bucket_cutoffs} and bucket_weights = {bucket_weights}"
        )

        return bucket_cutoffs, bucket_weights, avg_residual.mean()

    def _sample_pids(self) -> Set[int]:
        num_sample_pids = min(
            1 + int(16 * np.sqrt(TYPICAL_DOCLEN * self.num_doc)), self.num_doc
        )
        return set(random.sample(range(self.num_doc), num_sample_pids))

    def _check_all_files_are_saved(self) -> None:
        """Check all chunks are created"""
        self._log_info("Checking all files are saved...")
        check_results = [
            self.check_chunk_exists(chunk_idx) for chunk_idx in range(self.num_chunks)
        ]
        if all(check_results):
            self._log_info("All files are saved!")
        else:
            raise ValueError(f"Some files are missing! (check_results:{check_results})")
        return None

    def _concatenate_and_split_sample(self):
        # TODO: Allocate a float16 array. Load the samples from disk, copy to array.
        sample = torch.empty(self.num_sample_embs, self.cfg.dim, dtype=torch.float16)

        offset = 0
        for r in range(self.world_size):
            sub_sample_path = os.path.join(self.dir_path, f"tok_sample.{r}.pt")
            sub_sample = torch.load(sub_sample_path, weights_only=True)
            os.remove(sub_sample_path)

            endpos = offset + sub_sample.size(0)
            sample[offset:endpos] = sub_sample
            offset = endpos

        assert endpos == sample.size(0), (endpos, sample.size())

        # Shuffle and split out a 5% "heldout" sub-sample [up to 50k elements]
        sample = sample[torch.randperm(sample.size(0))]

        heldout_fraction = 0.05
        heldout_size = int(min(heldout_fraction * sample.size(0), 50_000))
        sample, sample_heldout = sample.split(
            [sample.size(0) - heldout_size, heldout_size], dim=0
        )

        return sample, sample_heldout

    def plan(self) -> None:
        """Calcuate and saves plan.json for the whole corpus."""
        sample_pids = self._sample_pids()
        avg_doclen_est = self._sample_embeddings(sample_pids)

        # Select the number of partitions
        self.num_embeddings_est = self.num_doc * avg_doclen_est
        self.num_partitions = int(
            2 ** np.floor(np.log2(16 * np.sqrt(self.num_embeddings_est)))
        )

        self._log_info_main_thread_only(
            f"Estimated average doclen: {avg_doclen_est:.2f}"
        )
        self._log_info_main_thread_only(
            f"Estimated number of embeddings: {self.num_embeddings_est:.2f}"
        )

        self._save_plan()

    @main_thread_only
    def train_kmeans(self) -> None:
        # Shuffle and split out a 5% "heldout" sub-sample [up to 50k elements]
        sample, heldout_sample = self._concatenate_and_split_sample()

        assert torch.cuda.is_available(), "CUDA is not available."
        torch.cuda.empty_cache()

        # START
        kmeans = faiss.Kmeans(
            d=self.cfg.dim,
            k=self.num_partitions,
            niter=self.cfg.kmeans_niters,
            gpu=True,
            verbose=True,
            seed=self.cfg.seed,
        )
        kmeans.train(sample)

        centroids = torch.from_numpy(kmeans.centroids)
        centroids = torch.nn.functional.normalize(centroids, dim=-1).half()
        del sample

        bucket_cutoffs, bucket_weights, avg_residual = self._compute_avg_residual(
            centroids, heldout_sample
        )

        self._log_info_main_thread_only(f"avg_residual = {avg_residual}")

        # Compute and save codec into avg_residual.pt, buckets.pt and centroids.pt
        codec = ResidualCodec(
            cfg=self.cfg,
            centroids=centroids,
            avg_residual=avg_residual,
            bucket_cutoffs=bucket_cutoffs,
            bucket_weights=bucket_weights,
        )
        codec.save(index_path=self.dir_path)

    def _saver_thread(self) -> None:
        for args in iter(self.saver_queue.get, None):
            self._write_chunk_to_disk(*args)

    def _write_chunk_to_disk(
        self,
        chunk_idx: int,
        offset: int,
        tok_ids: torch.Tensor,
        compressed_tok_embs: BaseResidualEmbeddings,
        tok_lens: List[int],
    ) -> None:
        path_prefix = os.path.join(self.dir_path, str(chunk_idx))
        compressed_tok_embs.save(path_prefix + "-tok")

        cls_lens_path = os.path.join(self.dir_path, f"cls_lens.{chunk_idx}.json")
        tok_lens_path = os.path.join(self.dir_path, f"tok_lens.{chunk_idx}.json")
        phrase_lens_path = os.path.join(self.dir_path, f"phrase_lens.{chunk_idx}.json")

        # Save the lengths of the embeddings
        with open(tok_lens_path, "w") as output_tok_lens:
            ujson.dump(tok_lens, output_tok_lens)

        # Save token ids
        tok_ids_path = os.path.join(self.dir_path, f"tok_ids.{chunk_idx}.pt")
        torch.save(tok_ids, tok_ids_path)

        metadata_path = os.path.join(self.dir_path, f"{chunk_idx}.metadata.json")
        with open(metadata_path, "w") as output_metadata:
            metadata = {
                "passage_offset": offset,
                "num_passages": len(tok_lens),
                "num_tok_embeddings": len(compressed_tok_embs),
            }
            ujson.dump(metadata, output_metadata)

    def save_codec(self, codec: ResidualCodec) -> None:
        codec.save(index_path=self.dir_path)

    def load_codec(self):
        return ResidualCodec.load(index_path=self.dir_path)

    def try_load_codec(self) -> bool:
        try:
            ResidualCodec.load(index_path=self.dir_path)
            return True
        except Exception as e:
            return False

    def check_chunk_exists(self, chunk_idx):
        # TODO: Verify that the chunk has the right amount of data?

        tok_lens_path = os.path.join(self.dir_path, f"tok_lens.{chunk_idx}.json")
        if not os.path.exists(tok_lens_path):
            return False

        metadata_path = os.path.join(self.dir_path, f"{chunk_idx}.metadata.json")
        if not os.path.exists(metadata_path):
            return False

        path_prefix = os.path.join(self.dir_path, str(chunk_idx))
        codes_path = f"{path_prefix}-tok.codes.pt"
        if not os.path.exists(codes_path):
            return False

        residuals_path = (
            f"{path_prefix}-tok.residuals.pt"  # f'{path_prefix}.residuals.bn'
        )
        if not os.path.exists(residuals_path):
            return False

        return True

    @contextmanager
    def thread(self) -> Generator:
        self.codec = self.load_codec()

        self.saver_queue = queue.Queue(maxsize=3)
        thread = threading.Thread(target=self._saver_thread)
        thread.start()

        try:
            yield

        finally:
            self.saver_queue.put(None)
            thread.join()

            del self.saver_queue
            del self.codec

    def save_chunk(
        self,
        chunk_idx: int,
        offset: int,
        tok_ids: torch.Tensor,
        tok_embs: torch.Tensor,
        tok_lens: List[int],
    ) -> None:
        compressed_tok_embs = self.codec.compress(tok_embs)

        self.saver_queue.put(
            (
                chunk_idx,
                offset,
                tok_ids,
                compressed_tok_embs,
                tok_lens,
            )
        )

    @abc.abstractmethod
    def _collect_embedding_id_offset(self, *args, **kwargs) -> Tuple:
        raise NotImplementedError

    @abc.abstractmethod
    def _build_ivf(self, *args, **kwargs) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def _update_metadata(self, *args, **kwargs) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def encoding(self) -> None:
        raise NotImplementedError

    @main_thread_only
    @abc.abstractmethod
    def finalize(self) -> None:
        raise NotImplementedError
