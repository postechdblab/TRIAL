import logging
import os
import random
from typing import *

import faiss
import hkkang_utils.file as file_utils
import hkkang_utils.list as list_utils
import numpy as np
import torch
import torch.multiprocessing as mp
import tqdm
import ujson
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP

from eagle.dataset.utils import get_mask
from eagle.index.codecs.residual import ResidualCodec
from eagle.index.corpus import Corpus
from eagle.index.index_saver import IndexSaver
from eagle.index.utils import all_gather_nd, optimize_ivf
from eagle.model.utils import _sort_by_length, _split_into_batches
from eagle.tokenizer import Tokenizers
from eagle.utils import add_global_configs

TYPICAL_DOCLEN = 120
HELDOUT_FRACTION = 0.05
MIN_HELDOUT_NUM = 50_000

logger = logging.getLogger("Indexer")


# Decorator to check self.is_main_thread and call the function only when it is true
def main_thread_only(func):
    def wrapper(self, *args, **kwargs):
        if self.is_main_thread:
            return func(self, *args, **kwargs)
        else:
            return None

    return wrapper


class Indexer:
    def __init__(self, cfg: DictConfig, rank: int = 0, world_size: int = 1) -> None:
        self.cfg = cfg.indexing
        self.rank = rank
        self.world_size = world_size
        self.corpus: Corpus = Corpus(cfg)
        self.tokenizers = Tokenizers(cfg.q_tokenizer, cfg.d_tokenizer, cfg.model.name)
        # Set model
        assert cfg.model.ckpt_path, "model ckpt_path is not provided."
        # TODO: Placing the import here to avoid circular import. Need to fix this
        from eagle.model.late_interaction import EAGLE

        self.model = DDP(
            EAGLE(cfg=cfg.model, tokenizers=self.tokenizers).to(self.device),
            device_ids=[self.rank],
        )
        self.saver = IndexSaver(cfg=cfg, dir_path=self.dir_path)
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
        sample_embs = self.plan()

        # Train kmeans
        self.train_kmeans(sample_embs)
        self._distributed_barrier()

        # Encode all documents
        self.encoding()
        self._distributed_barrier()

        # Finalize
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
            exit(0)
        else:
            # Create direcrtory if not exists
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

    def _sample_embeddings(self, sampled_pids: List[int]) -> Tuple[torch.Tensor, float]:
        # Extract documents
        local_samples: List[str] = [
            str(passage)
            for pid, passage in self.corpus.enumerate(
                rank=self.rank, world_size=self.world_size
            )
            if pid in sampled_pids
        ]

        local_sample_embs, doclens = self.encode_passages(
            local_samples, show_progress=self.is_main_thread
        )

        # Compute average document length estimattion
        avg_doclen_est = sum(doclens) / len(doclens) if doclens else 0
        avg_doclen_est = torch.tensor([avg_doclen_est], device=local_sample_embs.device)
        torch.distributed.all_reduce(avg_doclen_est)
        nonzero_ranks = torch.tensor(
            [float(len(local_samples) > 0)], device=avg_doclen_est.device
        )
        torch.distributed.all_reduce(nonzero_ranks)
        avg_doclen_est = avg_doclen_est.item() / nonzero_ranks.item()
        self.avg_doclen_est = avg_doclen_est

        # Gather all samples
        sample_embs = all_gather_nd(local_sample_embs, world_size=self.world_size)
        if self.is_main_thread:
            sample_embs = torch.cat(sample_embs).cpu()
        else:
            sample_embs = None

        return sample_embs, self.avg_doclen_est

    def _check_all_files_are_saved(self) -> None:
        """Check all chunks are created"""
        self._log_info("Checking all files are saved...")
        check_results = [
            self.saver.check_chunk_exists(chunk_idx)
            for chunk_idx in range(self.num_chunks)
        ]
        if all(check_results):
            self._log_info("All files are saved!")
        else:
            raise ValueError(f"Some files are missing! (check_results:{check_results})")
        return None

    def _collect_embedding_id_offset(self) -> Tuple[int, List[int]]:
        """
        Count the embedding offsets from the first chunk to the last chunk.
        This is done here because we asynchronously save the embeddings.
        """
        passage_offset = 0
        embedding_offset = 0

        # Calculate embedding ids for each chunk
        embedding_offsets: List[int] = []
        for chunk_idx in range(self.num_chunks):
            # Get metadata path
            metadata_path = os.path.join(self.dir_path, f"{chunk_idx}.metadata.json")
            # Read metadata for the chunk
            with open(metadata_path) as f:
                chunk_metadata = ujson.load(f)

                # Check the values
                assert chunk_metadata["passage_offset"] == passage_offset, (
                    chunk_idx,
                    passage_offset,
                    chunk_metadata,
                )

                # Add the embedding offset
                chunk_metadata["embedding_offset"] = embedding_offset

                # Update the offsets
                embedding_offsets.append(embedding_offset)
                passage_offset += chunk_metadata["num_passages"]
                embedding_offset += chunk_metadata["num_embeddings"]

            # Write the updated metadata back
            with open(metadata_path, "w") as f:
                f.write(ujson.dumps(chunk_metadata, indent=4) + "\n")

        assert (
            len(embedding_offsets) == self.num_chunks
        ), f"{len(embedding_offsets)} vs. {self.num_chunks}"

        return embedding_offset, embedding_offsets

    def _build_ivf(self, num_embeddings: int, embedding_offsets: List[int]) -> None:
        """"""
        # Maybe we should several small IVFs? Every 250M embeddings, so that's every 1 GB.
        # It would save *memory* here and *disk space* regarding the int64.
        # But we'd have to decide how many IVFs to use during retrieval: many (loop) or one?
        # A loop seems nice if we can find a size that's large enough for speed yet small enough to fit on GPU!
        # Then it would help nicely for batching later: 1GB.

        self._log_info("Building IVF...")
        codes = torch.zeros(num_embeddings, dtype=torch.long)

        self._log_info("Loading codes...")
        for chunk_idx in tqdm.tqdm(range(self.num_chunks)):
            offset = embedding_offsets[chunk_idx]
            chunk_codes = ResidualCodec.Embeddings.load_codes(self.dir_path, chunk_idx)
            codes[offset : offset + chunk_codes.size(0)] = chunk_codes

        assert offset + chunk_codes.size(0) == codes.size(0), (
            offset,
            chunk_codes.size(0),
            codes.size(),
        )
        self._log_info(f"Sorting codes...")

        codes = codes.sort()
        ivf, values = codes.indices, codes.values

        self._log_info(f"Getting unique codes...")
        ivf_lengths = torch.bincount(values, minlength=self.num_partitions)
        assert ivf_lengths.size(0) == self.num_partitions

        # Transforms centroid->embedding ivf to centroid->passage ivf
        _, _ = optimize_ivf(ivf, ivf_lengths, self.dir_path)

    def _update_metadata(self, num_embeddings: int) -> None:
        metadata_path = os.path.join(self.dir_path, "metadata.json")
        self._log_info(f"#> Saving the indexing metadata to {metadata_path} ...")

        with open(metadata_path, "w") as f:
            d = {"config": dict(self.cfg)}
            d["num_chunks"] = self.num_chunks
            d["num_partitions"] = self.num_partitions
            d["num_embeddings"] = num_embeddings
            d["avg_doclen"] = num_embeddings / len(self.corpus)

            f.write(ujson.dumps(d, indent=4) + "\n")

    def plan(self) -> torch.Tensor:
        """Calcuate and saves plan.json for the whole corpus."""
        sample_pids = self._sample_pids()
        sample_embs, avg_doclen_est = self._sample_embeddings(sample_pids)

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

        return sample_embs

    @main_thread_only
    def train_kmeans(self, sample: torch.Tensor) -> None:
        # Shuffle and split out a 5% "heldout" sub-sample [up to 50k elements]
        sample_num = sample.size(0)
        sample = sample[torch.randperm(sample_num)]

        # Shuffle and split out a 5% "heldout" sub-sample [up to 50k elements]
        heldout_num = self._get_kmeans_heldout_size(sample_num)
        sample, heldout_sample = sample.split(
            [sample_num - heldout_num, heldout_num], dim=0
        )
        sample = sample[: sample_num - heldout_num].numpy()

        assert torch.cuda.is_available(), "CUDA is not available."
        torch.cuda.empty_cache()

        # START
        kmeans = faiss.Kmeans(
            d=self.cfg.dim,
            k=self.num_partitions,
            niter=self.cfg.kmeans_niters,
            gpu=True,
            verbose=True,
            seed=123,
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

    def encoding(self) -> None:
        """
        Encode embeddings for all passages in the corpus.
        Each embedding is converted to code (centroid id) and residual.
        Embeddings stored according to passage order in contiguous chunks of memory.

        Saved data files described below:
            {CHUNK#}.codes.pt:      centroid id for each embedding in chunk
            {CHUNK#}.residuals.pt:  16-bits residual for each embedding in chunk
            doclens.{CHUNK#}.pt:    number of embeddings within each passage in chunk
        """
        self._log_info_main_thread_only("Encoding documents...")
        with self.saver.thread():
            for chunk_idx, offset, passages in tqdm.tqdm(
                self.corpus.enumerate_chunk(rank=self.rank, world_size=self.world_size),
                disable=not self.is_main_thread,
            ):
                # Convert Document to string
                passages: List[str] = [str(passage) for passage in passages]
                # Encode passages into embeddings with the checkpoint model
                embs, doclens = self.encode_passages(
                    passages, show_progress=self.is_main_thread
                )
                assert embs.dtype == (torch.float16 if self.use_gpu else torch.float32)
                embs = embs.half()
                self._log_info_main_thread_only(
                    f"#> Saving chunk {chunk_idx}: \t {len(passages):,} passages "
                    f"and {embs.size(0):,} embeddings. From #{offset:,} onward."
                )

                self.saver.save_chunk(
                    chunk_idx, offset, embs, doclens
                )  # offset = first passage index in chunk
                del embs, doclens

    @main_thread_only
    def finalize(self) -> None:
        """
        Aggregates and stores metadata for each chunk and the whole index
        Builds and saves inverse mapping from centroids to passage IDs

        Saved data files described below:
            {CHUNK#}.metadata.json: [ passage_offset, num_passages, num_embeddings, embedding_offset ]
            metadata.json: [ num_chunks, num_partitions, num_embeddings, avg_doclen ]
            inv.pid.pt: [ ivf, ivf_lengths ]
                ivf is an array of passage IDs for centroids 0, 1, ...
                ivf_length contains the number of passage IDs for each centroid
        """
        # Check encoding is done for all chunks
        self._check_all_files_are_saved()

        # Get the total number and offsets of embeddings
        num_embeddings, embedding_offsets = self._collect_embedding_id_offset()

        # build ivf and update the metadata
        self._build_ivf(
            num_embeddings=num_embeddings, embedding_offsets=embedding_offsets
        )
        self._update_metadata(num_embeddings=num_embeddings)

    def encode_passages(
        self, passages: List[str], show_progress: bool = False
    ) -> Tuple[torch.Tensor, List[int]]:
        with torch.inference_mode():
            all_embs, all_doclens = [], []
            for passages_batch in list_utils.chunks(
                passages, chunk_size=self.cfg.bsize, show_progress=show_progress
            ):
                # Tokenize given texts
                result = self.tokenizers.d_tokenizer(
                    passages_batch, padding=True, return_tensors="pt"
                )
                ids, att_mask = result["input_ids"], result["attention_mask"]
                ids, att_mask, reverse_indices = _sort_by_length(
                    ids, att_mask, self.cfg.bsize
                )

                # Create mask
                # TODO: Need to align this with the trained model
                tok_mask = get_mask(
                    input_ids=ids, skip_ids=self.tokenizers.d_tokenizer.special_toks_ids
                ).unsqueeze(-1)

                # Create batch
                text_batches = _split_into_batches(
                    ids, att_mask, tok_mask, bsize=self.cfg.bsize
                )

                # Encode
                result_batches = [
                    self.model.module.encode_d_text(
                        tok_ids=input_ids.to(self.device),
                        att_mask=attention_mask.to(self.device),
                        is_encoding=True,
                    )
                    for input_ids, attention_mask, token_mask in text_batches
                ]

                # Flatten
                D, mask = [], []
                for i in range(len(result_batches)):
                    D_ = result_batches[i][1].half()
                    mask_ = text_batches[i][2].bool()
                    D.append(D_)
                    mask.append(mask_)

                D, mask = (
                    torch.cat(D)[reverse_indices],
                    torch.cat(mask)[reverse_indices],
                )
                doclens = mask.squeeze(-1).sum(-1).tolist()

                # Serialize and remove the masked tokens
                D = D.view(-1, D.shape[-1])
                D = D[mask.bool().flatten()]

                # Check if flatten is correct
                assert len(D) == sum(
                    doclens
                ), f"len(D)={len(D)} != sum(doclens)={sum(doclens)}"

                all_embs.append(D)
                all_doclens.extend(doclens)

            all_embs = torch.cat(all_embs)

        return all_embs, all_doclens


import hydra


def multi_process_indexing(rank: int, cfg: DictConfig, world_size: int) -> None:
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    # Initialize the process group
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.distributed.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo",
        init_method="env://",
        world_size=world_size,
    )
    # Set default device
    torch.cuda.set_device(rank)
    # Create indexer
    indexer = Indexer(cfg=cfg, rank=rank, world_size=world_size)
    indexer()


@hydra.main(version_base=None, config_path="/root/EAGLE/config", config_name="config")
def _main(cfg: DictConfig) -> None:
    from omegaconf import open_dict

    cfg: DictConfig = add_global_configs(cfg, exclude_keys=["args"])
    with open_dict(cfg):
        cfg.model.ckpt_path = (
            "/root/EAGLE/runs/new_new_new_qdweight_nway32_bsize32_distill3/last.ckpt"
        )
        cfg.dataset.name = "beir-arguana"
        cfg.indexing.path = f"/root/EAGLE/index/"
    world_size = 2  # torch.cuda.device_count()
    mp.spawn(
        multi_process_indexing,
        args=(cfg, world_size),
        nprocs=world_size,
        join=True,
    )
    print("Done")


if __name__ == "__main__":
    _main()
