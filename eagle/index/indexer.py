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

from eagle.dataset.utils import (
    combine_word_phrase_ranges,
    convert_range_for_scatter,
    fill_ranges,
    get_mask,
)
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
        local_samples: List[str] = []
        local_samples_word_ranges: List[List[Tuple[int, int]]] = []
        local_samples_phrase_ranges: List[List[Tuple[int, int]]] = []
        for pid, passage, word_ranges, phrase_ranges in self.corpus.enumerate(
            rank=self.rank, world_size=self.world_size
        ):
            if pid in sampled_pids:
                local_samples.append(str(passage))
                local_samples_word_ranges.append(word_ranges)
                local_samples_phrase_ranges.append(phrase_ranges)

        (
            local_sample_cls_embs,
            local_sample_tok_embs,
            local_sample_phrase_embs,
            tok_lens,
            phrase_lens,
        ) = self.encode_passages(
            local_samples,
            word_ranges=local_samples_word_ranges,
            phrase_ranges=local_samples_phrase_ranges,
            show_progress=self.is_main_thread,
        )

        # Compute average document length estimattion
        avg_doclen_est = sum(tok_lens) / len(tok_lens) if tok_lens else 0
        avg_doclen_est = torch.tensor(
            [avg_doclen_est], device=local_sample_tok_embs.device
        )
        torch.distributed.all_reduce(avg_doclen_est)
        nonzero_ranks = torch.tensor(
            [float(len(local_samples) > 0)], device=avg_doclen_est.device
        )
        torch.distributed.all_reduce(nonzero_ranks)
        avg_doclen_est = avg_doclen_est.item() / nonzero_ranks.item()
        self.avg_doclen_est = avg_doclen_est

        # Gather all samples
        sample_embs = all_gather_nd(local_sample_tok_embs, world_size=self.world_size)
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

    def _collect_embedding_id_offset(
        self,
    ) -> Tuple[int, List[int], int, List[int], int, List[int]]:
        """
        Count the embedding offsets from the first chunk to the last chunk.
        This is done here because we asynchronously save the embeddings.
        """
        passage_offset = 0
        cls_embedding_offset = 0
        tok_embedding_offset = 0
        phrase_embedding_offset = 0

        # Calculate embedding ids for each chunk
        cls_embedding_offsets: List[int] = []
        tok_embedding_offsets: List[int] = []
        phrase_embedding_offsets: List[int] = []
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

                # Add the embedding offsets
                chunk_metadata["cls_embedding_offset"] = cls_embedding_offset
                chunk_metadata["tok_embedding_offset"] = tok_embedding_offset
                chunk_metadata["phrase_embedding_offset"] = phrase_embedding_offset

                # Update the offsets
                cls_embedding_offsets.append(cls_embedding_offset)
                tok_embedding_offsets.append(tok_embedding_offset)
                phrase_embedding_offsets.append(phrase_embedding_offset)

                passage_offset += chunk_metadata["num_passages"]
                cls_embedding_offset += chunk_metadata["num_cls_embeddings"]
                tok_embedding_offset += chunk_metadata["num_tok_embeddings"]
                phrase_embedding_offset += chunk_metadata["num_phrase_embeddings"]

            # Write the updated metadata back
            with open(metadata_path, "w") as f:
                f.write(ujson.dumps(chunk_metadata, indent=4) + "\n")

        assert (
            len(tok_embedding_offsets) == self.num_chunks
        ), f"{len(tok_embedding_offsets)} vs. {self.num_chunks}"

        return (
            cls_embedding_offset,
            cls_embedding_offsets,
            tok_embedding_offset,
            tok_embedding_offsets,
            phrase_embedding_offset,
            phrase_embedding_offsets,
        )

    def _build_ivf(
        self,
        num_cls_embeddings: int,
        cls_embedding_offsets: List[int],
        num_tok_embeddings: int,
        tok_embedding_offsets: List[int],
        num_phrase_embeddings: int,
        phrase_embedding_offsets: List[int],
    ) -> None:
        """"""
        # Maybe we should several small IVFs? Every 250M embeddings, so that's every 1 GB.
        # It would save *memory* here and *disk space* regarding the int64.
        # But we'd have to decide how many IVFs to use during retrieval: many (loop) or one?
        # A loop seems nice if we can find a size that's large enough for speed yet small enough to fit on GPU!
        # Then it would help nicely for batching later: 1GB.
        use_cls_embed = num_cls_embeddings > 0
        use_phrase_embed = num_phrase_embeddings > 0
        self._log_info("Building IVF...")
        cls_codes = (
            torch.zeros(num_cls_embeddings, dtype=torch.long)
            if num_cls_embeddings
            else None
        )
        tok_codes = torch.zeros(num_tok_embeddings, dtype=torch.long)
        phrase_codes = (
            torch.zeros(num_phrase_embeddings, dtype=torch.long)
            if num_phrase_embeddings
            else None
        )

        self._log_info("Loading codes...")
        for chunk_idx in tqdm.tqdm(range(self.num_chunks)):
            # Get codes
            cls_chunk_codes, tok_chunk_codes, phrase_chunk_codes = (
                ResidualCodec.Embeddings.load_codes(self.dir_path, chunk_idx)
            )
            # Handle token codes
            tok_offset = tok_embedding_offsets[chunk_idx]
            tok_codes[tok_offset : tok_offset + tok_chunk_codes.size(0)] = (
                tok_chunk_codes
            )
            # Handle cls codes
            if use_cls_embed:
                cls_offset = cls_embedding_offsets[chunk_idx]
                cls_codes[cls_offset : cls_offset + cls_chunk_codes.size(0)] = (
                    cls_chunk_codes
                )
            # Handle phrase codes
            if use_phrase_embed:
                phrase_offset = phrase_embedding_offsets[chunk_idx]
                phrase_codes[
                    phrase_offset : phrase_offset + phrase_chunk_codes.size(0)
                ] = phrase_chunk_codes

        assert tok_offset + tok_chunk_codes.size(0) == tok_codes.size(0), (
            tok_offset,
            tok_chunk_codes.size(0),
            tok_codes.size(),
        )
        if use_cls_embed:
            assert cls_offset + cls_chunk_codes.size(0) == cls_codes.size(0), (
                cls_offset,
                cls_chunk_codes.size(0),
                cls_codes.size(),
            )
        if use_phrase_embed:
            assert phrase_offset + phrase_chunk_codes.size(0) == phrase_codes.size(0), (
                phrase_offset,
                phrase_chunk_codes.size(0),
                phrase_codes.size(),
            )
        self._log_info(f"Sorting codes...")

        # Handle token codes
        tok_codes = tok_codes.sort()
        tok_ivf, tok_values = tok_codes.indices, tok_codes.values
        ## Handle phrase codes
        self._log_info(f"Getting unique codes for tokens...")
        tok_ivf_lengths = torch.bincount(tok_values, minlength=self.num_partitions)
        assert tok_ivf_lengths.size(0) == self.num_partitions
        # Transforms centroid->embedding ivf to centroid->passage ivf
        _, _ = optimize_ivf(tok_ivf, tok_ivf_lengths, self.dir_path, granularity="tok")

        # Handle cls codes
        if use_cls_embed:
            cls_codes = cls_codes.sort()
            cls_ivf, cls_values = cls_codes.indices, cls_codes.values
            ## Handle phrase codes
            self._log_info(f"Getting unique codes for tokens...")
            cls_ivf_lengths = torch.bincount(cls_values, minlength=self.num_partitions)
            assert cls_ivf_lengths.size(0) == self.num_partitions
            ## Transforms centroid->embedding ivf to centroid->passage ivf
            _, _ = optimize_ivf(
                cls_ivf, cls_ivf_lengths, self.dir_path, granularity="cls"
            )

        # Handle phrase codes
        if use_phrase_embed:
            phrase_codes = phrase_codes.sort()
            phrase_ivf, phrase_values = phrase_codes.indices, phrase_codes.values
            ## Handle phrase codes
            self._log_info(f"Getting unique codes for tokens...")
            phrase_ivf_lengths = torch.bincount(
                phrase_values, minlength=self.num_partitions
            )
            assert phrase_ivf_lengths.size(0) == self.num_partitions
            ## Transforms centroid->embedding ivf to centroid->passage ivf
            _, _ = optimize_ivf(
                phrase_ivf, phrase_ivf_lengths, self.dir_path, granularity="phrase"
            )

    def _update_metadata(
        self,
        num_cls_embeddings: int,
        num_tok_embeddings: int,
        num_phrase_embeddings: int,
    ) -> None:
        metadata_path = os.path.join(self.dir_path, "metadata.json")
        self._log_info(f"#> Saving the indexing metadata to {metadata_path} ...")

        with open(metadata_path, "w") as f:
            d = {"config": dict(self.cfg)}
            d["num_chunks"] = self.num_chunks
            d["num_partitions"] = self.num_partitions
            d["num_cls_embeddings"] = num_cls_embeddings
            d["num_tok_embeddings"] = num_tok_embeddings
            d["num_phrase_embeddings"] = num_phrase_embeddings
            d["avg_tok_len"] = num_tok_embeddings / len(self.corpus)
            d["avg_phrase_len"] = num_phrase_embeddings / len(self.corpus)

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
            for chunk_idx, offset, passages, word_ranges, phrase_ranges in tqdm.tqdm(
                self.corpus.enumerate_chunk(rank=self.rank, world_size=self.world_size),
                disable=not self.is_main_thread,
            ):
                # Convert Document to string
                passages: List[str] = [str(passage) for passage in passages]
                # Encode passages into embeddings with the checkpoint model
                cls_embs, tok_embs, phrase_embs, tok_lens, phrase_lens = (
                    self.encode_passages(
                        passages,
                        word_ranges=word_ranges,
                        phrase_ranges=phrase_ranges,
                        show_progress=self.is_main_thread,
                    )
                )
                assert tok_embs.dtype == (
                    torch.float16 if self.use_gpu else torch.float32
                )
                if cls_embs is not None:
                    cls_embs = cls_embs.half()
                tok_embs = tok_embs.half()
                if phrase_embs is not None:
                    phrase_embs = phrase_embs.half()
                self._log_info_main_thread_only(
                    f"#> Saving chunk {chunk_idx}: \t {len(passages):,} passages "
                    f"and {tok_embs.size(0):,} embeddings. From #{offset:,} onward."
                )

                self.saver.save_chunk(
                    chunk_idx=chunk_idx,
                    offset=offset,
                    cls_embs=cls_embs,
                    tok_embs=tok_embs,
                    phrase_embs=phrase_embs,
                    tok_lens=tok_lens,
                    phrase_lens=phrase_lens,
                )  # offset = first passage index in chunk
                del cls_embs, tok_embs, phrase_embs, tok_lens, phrase_lens

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
        (
            num_cls_embeddings,
            cls_embedding_offsets,
            num_tok_embeddings,
            tok_embedding_offsets,
            num_phrase_embeddings,
            phrase_embedding_offsets,
        ) = self._collect_embedding_id_offset()

        # build ivf and update the metadata
        self._build_ivf(
            num_cls_embeddings=num_cls_embeddings,
            cls_embedding_offsets=cls_embedding_offsets,
            num_tok_embeddings=num_tok_embeddings,
            tok_embedding_offsets=tok_embedding_offsets,
            num_phrase_embeddings=num_phrase_embeddings,
            phrase_embedding_offsets=phrase_embedding_offsets,
        )
        self._update_metadata(
            num_cls_embeddings=num_cls_embeddings,
            num_tok_embeddings=num_tok_embeddings,
            num_phrase_embeddings=num_phrase_embeddings,
        )

    def encode_passages(
        self,
        passages: List[str],
        word_ranges: List[List[Tuple[int, int]]] = None,
        phrase_ranges: List[List[Tuple[int, int]]] = None,
        show_progress: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        with torch.inference_mode():
            (
                all_cls_embs,
                all_tok_embs,
                all_phrase_embs,
                all_tok_lens,
                all_phrase_lens,
            ) = ([], [], [], [], [])
            indices = list(range(len(passages)))
            for indices_batch in list_utils.chunks(
                indices, chunk_size=self.cfg.bsize, show_progress=show_progress
            ):
                # Get batch inputs
                passages_batch = [passages[i] for i in indices_batch]
                word_ranges_batch = (
                    [word_ranges[i] for i in indices_batch] if word_ranges else None
                )
                phrase_ranges_batch = (
                    [phrase_ranges[i] for i in indices_batch] if phrase_ranges else None
                )

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

                # Select scatter indices
                if self.model.module.granularity_level in ["word"]:
                    scatter_indices = []
                    for i, word_ranges_batch_item in enumerate(word_ranges_batch):
                        # Cutoff my max tokenized length
                        max_len = ids[i].size(0)
                        word_ranges_batch_item = [
                            (start, end)
                            for start, end in word_ranges_batch_item
                            if end <= max_len
                        ]
                        ranges = fill_ranges(
                            word_ranges_batch_item,
                            max_len=max_len,
                        )
                        indices = convert_range_for_scatter(ranges)
                        scatter_indices.append(
                            torch.tensor(indices, device=self.device)
                        )
                    scatter_indices = torch.stack(scatter_indices)
                elif self.model.module.granularity_level in ["phrase", "multi"]:
                    scatter_indices = []
                    for i, (
                        word_ranges_batch_item,
                        phrase_ranges_batch_item,
                    ) in enumerate(zip(word_ranges_batch, phrase_ranges_batch)):
                        # Cutoff my max tokenized length
                        max_len = ids[i].size(0)
                        word_ranges_batch_item = [
                            (start, end)
                            for start, end in word_ranges_batch_item
                            if end <= max_len
                        ]
                        phrase_ranges_batch_item = [
                            (start, end)
                            for start, end in phrase_ranges_batch_item
                            if end <= max_len
                        ]
                        ranges = fill_ranges(
                            combine_word_phrase_ranges(
                                word_ranges_batch_item, phrase_ranges_batch_item
                            ),
                            max_len=max_len,
                        )
                        indices = convert_range_for_scatter(ranges)
                        scatter_indices.append(
                            torch.tensor(indices, device=self.device)
                        )
                    scatter_indices = torch.stack(scatter_indices)
                else:
                    scatter_indices = None

                # Create batch
                text_batches = _split_into_batches(
                    ids, att_mask, tok_mask, scatter_indices, bsize=self.cfg.bsize
                )

                # Encode
                result_batches = [
                    self.model.module.encode_d_text(
                        tok_ids=input_ids.to(self.device),
                        att_mask=attention_mask.to(self.device),
                        scatter_indices=scatter_indices_item,
                        is_encoding=True,
                    )
                    for input_ids, attention_mask, token_mask, scatter_indices_item in text_batches
                ]

                # Flatten
                D_cls, D_tok, D_phrase, mask = [], [], [], []
                for i in range(len(result_batches)):
                    if result_batches[i][0] is not None:
                        D_cls_ = result_batches[i][0].cpu().half()
                        D_cls.append(D_cls_)
                    if result_batches[i][2] is not None:
                        D_phrase_ = result_batches[i][2].cpu().half()
                        D_phrase.append(D_phrase_)

                    D_tok_ = result_batches[i][1].cpu().half()
                    mask_ = text_batches[i][2].cpu().bool()
                    D_tok.append(D_tok_)
                    mask.append(mask_)

                if len(D_cls) > 0:
                    D_cls = torch.cat(D_cls)[reverse_indices]
                if len(D_phrase) > 0:
                    D_phrase = torch.cat(D_phrase)[reverse_indices]
                D_tok, mask = (
                    torch.cat(D_tok)[reverse_indices],
                    torch.cat(mask)[reverse_indices],
                )
                tok_lens = mask.squeeze(-1).sum(-1).tolist()

                if len(D_cls) > 0:
                    D_cls = D_cls.view(-1, D_cls.shape[-1])

                if len(D_phrase) > 0:
                    phrase_lens = []
                    new_D_phrase = []
                    i = 0
                    for _, _, _, scatter_indices_item in text_batches:
                        for indices in scatter_indices_item:
                            phrase_len = indices[-1].item() + 1
                            phrase_lens.append(phrase_len)
                            new_D_phrase.append(D_phrase[i][:phrase_len])
                            i += 1
                    D_phrase = torch.cat(new_D_phrase)
                    assert len(tok_lens) == len(
                        phrase_lens
                    ), f"{len(tok_lens)} vs.{len(phrase_lens)}"
                    D_phrase = D_phrase.view(-1, D_phrase.shape[-1])

                # Serialize and remove the masked tokens
                D_tok = D_tok.view(-1, D_tok.shape[-1])
                D_tok = D_tok[mask.bool().flatten()]

                # Check if flatten is correct
                assert len(D_tok) == sum(
                    tok_lens
                ), f"len(D)={len(D_tok)} != sum(tok_lens)={sum(tok_lens)}"

                if len(D_cls) > 0:
                    all_cls_embs.append(D_cls)

                if len(D_phrase) > 0:
                    all_phrase_embs.append(D_phrase)
                    all_phrase_lens.extend(phrase_lens)

                all_tok_embs.append(D_tok)
                all_tok_lens.extend(tok_lens)

            if len(all_cls_embs) > 0:
                all_cls_embs = torch.cat(all_cls_embs)
            else:
                all_cls_embs = None

            if len(all_phrase_embs) > 0:
                all_phrase_embs = torch.cat(all_phrase_embs)
            else:
                all_phrase_embs = None

            all_tok_embs = torch.cat(all_tok_embs)

        return (
            all_cls_embs,
            all_tok_embs,
            all_phrase_embs,
            all_tok_lens,
            all_phrase_lens,
        )


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
