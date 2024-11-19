import logging
import os
from typing import *

import torch
import tqdm
import ujson
from omegaconf import DictConfig

from eagle.dataset.corpus import Document
from eagle.index.base_indexer import BaseIndexer, main_thread_only
from eagle.index.codecs.residual import ResidualCodec
from eagle.index.utils import flatten_items_with_mask, optimize_ivf

logger = logging.getLogger("EAGLEIndexer")


class EAGLEIndexer(BaseIndexer):
    def __init__(self, cfg: DictConfig, rank: int = 0, world_size: int = 1) -> None:
        super().__init__(cfg=cfg, rank=rank, world_size=world_size)

    def _collect_embedding_id_offset(
        self,
    ) -> Tuple[int, List[int], int, List[int], int, List[int]]:
        """
        Count the embedding offsets from the first chunk to the last chunk.
        This is done here because we asynchronously save the embeddings.
        """
        passage_offset = 0
        tok_embedding_offset = 0

        # Calculate embedding ids for each chunk
        tok_embedding_offsets: List[int] = []
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
                chunk_metadata["tok_embedding_offset"] = tok_embedding_offset

                # Update the offsets
                tok_embedding_offsets.append(tok_embedding_offset)

                passage_offset += chunk_metadata["num_passages"]
                tok_embedding_offset += chunk_metadata["num_tok_embeddings"]

            # Write the updated metadata back
            with open(metadata_path, "w") as f:
                f.write(ujson.dumps(chunk_metadata, indent=4) + "\n")

        assert (
            len(tok_embedding_offsets) == self.num_chunks
        ), f"{len(tok_embedding_offsets)} vs. {self.num_chunks}"

        return (
            tok_embedding_offset,
            tok_embedding_offsets,
        )

    def _sample_embeddings(self, sampled_pids: List[int]) -> Tuple[torch.Tensor, float]:
        """
        Encode sampled passages and save them temporarily.
        This is called on all the processes parallelly.
        """
        # Extract documents
        local_samples: List[Document] = []
        for pid, documents in self.corpus.enumerate(
            rank=self.rank, world_size=self.world_size
        ):
            if pid in sampled_pids:
                local_samples.append(documents)

        self._log_info(f"Encoding {len(local_samples)} sampled passages...")
        (
            local_sample_tok_ids,
            local_sample_tok_embs,
            local_sample_tok_masks,
        ) = self.model.encode_documents(
            local_samples,
            show_progress=self.is_main_thread,
            truncation=True,
        )
        # Flatten tok_embs
        local_sample_tok_ids, local_sample_tok_lens = flatten_items_with_mask(
            local_sample_tok_ids, local_sample_tok_masks
        )
        local_sample_tok_embs, _ = flatten_items_with_mask(
            local_sample_tok_embs, local_sample_tok_masks
        )

        if torch.cuda.is_available():
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                self.num_sample_embs = torch.tensor(
                    [local_sample_tok_embs.size(0)]
                ).cuda()
                torch.distributed.all_reduce(self.num_sample_embs)

                avg_doclen_est = (
                    sum(local_sample_tok_lens) / len(local_sample_tok_lens)
                    if local_sample_tok_lens
                    else 0
                )
                avg_doclen_est = torch.tensor([avg_doclen_est]).cuda()
                torch.distributed.all_reduce(avg_doclen_est)

                nonzero_ranks = torch.tensor(
                    [float(len(local_sample_tok_embs) > 0)]
                ).cuda()
                torch.distributed.all_reduce(nonzero_ranks)
            else:
                self.num_sample_embs = torch.tensor(
                    [local_sample_tok_embs.size(0)]
                ).cuda()

                avg_doclen_est = (
                    sum(local_sample_tok_lens) / len(local_sample_tok_lens)
                    if local_sample_tok_lens
                    else 0
                )
                avg_doclen_est = torch.tensor([avg_doclen_est]).cuda()

                nonzero_ranks = torch.tensor([float(len(local_samples) > 0)]).cuda()
        else:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                self.num_sample_embs = torch.tensor(
                    [local_sample_tok_embs.size(0)]
                ).cpu()
                torch.distributed.all_reduce(self.num_sample_embs)

                avg_doclen_est = (
                    sum(local_sample_tok_lens) / len(local_sample_tok_lens)
                    if local_sample_tok_lens
                    else 0
                )
                avg_doclen_est = torch.tensor([avg_doclen_est]).cpu()
                torch.distributed.all_reduce(avg_doclen_est)

                nonzero_ranks = torch.tensor([float(len(local_samples) > 0)]).cpu()
                torch.distributed.all_reduce(nonzero_ranks)
            else:
                self.num_sample_embs = torch.tensor(
                    [local_sample_tok_embs.size(0)]
                ).cpu()

                avg_doclen_est = (
                    sum(local_sample_tok_lens) / len(local_sample_tok_lens)
                    if local_sample_tok_lens
                    else 0
                )
                avg_doclen_est = torch.tensor([avg_doclen_est]).cpu()

                nonzero_ranks = torch.tensor([float(len(local_samples) > 0)]).cpu()

        avg_doclen_est = avg_doclen_est.item() / nonzero_ranks.item()
        self.avg_doclen_est = avg_doclen_est

        self._log_info_main_thread_only(
            f"avg_doclen_est = {avg_doclen_est} \t len(local_sample_tok_embs) = {len(local_sample_tok_embs):,}"
        )

        torch.save(
            local_sample_tok_embs.half(),
            os.path.join(self.dir_path, f"tok_sample.{self.rank}.pt"),
        )

        return avg_doclen_est

    def _build_ivf(
        self,
        num_tok_embeddings: int,
        tok_embedding_offsets: List[int],
    ) -> None:
        """"""
        # Maybe we should several small IVFs? Every 250M embeddings, so that's every 1 GB.
        # It would save *memory* here and *disk space* regarding the int64.
        # But we'd have to decide how many IVFs to use during retrieval: many (loop) or one?
        # A loop seems nice if we can find a size that's large enough for speed yet small enough to fit on GPU!
        # Then it would help nicely for batching later: 1GB.
        self._log_info("Building IVF...")
        tok_codes = torch.zeros(num_tok_embeddings, dtype=torch.long)

        self._log_info("Loading codes...")
        for chunk_idx in tqdm.tqdm(range(self.num_chunks)):
            # Get codes
            tok_chunk_codes = ResidualCodec.Embeddings.load_codes(
                self.dir_path, chunk_idx
            )
            # Handle token codes
            tok_offset = tok_embedding_offsets[chunk_idx]
            tok_codes[tok_offset : tok_offset + tok_chunk_codes.size(0)] = (
                tok_chunk_codes
            )
        assert tok_offset + tok_chunk_codes.size(0) == tok_codes.size(0), (
            tok_offset,
            tok_chunk_codes.size(0),
            tok_codes.size(),
        )
        self._log_info(f"Sorting codes...")

        # Handle token codes
        tok_codes = tok_codes.sort()
        tok_ivf, tok_values = tok_codes.indices, tok_codes.values
        self._log_info(f"Getting unique codes for tokens...")
        tok_ivf_lengths = torch.bincount(tok_values, minlength=self.num_partitions)
        assert tok_ivf_lengths.size(0) == self.num_partitions
        # Transforms centroid->embedding ivf to centroid->passage ivf
        _, _ = optimize_ivf(tok_ivf, tok_ivf_lengths, self.dir_path, granularity="tok")

    def _update_metadata(
        self,
        num_tok_embeddings: int,
    ) -> None:
        metadata_path = os.path.join(self.dir_path, "metadata.json")
        self._log_info(f"#> Saving the indexing metadata to {metadata_path} ...")

        with open(metadata_path, "w") as f:
            d = {"config": dict(self.cfg)}
            d["num_chunks"] = self.num_chunks
            d["num_partitions"] = self.num_partitions
            d["num_tok_embeddings"] = num_tok_embeddings
            d["avg_tok_len"] = num_tok_embeddings / len(self.corpus)

            f.write(ujson.dumps(d, indent=4) + "\n")

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
        with self.thread():
            for chunk_idx, offset, documents in tqdm.tqdm(
                self.corpus.enumerate_chunk(rank=self.rank, world_size=self.world_size),
                disable=not self.is_main_thread,
            ):
                # Encode passages into embeddings with the checkpoint model
                tok_ids, tok_embs, tok_masks = self.model.encode_documents(
                    documents,
                    show_progress=self.is_main_thread,
                    truncation=True,
                )
                # Flatten tok_embs
                tok_ids, tok_lens = flatten_items_with_mask(tok_ids, tok_masks)
                tok_embs, _ = flatten_items_with_mask(tok_embs, tok_masks)
                assert tok_embs.dtype == (
                    torch.float16 if self.use_gpu else torch.float32
                )
                tok_embs = tok_embs.half()
                self._log_info_main_thread_only(
                    f"#> Saving chunk {chunk_idx}: \t {len(documents):,} passages "
                    f"and {tok_embs.size(0):,} embeddings. From #{offset:,} onward."
                )

                self.save_chunk(
                    chunk_idx=chunk_idx,
                    offset=offset,
                    tok_ids=tok_ids,
                    tok_embs=tok_embs,
                    tok_lens=tok_lens,
                )  # offset = first passage index in chunk
                del tok_embs, tok_lens

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
            num_tok_embeddings,
            tok_embedding_offsets,
        ) = self._collect_embedding_id_offset()

        # build ivf and update the metadata
        self._build_ivf(
            num_tok_embeddings=num_tok_embeddings,
            tok_embedding_offsets=tok_embedding_offsets,
        )
        self._update_metadata(
            num_tok_embeddings=num_tok_embeddings,
        )
