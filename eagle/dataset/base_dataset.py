import logging
import os
from typing import *

import hkkang_utils.file as file_utils
import numpy as np
from omegaconf import DictConfig

from eagle.dataset.utils import (
    get_indices_to_avoid_repeated_qids_in_minibatch,
    get_labels,
    read_qrels_qids,
)
from eagle.tokenization.tokenizers import Tokenizers

logger = logging.getLogger("BaseDataset")


class BaseDataset:
    def __init__(
        self,
        cfg: DictConfig,
        cfg_dataset: DictConfig,
        tokenizers: Tokenizers,
        tokenized_queries: Dict,
        tokenized_corpus: Dict,
    ):
        self.cfg = cfg
        self.cfg_dataset = cfg_dataset
        self.tokenizers = tokenizers
        self.data = self._read_data(self.data_path)
        # Save cached information
        self.tokenized_queries = tokenized_queries
        self.tokenized_corpus = tokenized_corpus

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Any:
        # Go through proxy indices for shuffling
        assert idx < len(self), f"Index {idx} out of range {len(self)}"
        target_idx = self.indices[idx]
        return self.data[target_idx]

    @property
    def indices(self) -> List[int]:
        if not hasattr(self, "_indices"):
            self._indices = list(range(len(self)))
        return self._indices

    @property
    def data_path(self) -> str:
        return os.path.join(
            self.cfg_dataset.dir_path, self.cfg_dataset.name, self.cfg.data_file
        )

    @property
    def nway(self) -> int:
        return self.cfg.nway

    @property
    def neg_num(self) -> int:
        return self.nway - 1

    @property
    def neg_start_idx(self) -> int:
        return self.cfg.negative_start_offset

    @property
    def neg_end_idx(self) -> int:
        return self.cfg.negative_start_offset + self.neg_num

    @property
    def labels(self) -> np.array:
        return get_labels(bsize=1, neg_num=self.neg_num).squeeze(0)

    @property
    def tokenized_queries_key_type(self) -> type:
        if len(self.tokenized_queries) == 0:
            return None
        if not hasattr(self, "_tokenized_queries_key_type"):
            self._tokenized_queries_key_type = type(
                list(self.tokenized_queries.keys())[0]
            )
        return self._tokenized_queries_key_type

    @property
    def tokenized_corpus_key_type(self) -> type:
        if len(self.tokenized_corpus) == 0:
            return None
        if not hasattr(self, "_tokenized_corpus_key_type"):
            self._tokenized_corpus_key_type = type(
                list(self.tokenized_corpus.keys())[0]
            )
        return self._tokenized_corpus_key_type

    def _read_data(self, path: str) -> List[List[int]]:
        data: List = file_utils.read_json_file(path, auto_detect_extension=True)
        # Sample data if needed
        sample_size = (
            self.cfg.debug_sample_size if self.cfg.is_debug else self.cfg.sample_size
        )
        if sample_size > 0:
            data = data[:sample_size]
        return data

    def shuffle_indices_to_avoid_qid_repetition(
        self, qrels_path: str, bsize: int
    ) -> None:
        """Make sure that there are no repeated qids in the batch.
        So that in-batch negatives are valid."""
        # Load training qrels to avoid loading the entire training dataset (instead, we stream it during training)
        train_qids: List[str] = read_qrels_qids(qrels_path)
        # Check the loaded data is valid
        assert len(train_qids) == len(
            self.data
        ), f"len(qids)={len(train_qids)}, len(train_dataset)={len(self.data)}"

        # Get new indices to avoid repeated qids in the mini-batch
        indices: List[int] = get_indices_to_avoid_repeated_qids_in_minibatch(
            train_qids, bsize
        )
        assert len(indices) == len(
            train_qids
        ), f"len(indices)={len(indices)}, len(train_qids)={len(train_qids)}, bsize={bsize}"

        # Check if there are repeated qids in the mini-batch
        repeated_mini_batch_indices = []
        for i in range(0, len(train_qids), bsize):
            if i + bsize > len(train_qids):
                break
            qids = [train_qids[indices[j]] for j in range(i, i + bsize)]
            if len(qids) != len(set(qids)):
                repeated_mini_batch_indices.append(i)
        if repeated_mini_batch_indices:
            logger.info(
                f"Num of mini-batch with repeated qids: {len(repeated_mini_batch_indices)}/{len(train_qids)//bsize}."
            )
            assert (
                len(repeated_mini_batch_indices) < 10
            ), f"len(repeated_mini_batch_indices)={len(repeated_mini_batch_indices)}"
        else:
            logger.info(f"No repeated qids in the mini-batch.")

        self._indices = indices
