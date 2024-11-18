import logging
from typing import *

import torch
from omegaconf import DictConfig

from eagle.dataset.base_dataset import BaseDataset
from eagle.dataset.utils import (
    extract_pids_from_msmarco_data,
    extract_pids_from_non_msmarco_data,
    extract_qids_from_msmarco_data,
    extract_qids_from_non_msmarco_data,
)
from eagle.tokenization.tokenizers import Tokenizers
from eagle.tokenization.utils import combine_splitted_tok_ids

logger = logging.getLogger("InferenceDataset")


class InferenceDataset(BaseDataset):
    def __init__(
        self,
        cfg: DictConfig,
        cfg_dataset: DictConfig,
        tokenizers: Tokenizers,
        tokenized_queries: Dict,
        tokenized_corpus: Dict,
    ):
        super().__init__(
            cfg=cfg,
            cfg_dataset=cfg_dataset,
            tokenizers=tokenizers,
            tokenized_queries=tokenized_queries,
            tokenized_corpus=tokenized_corpus,
        )
        self.tokenized_queries = tokenized_queries
        self.tokenized_corpus = tokenized_corpus

    def __getitem__(self, idx: int) -> Tuple[int, List[str]]:
        if type(self.data[idx]) == list:
            qid = str(self.data[idx][0])
            pos_doc_ids = [self.data[idx][1]]
            # Get token and attention mask
            q_tok_ids = self.tokenized_queries[qid]
            q_tok_att_mask = [True] * len(q_tok_ids)
        elif type(self.data[idx]) == dict:
            qid = self.data[idx]["id"]
            pos_doc_ids = [item for item in self.data[idx]["answers"]]
            # Convert data type
            qid = str(qid)
            # Get token and attention mask
            q_tok_ids = self.tokenized_queries[qid]
            q_tok_att_mask = [True] * len(q_tok_ids)
        else:
            raise ValueError(f"Invalid data type: {type(self.data[idx])}")

        # Combine multiple sentences into single text
        q_tok_ids, q_sent_start_indices = combine_splitted_tok_ids(q_tok_ids)

        # Cut off by max length
        q_tok_ids = self.tokenizers.q_tokenizer.cutoff_by_max_len(q_tok_ids)
        q_sent_start_indices = (
            self.tokenizers.q_tokenizer.cut_off_sent_indices_by_max_len(
                q_sent_start_indices
            )
        )

        # Convert list to tensor
        q_tok_ids = torch.tensor(q_tok_ids, dtype=torch.int64, device="cpu")

        return {
            "q_id": qid,
            "q_tok_ids": q_tok_ids,
            "q_tok_att_mask": q_tok_att_mask,
            "pos_doc_ids": pos_doc_ids,
        }

    def _remove_redundant_tokenized_queries(self) -> None:
        """Delete redundant tokenized queries for memory saving."""
        # Get qids from the data
        if self.cfg_dataset.name == "beir-msmarco":
            required_qids: Set[int] = extract_qids_from_msmarco_data(self.data)
            required_qids: Set[str] = set([str(item) for item in required_qids])
        else:
            required_qids: Set[str] = extract_qids_from_non_msmarco_data(self.data)
        all_qids: List[str] = list(self.tokenized_queries.keys())
        # Remove redundant tokenized queries
        new_data: Dict[str, List[List[int]]] = {}
        for qid in all_qids:
            if qid in required_qids:
                new_data[qid] = self.tokenized_queries[qid]
        removed_cnt = len(all_qids) - len(new_data)
        logger.info(
            f"Removed {removed_cnt} and {len(new_data)} left for tokenized queries."
        )
        # Update tokenized queries
        self.tokenized_queries = new_data
        return None

    def _remove_redundant_tokenized_corpus(self) -> None:
        """Delete redundant tokenized corpus for memory saving."""
        # Get pids from the data
        if self.cfg_dataset.name == "beir-msmarco":
            required_pids: Set[int] = extract_pids_from_msmarco_data(self.data)
        else:
            required_pids: Set[int] = extract_pids_from_non_msmarco_data(self.data)
        all_pids: List[int] = [item for item in self.tokenized_corpus.keys()]
        # Remove redundant tokenized corpus
        new_data: Dict[int, List[List[int]]] = {}
        for pid in all_pids:
            if pid in required_pids:
                new_data[pid] = self.tokenized_corpus[pid]
        removed_cnt = len(self.tokenized_corpus) - len(new_data)
        logger.info(
            f"Removed {removed_cnt} and {len(new_data)} left for tokenized corpus."
        )
        # Update tokenized corpus
        self.tokenized_corpus = new_data
        return None
