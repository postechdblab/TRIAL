import copy
import logging
from typing import *

import hkkang_utils.list as list_utils
import torch
from omegaconf import DictConfig
from torch.nn.utils.rnn import pad_sequence

from eagle.dataset.base_dataset import BaseDataset
from eagle.tokenization.tokenizers import Tokenizers
from eagle.tokenization.utils import combine_splitted_tok_ids

logger = logging.getLogger("ContrastiveDataset")


class ContrastiveDataset(BaseDataset):
    def __init__(
        self,
        cfg: DictConfig,
        cfg_dataset: DictConfig,
        tokenizers: Tokenizers,
        tokenized_queries: Dict,
        tokenized_corpus: Dict,
        is_eval: bool = False,
    ):
        super().__init__(
            cfg=cfg,
            cfg_dataset=cfg_dataset,
            tokenizers=tokenizers,
            tokenized_queries=tokenized_queries,
            tokenized_corpus=tokenized_corpus,
        )
        self.is_eval = is_eval

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Get the target data through the parent class (for shuffling)
        target_data = super().__getitem__(idx)

        # Get qid, pos_doc_ids, and neg_doc_ids
        qid = target_data[0]
        pos_doc_ids = [target_data[1]]
        neg_doc_ids = target_data[2:][self.neg_start_idx : self.neg_end_idx]

        # Get token ids and attention mask
        q_tok_ids_per_sentences: List[List[int]] = self.tokenized_queries[
            self.tokenized_queries_key_type(qid)
        ]
        q_tok_ids, q_sent_start_indices = combine_splitted_tok_ids(
            q_tok_ids_per_sentences
        )

        d_tok_ids_per_sentences: List[List[List[int]]] = [
            self.tokenized_corpus[self.tokenized_corpus_key_type(pid)]
            for pid in pos_doc_ids + neg_doc_ids
        ]
        d_tok_ids = []
        d_sent_start_indices = []
        for item in d_tok_ids_per_sentences:
            tok_ids, sent_start_indices = combine_splitted_tok_ids(item)
            d_tok_ids.append(tok_ids)
            d_sent_start_indices.append(sent_start_indices)

        # Cut off by max length
        q_tok_ids = self.tokenizers.q_tokenizer.cutoff_by_max_len(q_tok_ids)
        d_tok_ids = [
            self.tokenizers.d_tokenizer.cutoff_by_max_len(item) for item in d_tok_ids
        ]
        q_sent_start_indices = (
            self.tokenizers.q_tokenizer.cut_off_sent_indices_by_max_len(
                q_sent_start_indices
            )
        )
        d_sent_start_indices = [
            self.tokenizers.q_tokenizer.cut_off_sent_indices_by_max_len(item)
            for item in d_sent_start_indices
        ]

        # Convert list to tensor
        q_tok_ids = torch.tensor(q_tok_ids, dtype=torch.int64, device="cpu")
        d_tok_ids = [
            torch.tensor(item, dtype=torch.int64, device="cpu") for item in d_tok_ids
        ]
        d_tok_ids = pad_sequence(d_tok_ids, batch_first=True)

        result = {
            "q_id": qid,
            "q_tok_ids": q_tok_ids,
            "doc_tok_ids": d_tok_ids,
            "pos_doc_ids": pos_doc_ids,
            "neg_doc_ids": neg_doc_ids,
            "q_sent_start_indices": q_sent_start_indices,
            "doc_sent_start_indices": d_sent_start_indices,
        }

        # Add labels during evaluation
        result["labels"] = (
            torch.tensor(self.labels, dtype=torch.bool, device="cpu")
            if self.is_eval
            else None
        )

        return result

    def _remove_redundant_tokenized_queries(self) -> None:
        """Delete redundant tokenized queries for memory saving."""
        # Get qids from the data
        required_qids: Set[int] = set([item[0] for item in self.data])
        all_qids: List[str] = list(self.tokenized_queries.keys())
        # Remove redundant tokenized queries
        new_data: Dict[str, List[List[int]]] = {}
        for qid in all_qids:
            if int(qid) in required_qids:
                new_data[qid] = copy.deepcopy(self.tokenized_queries[qid])
        removed_cnt = len(self.tokenized_queries) - len(new_data)
        logger.info(
            f"Removed {removed_cnt} and {len(new_data)} left for tokenized queries."
        )
        # Update tokenized queries
        self.tokenized_queries = new_data
        return None

    def _remove_redundant_tokenized_corpus(self) -> None:
        """Delete redundant tokenized corpus for memory saving."""
        # Get doc ids from the data
        doc_ids: Set[int] = set(
            list_utils.do_flatten_list([item[1:] for item in self.data])
        )
        all_pids: List[int] = list(self.tokenized_corpus.keys())
        # Remove redundant tokenized corpus
        new_data: Dict[int, List[List[int]]] = {}
        for pid in all_pids:
            if pid in doc_ids:
                new_data[pid] = copy.deepcopy(self.tokenized_corpus[pid])
        removed_cnt = len(self.tokenized_corpus) - len(new_data)
        logger.info(
            f"Removed {removed_cnt} and {len(new_data)} left for tokenized corpus."
        )
        # Update tokenized corpus
        self.tokenized_corpus = new_data
        return None
