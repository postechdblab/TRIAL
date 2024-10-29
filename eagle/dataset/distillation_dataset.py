import logging
from typing import *

import torch
from omegaconf import DictConfig
from torch.nn.utils.rnn import pad_sequence

from eagle.dataset.base_dataset import BaseDataset
from eagle.tokenization import Tokenizers
from eagle.tokenization.utils import combine_splitted_tok_ids

logger = logging.getLogger("DistillationDataset")


class DistillationDataset(BaseDataset):
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
        target_data = super().__getitem__(idx)

        qid = target_data[0]
        pos_doc_ids, pos_doc_score = target_data[1]
        neg_doc_ids, neg_doc_scores = zip(
            *target_data[2:][self.neg_start_idx : self.neg_end_idx]
        )
        pos_doc_ids = [pos_doc_ids]

        # Convert data type
        qid = str(qid)
        pos_doc_ids = [str(item) for item in pos_doc_ids]
        neg_doc_ids = [str(n_id) for n_id in neg_doc_ids]

        # Get token
        q_tok_ids_per_sentences = self.tokenized_queries[qid]
        q_tok_ids, q_sent_start_indices = combine_splitted_tok_ids(
            q_tok_ids_per_sentences
        )
        d_tok_ids_per_sentences = [
            self.tokenized_corpus[pid] for pid in pos_doc_ids + neg_doc_ids
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
        # Convert list to tensor
        q_tok_ids = torch.tensor(q_tok_ids, dtype=torch.int64, device="cpu")
        d_tok_ids = [
            torch.tensor(item, dtype=torch.int64, device="cpu") for item in d_tok_ids
        ]
        d_tok_ids = pad_sequence(d_tok_ids, batch_first=True)

        # Create distillation scores
        distillation_scores = [pos_doc_score] + list(neg_doc_scores)
        if distillation_scores:
            distillation_scores = torch.tensor(
                distillation_scores, dtype=torch.float32, device="cpu"
            )

        result = {
            "q_id": qid,
            "q_tok_ids": q_tok_ids,
            "doc_tok_ids": d_tok_ids,
            "pos_doc_ids": pos_doc_ids,
            "neg_doc_ids": neg_doc_ids,
            "q_sent_start_indices": q_sent_start_indices,
            "doc_sent_start_indices": d_sent_start_indices,
            "distillation_scores": distillation_scores,
        }
        if self.is_eval:
            result["labels"] = self.labels
        return result
