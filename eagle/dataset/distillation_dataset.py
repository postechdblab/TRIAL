import logging
from typing import *

from omegaconf import DictConfig

from eagle.dataset.base_dataset import BaseDataset

logger = logging.getLogger("DistillationDataset")


class DistillationDataset(BaseDataset):
    def __init__(
        self,
        cfg: DictConfig,
        cfg_dataset: DictConfig,
        tokenized_queries: Dict,
        tokenized_corpus: Dict,
        query_phrase_ranges: Dict = None,
        corpus_phrase_ranges: Dict = None,
        override_nway: Optional[int] = None,
        is_eval: bool = False,
    ):
        super().__init__(
            cfg=cfg,
            cfg_dataset=cfg_dataset,
            tokenized_queries=tokenized_queries,
            tokenized_corpus=tokenized_corpus,
            query_phrase_ranges=query_phrase_ranges,
            corpus_phrase_ranges=corpus_phrase_ranges,
        )
        self.nway = cfg.nway if override_nway is None else override_nway
        self.is_eval = is_eval

    def __getitem__(
        self, idx: int
    ) -> Tuple[int, List[str], List[str], Optional[List[float]], Optional[List[float]]]:
        qid = self.data[idx][0]
        pos_doc_ids, pos_doc_score = self.data[idx][1]
        neg_doc_ids, neg_doc_scores = zip(
            *self.data[idx][2:][self.neg_start_idx : self.neg_end_idx]
        )
        pos_doc_ids = [pos_doc_ids]
        # Convert data type
        qid = str(qid)
        pos_doc_ids = [str(item) for item in pos_doc_ids]
        neg_doc_ids = [str(n_id) for n_id in neg_doc_ids]

        # Get token and attention mask
        q_tok_ids = self.tokenized_queries[qid]
        q_tok_att_mask = [True] * len(q_tok_ids)
        d_tok_ids = [self.tokenized_corpus[pid] for pid in pos_doc_ids + neg_doc_ids]
        d_tok_att_mask = [[True] * len(item) for item in d_tok_ids]

        result = {
            "q_id": qid,
            "q_tok_ids": q_tok_ids,
            "q_tok_att_mask": q_tok_att_mask,
            "doc_tok_ids": d_tok_ids,
            "doc_tok_att_mask": d_tok_att_mask,
            "pos_doc_ids": pos_doc_ids,
            "neg_doc_ids": neg_doc_ids,
            "distillation_scores": [pos_doc_score] + list(neg_doc_scores),
        }
        if self.is_eval:
            result["labels"] = self.labels
        return result
