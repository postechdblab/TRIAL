import logging
from typing import *

from omegaconf import DictConfig

from eagle.dataset.base_dataset import BaseDataset

logger = logging.getLogger("InferenceDataset")


class InferenceDataset(BaseDataset):
    def __init__(
        self,
        cfg: DictConfig,
        cfg_dataset: DictConfig,
        tokenized_queries: Dict,
        tokenized_corpus: Dict,
    ):
        super().__init__(
            cfg=cfg,
            cfg_dataset=cfg_dataset,
            tokenized_queries=tokenized_queries,
            tokenized_corpus=tokenized_corpus,
        )
        self.tokenized_queries = tokenized_queries
        self.tokenized_corpus = tokenized_corpus

    def __getitem__(self, idx: int) -> Tuple[int, List[str]]:
        if type(self.data[idx]) == list:
            qid = str(self.data[idx][0])
            pos_doc_ids = [str(self.data[idx][1])]
            # Get token and attention mask
            q_tok_ids = self.tokenized_queries[qid]
            q_tok_att_mask = [True] * len(q_tok_ids)
        elif type(self.data[idx]) == dict:
            qid = self.data[idx]["id"]
            pos_doc_ids = [item for item in self.data[idx]["answers"]]
            # Convert data type
            qid = str(qid)
            pos_doc_ids = [str(item) for item in pos_doc_ids]
            # Get token and attention mask
            q_tok_ids = self.tokenized_queries[qid]
            q_tok_att_mask = [True] * len(q_tok_ids)
        else:
            raise ValueError(f"Invalid data type: {type(self.data[idx])}")

        return {
            "q_id": qid,
            "q_tok_ids": q_tok_ids,
            "q_tok_att_mask": q_tok_att_mask,
            "pos_doc_ids": pos_doc_ids,
        }
