import logging
from typing import *

import tqdm
from omegaconf import DictConfig

from eagle.dataset.base_dataset import BaseDataset

logger = logging.getLogger("ContrastiveDataset")


class ContrastiveDataset(BaseDataset):
    def __init__(
        self,
        cfg: DictConfig,
        cfg_dataset: DictConfig,
        queries: List[Dict],
        override_nway: Optional[int] = None,
    ):
        super().__init__(cfg=cfg, cfg_dataset=cfg_dataset)
        self.nway = cfg.nway if override_nway is None else override_nway
        self.queries = queries

    def __getitem__(
        self, idx: int
    ) -> Tuple[int, List[str], List[str]]:
        qid = self.data[idx][0]
        pos_doc_ids = self.data[idx][1]
        neg_doc_ids = self.data[idx][2:]
        pos_doc_ids = [pos_doc_ids]
        # Convert data type
        qid = str(qid)
        pos_doc_ids = [str(item) for item in pos_doc_ids]
        neg_doc_ids = [str(n_id) for n_id in neg_doc_ids]
        return qid, pos_doc_ids, neg_doc_ids

    @property
    def dict_keys(self) -> List[str]:
        """This has to be consistent with the keys in the to_dict method."""
        return [
            "q_ids",
            "q_texts",
            "pos_doc_texts_list",
            "neg_doc_texts_list",
            "pos_doc_ids_list",
            "neg_doc_ids_list",
        ]

    def to_dict(self, corpus: Dict[str, str]) -> Dict:
        # Get negative doc indices
        neg_start_idx = self.cfg.negative_start_offset
        neg_end_idx = neg_start_idx + self.nway - 1

        # Prepare data
        q_ids: List[str] = []
        q_texts: List[str] = []
        pos_doc_texts_list: List[int] = []
        neg_doc_texts_list: List[List[int]] = []
        pos_doc_ids_list: List[str] = []
        neg_doc_ids_list: List[str] = []
        for i, (
            qid,
            pos_doc_ids,
            neg_doc_ids,
        ) in tqdm.tqdm(enumerate(self), desc="Converting to dict", total=len(self)):
            q_ids.append(qid)
            q_texts.append(self.queries[qid])
            pos_doc_texts_list.append(
                [corpus[pos_doc_id] for pos_doc_id in pos_doc_ids]
            )
            for pos_doc_id in pos_doc_ids:
                assert (
                    pos_doc_id not in neg_doc_ids
                ), f"Positive doc id is in negative doc ids: {pos_doc_id} in {neg_doc_ids}"
            neg_doc_texts_list.append(
                [corpus[n_id] for n_id in neg_doc_ids[neg_start_idx:neg_end_idx]]
            )
            pos_doc_ids_list.append(pos_doc_ids)
            neg_doc_ids_list.append(neg_doc_ids[neg_start_idx:neg_end_idx])

        # Create return dict
        return_dict = {
            "q_ids": q_ids,
            "q_texts": q_texts,
            "pos_doc_texts_list": pos_doc_texts_list,
            "neg_doc_texts_list": neg_doc_texts_list,
            "pos_doc_ids_list": pos_doc_ids_list,
            "neg_doc_ids_list": neg_doc_ids_list,
        }

        assert set(return_dict.keys()) == set(self.dict_keys), f"Keys are not consistent: {return_dict.keys()} vs {self.dict_keys}"

        return return_dict