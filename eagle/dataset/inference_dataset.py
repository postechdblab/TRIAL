import logging
from typing import *

import tqdm
from omegaconf import DictConfig

from eagle.dataset.base_dataset import BaseDataset

logger = logging.getLogger("InferenceDataset")


class InferenceDataset(BaseDataset):
    def __init__(
        self,
        cfg: DictConfig,
        cfg_dataset: DictConfig,
        queries: List[Dict],
    ):
        super().__init__(cfg=cfg, cfg_dataset=cfg_dataset)
        self.queries = queries

    def __getitem__(
        self, idx: int
    ) -> Tuple[int, List[str]]:
        qid = self.data[idx]["id"]
        pos_doc_ids = [item for item in self.data[idx]["answers"]]
        # Convert data type
        qid = str(qid)
        pos_doc_ids = [str(item) for item in pos_doc_ids]
        return qid, pos_doc_ids

    @property
    def dict_keys(self) -> List[str]:
        """This has to be consistent with the keys in the to_dict method."""
        return [
            "q_ids",
            "q_texts",
            "pos_doc_texts_list",
            "pos_doc_ids_list",
        ]

    def to_dict(self, corpus: Dict[str, str]) -> Dict:
        # Prepare data
        q_ids: List[str] = []
        q_texts: List[str] = []
        pos_doc_texts_list: List[int] = []
        pos_doc_ids_list: List[str] = []
        for i, (
            qid,
            pos_doc_ids
        ) in tqdm.tqdm(enumerate(self), desc="Converting to dict", total=len(self)):
            q_ids.append(qid)
            q_texts.append(self.queries[qid])
            pos_doc_texts_list.append(
                [corpus[pos_doc_id] for pos_doc_id in pos_doc_ids]
            )
            pos_doc_ids_list.append(pos_doc_ids)

        # Create return dict
        return_dict = {
            "q_ids": q_ids,
            "q_texts": q_texts,
            "pos_doc_texts_list": pos_doc_texts_list,
            "pos_doc_ids_list": pos_doc_ids_list,
        }

        assert set(return_dict.keys()) == set(self.dict_keys), f"Keys are not consistent: {return_dict.keys()} vs {self.dict_keys}"
        
        return return_dict