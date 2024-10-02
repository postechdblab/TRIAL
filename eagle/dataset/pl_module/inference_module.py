import logging
from typing import *

from omegaconf import DictConfig

from eagle.dataset import InferenceDataset
from eagle.dataset.pl_module.base_module import BaseDataModule

logger = logging.getLogger("ContrastiveDataModule")


class InferenceDataModule(BaseDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg=cfg, skip_train=True)

    @property
    def train_qrels_path(self) -> str:
        return ""

    def _load_train_data(self, tokenized_queries: Dict, tokenized_corpus: Dict) -> None:
        return None

    def _load_val_data(
        self,
        tokenized_queries: Dict,
        tokenized_corpus: Dict,
    ) -> InferenceDataset:
        val_raw_dataset = InferenceDataset(
            cfg=self.cfg.val,
            cfg_dataset=self.cfg,
            tokenized_queries=tokenized_queries,
            tokenized_corpus=tokenized_corpus,
        )

        return val_raw_dataset
