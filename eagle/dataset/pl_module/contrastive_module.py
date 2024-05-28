import logging
from typing import *

from datasets import Dataset

from eagle.dataset import ContrastiveDataset
from eagle.dataset.pl_module.base_module import BaseDataModule

logger = logging.getLogger("ContrastiveDataModule")


class ContrastiveDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_train_data(self, queries: Dict) -> Dataset:
        train_raw_dataset = ContrastiveDataset(
            cfg=self.cfg.train_contrastive,
            cfg_dataset=self.cfg,
            queries=queries,
            override_nway=self.cfg.cache_nway,
        )

        return train_raw_dataset

    def _load_val_data(self, queries: Dict) -> Dataset:
        val_raw_dataset = ContrastiveDataset(
            cfg=self.cfg.val,
            cfg_dataset=self.cfg,
            queries=queries,
            override_nway=self.cfg.val.override_nway,
        )
        
        return val_raw_dataset