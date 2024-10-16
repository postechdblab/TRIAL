import logging
import os
from typing import *

from eagle.dataset import ContrastiveDataset, DistillationDataset
from eagle.dataset.pl_module.base_module import BaseDataModule

logger = logging.getLogger("DistillationDataModule")


class DistillationDataModule(BaseDataModule):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    @property
    def train_qrels_path(self) -> str:
        return os.path.join(
            self.cfg.dir_path, self.cfg.name, self.cfg.train_distillation.qrel_file
        )

    def _load_train_data(
        self,
        tokenized_queries: Dict,
        tokenized_corpus: Dict,
    ) -> DistillationDataset:
        train_raw_dataset = DistillationDataset(
            cfg=self.cfg.train_distillation,
            cfg_dataset=self.cfg,
            tokenizers=self.tokenizers,
            tokenized_queries=tokenized_queries,
            tokenized_corpus=tokenized_corpus,
            is_eval=False,
        )
        return train_raw_dataset

    def _load_val_data(
        self,
        tokenized_queries: Dict,
        tokenized_corpus: Dict,
    ) -> ContrastiveDataset:
        val_raw_dataset = ContrastiveDataset(
            cfg=self.cfg.val,
            cfg_dataset=self.cfg,
            tokenizers=self.tokenizers,
            tokenized_queries=tokenized_queries,
            tokenized_corpus=tokenized_corpus,
            is_eval=True,
        )
        return val_raw_dataset
