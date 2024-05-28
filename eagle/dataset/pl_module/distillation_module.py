import logging
import os
from typing import *

import lightning as L
from omegaconf import DictConfig

from eagle.dataset.pl_module.base_module import BaseDataModule
from eagle.tokenizer import NewTokenizer

logger = logging.getLogger("DistillationDataModule")


class DistillationDataModule(BaseDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()

    def prepare_data(self) -> None:
        """
        Preprocess data for single process before spawning.
        Create cache if not in debug mode.
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Preprocess data for each process after spawning.
        Load cache if not debug. Otherwise, load sample data.
        """
        pass
