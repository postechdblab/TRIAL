import abc
import logging
import os
from typing import *

import hkkang_utils.file as file_utils
import numpy as np
from omegaconf import DictConfig

from eagle.dataset.utils import get_labels

logger = logging.getLogger("BaseDataset")


class BaseDataset:
    def __init__(self, cfg: DictConfig, cfg_dataset: DictConfig):
        self.cfg = cfg
        self.cfg_dataset = cfg_dataset
        self.data = self._read_data(self.data_path)

    def __len__(self) -> int:
        return len(self.data)

    @property
    def data_path(self) -> str:
        return os.path.join(
            self.cfg_dataset.dir_path, self.cfg_dataset.name, self.cfg.data_file
        )

    @property
    def neg_num(self) -> int:
        return self.nway - 1

    @property
    def neg_start_idx(self) -> int:
        return self.cfg.negative_start_offset

    @property
    def neg_end_idx(self) -> int:
        return self.cfg.negative_start_offset + self.neg_num

    @property
    def labels(self) -> np.array:
        return get_labels(bsize=1, neg_num=self.neg_num).squeeze(0)

    def _read_data(self, path: str) -> List[List[int]]:
        data: List = file_utils.read_json_file(path, auto_detect_extension=True)
        # Sample data if needed
        sample_size = (
            self.cfg.debug_sample_size if self.cfg.is_debug else self.cfg.sample_size
        )
        if sample_size > 0:
            data = data[:sample_size]
        return data

    @abc.abstractmethod
    def __getitem__(self):
        raise NotImplementedError()
