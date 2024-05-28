import abc
import logging
import os
from typing import *

import hkkang_utils.file as file_utils
from omegaconf import DictConfig

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
        return os.path.join(self.cfg_dataset.dir_path, self.cfg_dataset.name, self.cfg.data_file)

    def _read_data(self, path: str) -> List[List[int]]:
        data: List = file_utils.read_json_file(path, auto_detect_extension=True)
        # Sample data if needed
        sample_size = self.cfg.debug_sample_size if self.cfg.is_debug else self.cfg.sample_size
        if sample_size > 0:
            data = data[: sample_size]
        return data
    
    @abc.abstractmethod
    def __getitem__(self):
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def dict_keys(self) -> List[str]:
        raise NotImplementedError()
    
    @abc.abstractmethod
    def to_dict(self, corpus: Dict[str, str]) -> Dict:
        raise NotImplementedError()