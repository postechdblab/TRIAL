import abc
from typing import *

from torch.utils.data import Dataset as TorchDataset

from eagle.dataset import BaseDataset


class BaseBatch(TorchDataset):
    def __init__(self, dataset: BaseDataset, skip_tok_ids: List[int] = None) -> None:
        self.dataset = dataset
        self.skip_tok_ids = skip_tok_ids if skip_tok_ids is not None else []

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.parse_data(self.dataset[index])

    @abc.abstractmethod
    def parse_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def collate_fn(input_dics: List[Dict]) -> Dict:
        raise NotImplementedError
