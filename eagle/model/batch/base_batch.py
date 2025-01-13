import abc
from typing import *

from torch.utils.data import Dataset as TorchDataset

from eagle.dataset.base_dataset import BaseDataset


class BaseBatch(TorchDataset):
    def __init__(
        self,
        dataset: BaseDataset,
        q_skip_tok_ids: List[int] = None,
        d_skip_tok_ids: List[int] = None,
        pad_to_max_length: bool = False,
    ) -> None:
        self.dataset = dataset
        self.q_skip_tok_ids = q_skip_tok_ids if q_skip_tok_ids is not None else []
        self.d_skip_tok_ids = d_skip_tok_ids if d_skip_tok_ids is not None else []
        self.pad_to_max_length = pad_to_max_length

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.parse_data(self.dataset[index])

    @abc.abstractmethod
    def parse_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def collate_fn(self, input_dics: List[Dict]) -> Dict:
        raise NotImplementedError
