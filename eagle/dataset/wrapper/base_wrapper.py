import abc
import logging
from typing import *

from eagle.dataset.base_dataset import BaseDataset

logger = logging.getLogger("BaseDatasetWrapper")


class BaseDatasetWrapper:
    def __init__(
        self,
        dataset: BaseDataset,
        indices: Optional[List[int]] = None,
        nway: int = None,
        cache_nway: int = None,
        query_mapping: Dict[int, int] = None,
        corpus_mapping: Dict[int, int] = None,
        q_skip_ids: List[int] = None,
        d_skip_ids: List[int] = None,
    ):
        self.dataset = dataset
        self._indices = indices
        self.nway = nway
        self.cache_nway = cache_nway
        self.q_skip_ids = q_skip_ids
        self.d_skip_ids = d_skip_ids
        self.query_mapping = query_mapping
        self.corpus_mapping = corpus_mapping
        # Check if variables are valid
        assert len(self.dataset) == len(
            self.indices
        ), f"len(self.dataset)={len(self.dataset)}, len(self.indices)={len(self.indices)}"
        assert nway is not None, f"nway is None. Please provide nway."
        assert cache_nway is not None, f"cache_nway is None. Please provide cache_nway."
        assert cache_nway >= nway, f"cache_nway={cache_nway}, nway={nway}"

    @property
    def indices(self) -> List[int]:
        if self._indices is None:
            self._indices = [i for i in range(len(self.dataset))]
        return self._indices

    def __len__(self) -> int:
        return len(self.dataset)

    @abc.abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Any]:
        raise NotImplementedError("This method should be implemented in the subclass.")
