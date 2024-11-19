import abc
from typing import *

import hkkang_utils.time as time_utils
import torch
from omegaconf import DictConfig

from eagle.model.base_model import BaseModel


class BaseSearcher:
    def __init__(self, cfg: DictConfig, model: BaseModel) -> None:
        self.cfg = cfg
        self.model = model
        self.timer = time_utils.Timer(
            class_name=self.__class__.__name__, func_name="search"
        )

    def __call__(self, *args, **kwargs) -> Tuple[List[List[int]], List[List[float]]]:
        with self.timer.measure():
            result = self.search(*args, **kwargs)
        return result

    @abc.abstractmethod
    def search(self, *args, **kwargs) -> Tuple[List[List[int]], List[List[float]]]:
        """Search for relevant documents given a query.

        :raises Implement this method in the subclass
        :return: List of document ids and their scores
        :rtype: Tuple[List[List[int]], List[List[float]]]
        """
        raise NotImplementedError

    def preprocess_doc_indices(
        self, doc_indices: Union[List[List[int]], torch.Tensor]
    ) -> torch.Tensor:
        """Subtract one to the indices to change from 1-based index to 0-based index.

        :param doc_indices: positive document indices
        :type doc_indices: List[List[int]]
        :return: preprocessed positive document indices
        :rtype: torch.Tensor
        """
        input_type = type(doc_indices)
        if input_type == list:
            doc_indices = torch.tensor(doc_indices, dtype=torch.int64, device="cpu")
        # Convert
        doc_indices = doc_indices - torch.ones_like(doc_indices)
        if input_type == list:
            doc_indices = doc_indices.tolist()
        return doc_indices

    def postprocess_doc_indices(
        self, doc_indices: Union[List[List[int]], torch.Tensor]
    ) -> List[List[int]]:
        """Plus one to the indices to change from 0-based index to 1-based index.

        :param doc_indices: positive document indices
        :type doc_indices: torch.Tensor
        :return: postprocessed positive document indices
        :rtype: List[List[int]]
        """
        input_type = type(doc_indices)
        if input_type == list:
            doc_indices = torch.tensor(doc_indices, dtype=torch.int64, device="cpu")
        # Convert
        doc_indices = doc_indices + torch.ones_like(doc_indices)
        if input_type == list:
            doc_indices = doc_indices.tolist()
        return doc_indices
