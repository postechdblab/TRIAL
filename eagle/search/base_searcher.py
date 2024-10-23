import abc
from typing import *

from omegaconf import DictConfig

from eagle.model.base_model import BaseModel


class BaseSearcher:
    def __init__(self, cfg: DictConfig, model: BaseModel) -> None:
        self.cfg = cfg
        self.model = model

    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> Tuple[List[List[int]], List[List[float]]]:
        """Search for relevant documents given a query.

        :raises Implement this method in the subclass
        :return: List of document ids and their scores
        :rtype: Tuple[List[List[int]], List[List[float]]]
        """
        raise NotImplementedError
