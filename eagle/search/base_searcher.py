import abc
from typing import *

import hkkang_utils.time as time_utils
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
