from typing import *

from omegaconf import DictConfig

from eagle.model.base_model import BaseModel
from eagle.search.base_searcher import BaseSearcher
from eagle.search.plaid import PLAID


class EAGLESearcher(BaseSearcher):
    def __init__(self, model: BaseModel, cfg: DictConfig, index_dir_path: str) -> None:
        super().__init__(cfg, model)
        self.index_dir_path = index_dir_path
        self.plaid = PLAID(
            index_path=self.index_dir_path,
            indexer_name=model.cfg.name,
            d_cross_attention_layer=self.model.d_cross_attention_layer,
            d_weight_project_layer=self.model.d_weight_project_layer,
            d_weight_layer_norm=self.model.d_weight_layer_norm,
        )

    def search(self, *args, **kwargs) -> Any:
        raise NotImplementedError
