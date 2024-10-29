from typing import *

from model import BaseRetriever, RetrievalResult
from omegaconf import DictConfig

from eagle.model.eagle import EAGLE
from eagle.search.plaid import PLAID
from eagle.tokenization import Tokenizers


class EAGLERetriever(BaseRetriever):
    def __init__(
        self,
        cfg: DictConfig,
        index_dir_path: str,
    ):
        self.cfg = cfg
        self.tokenizers = Tokenizers(
            cfg.q_tokenizer, cfg.d_tokenizer, cfg.model.backbone_name
        )
        self.model = EAGLE(cfg=cfg.model, tokenizers=self.tokenizers)
        self.searcher = PLAID(
            index_path=index_dir_path,
            d_cross_attention_layer=self.model.cross_att_layer,
            d_weight_project_layer=self.model.d_weight_layer,
            d_weight_layer_norm=self.model.d_weight_layer_norm,
        )

    def retrieve_batch(
        self, quries: List[str], topk: int = 100, return_scores: bool = False, **kwargs
    ) -> List[List[RetrievalResult]]:
        raise NotImplementedError

    def calculate_score_by_doc_batch(
        self, queries: List[str], doc_texts: List[str], **kwargs
    ) -> List[float]:
        raise NotImplementedError

    def create_index(self, index_name: str, corpus_path: str) -> None:
        raise NotImplementedError
