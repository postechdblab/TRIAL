from typing import Dict, Type
from eagle.search.colbert_searcher import ColBERTSearcher
from eagle.search.eagle_searcher import EAGLESearcher
from eagle.search.base_searcher import BaseSearcher

SEARCHER_REGISTRY: Dict[str, Type[BaseSearcher]] = {
    "eagle": EAGLESearcher,
    "colbert": ColBERTSearcher,
}
