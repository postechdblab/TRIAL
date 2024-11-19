from typing import Dict, Type

from eagle.index.base_indexer import BaseIndexer
from eagle.index.colbert_indexer import ColBERTIndexer
from eagle.index.eagle_indexer import EAGLEIndexer

INDEXER_REGISTRY: Dict[str, Type[BaseIndexer]] = {
    "colbert": ColBERTIndexer,
    "eagle": EAGLEIndexer,
}
