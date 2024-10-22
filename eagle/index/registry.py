from typing import Dict

from eagle.index.base_indexer import BaseIndexer
from eagle.index.colbert_indexer import ColBERTIndxer
from eagle.index.eagle_indexer import EAGLEIndexer

INDEXER_REGISTRY: Dict[str, BaseIndexer] = {
    "colbert": ColBERTIndxer,
    "eagle": EAGLEIndexer,
}
