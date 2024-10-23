from typing import Dict, Type

from eagle.index.codecs.colbert_residual_embeddings import (
    ColBERTResidualEmbeddings,
    ColBERTResidualEmbeddingsStrided,
)
from eagle.index.codecs.eagle_residual_embeddings import (
    EAGLEResidualEmbeddings,
    EAGLEResidualEmbeddingsStrided,
)
from eagle.index.codecs.residual_embeddings import BaseResidualEmbeddings

CODEC_REGISTRY: Dict[str, Type[BaseResidualEmbeddings]] = {
    "colbert": ColBERTResidualEmbeddings,
    "eagle": EAGLEResidualEmbeddings,
}

CODEC_STRIDED_REGISTRY: Dict = {
    "colbert": ColBERTResidualEmbeddingsStrided,
    "eagle": EAGLEResidualEmbeddingsStrided,
}
