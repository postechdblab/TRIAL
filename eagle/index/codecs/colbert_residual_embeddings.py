from typing import *

import torch

from eagle.index.codecs.residual_embeddings import BaseResidualEmbeddings
from eagle.search.strided_tensor import StridedTensor


class ColBERTResidualEmbeddings(BaseResidualEmbeddings):
    pass


class ColBERTResidualEmbeddingsStrided:
    def __init__(self, codec, embeddings, doclens, tok_ids=None):
        self.codec = codec
        self.codes = embeddings.codes
        self.residuals = embeddings.residuals
        self.use_gpu = self.codec.use_gpu

        self.codes_strided = StridedTensor(self.codes, doclens, use_gpu=self.use_gpu)
        self.residuals_strided = StridedTensor(
            self.residuals, doclens, use_gpu=self.use_gpu
        )
        self.tok_ids_strded = (
            StridedTensor(tok_ids, doclens, use_gpu=self.use_gpu)
            if tok_ids is not None
            else None
        )

    def lookup_pids(
        self, passage_ids
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        codes_packed, codes_lengths = self.codes_strided.lookup(
            passage_ids
        )  # .as_packed_tensor()
        residuals_packed, _ = self.residuals_strided.lookup(
            passage_ids
        )  # .as_packed_tensor()
        tok_ids, _ = (
            self.tok_ids_strded.lookup(passage_ids)
            if self.tok_ids_strded is not None
            else (None, None)
        )

        embeddings_packed = self.codec.decompress(
            ColBERTResidualEmbeddings(codes_packed, residuals_packed)
        )

        return embeddings_packed, codes_lengths, tok_ids

    def lookup_codes(self, passage_ids):
        return self.codes_strided.lookup(passage_ids)  # .as_packed_tensor()
