import os

import torch
import tqdm
import ujson
from typing import *


class BaseResidualEmbeddings:
    def __init__(self, codes, residuals):
        """
        Supply the already compressed residuals.
        """

        # assert isinstance(residuals, bitarray), type(residuals)
        assert codes.size(0) == residuals.size(0), (codes.size(), residuals.size())
        assert codes.dim() == 1 and residuals.dim() == 2, (
            codes.size(),
            residuals.size(),
        )
        assert residuals.dtype == torch.uint8

        self.codes = codes.to(torch.int32)  # (num_embeddings,) int32
        self.residuals = residuals  # (num_embeddings, compressed_dim) uint8

    @classmethod
    def load_chunks(
        cls,
        index_path: str,
        chunk_idxs: List,
        num_tok_embeddings: int,
    ) -> Any:
        num_tok_embeddings += 512

        dim, nbits = get_dim_and_nbits(index_path)

        print("#> Loading codes and residuals...")

        tok_codes = torch.empty(num_tok_embeddings, dtype=torch.int32)
        tok_residuals = torch.empty(
            num_tok_embeddings, dim // 8 * nbits, dtype=torch.uint8
        )

        tok_codes_offset = 0
        for chunk_idx in tqdm.tqdm(chunk_idxs):
            tok_chunk = cls.load(index_path, chunk_idx)

            # tok
            tok_codes_endpos = tok_codes_offset + tok_chunk.codes.size(0)
            tok_codes[tok_codes_offset:tok_codes_endpos] = tok_chunk.codes
            tok_residuals[tok_codes_offset:tok_codes_endpos] = tok_chunk.residuals
            tok_codes_offset = tok_codes_endpos

        tok_cls = cls(tok_codes, tok_residuals)
        return tok_cls

    @classmethod
    def load(cls, index_path, chunk_idx) -> torch:
        tok_codes = cls.load_codes(index_path, chunk_idx)
        tok_residuals = cls.load_residuals(index_path, chunk_idx)
        tok_cls = cls(tok_codes, tok_residuals)

        return tok_cls

    @classmethod
    def load_codes(
        self, index_path: str, chunk_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get paths
        tok_codes_path = os.path.join(index_path, f"{chunk_idx}-tok.codes.pt")

        # Load codes
        tok_codes = torch.load(tok_codes_path, map_location="cpu", weights_only=True)

        return tok_codes

    @classmethod
    def load_residuals(
        self, index_path, chunk_idx
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tok_residuals_path = os.path.join(index_path, f"{chunk_idx}-tok.residuals.pt")
        tok_residuals = torch.load(
            tok_residuals_path, map_location="cpu", weights_only=True
        )
        return tok_residuals

    def save(self, path_prefix):
        codes_path = f"{path_prefix}.codes.pt"
        residuals_path = f"{path_prefix}.residuals.pt"  # f'{path_prefix}.residuals.bn'

        torch.save(self.codes, codes_path)
        torch.save(self.residuals, residuals_path)
        # _save_bitarray(self.residuals, residuals_path)

    def __len__(self):
        return self.codes.size(0)


def get_dim_and_nbits(index_path):
    # TODO: Ideally load this using ColBERTConfig.load_from_index!
    with open(os.path.join(index_path, "metadata.json")) as f:
        metadata = ujson.load(f)["config"]

    dim = metadata["dim"]
    nbits = metadata["nbits"]

    assert (dim * nbits) % 8 == 0, (dim, nbits, dim * nbits)

    return dim, nbits


def get_codes_size(index_path, chunk_idx):
    # TODO: Ideally load this using ColBERTConfig.load_from_index!
    with open(os.path.join(index_path, f"{chunk_idx}.metadata.json")) as f:
        metadata = ujson.load(f)

    return metadata["num_embeddings"]


def get_residuals_size(index_path, chunk_idx):
    codes_size = get_codes_size(index_path, chunk_idx)
    dim, nbits = get_dim_and_nbits(index_path)

    packed_dim = dim // 8 * nbits
    return codes_size * packed_dim, codes_size, packed_dim
