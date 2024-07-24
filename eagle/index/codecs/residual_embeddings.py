import os

import torch
import tqdm
import ujson
from typing import *


class ResidualEmbeddings:
    # from colbert.indexing.codecs.residual_embeddings_strided import \
    #     ResidualEmbeddingsStrided
    # Strided = ResidualEmbeddingsStrided

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
        num_cls_embeddings: int,
        num_tok_embeddings: int,
        num_phrase_embeddings: int,
        load_index_with_mmap=False,
    ) -> Tuple:
        use_cls = num_cls_embeddings > 0
        use_phrase = num_phrase_embeddings > 0
        num_cls_embeddings += 512  # pad for access with strides
        num_tok_embeddings += 512
        num_phrase_embeddings += 512

        dim, nbits = get_dim_and_nbits(index_path)

        if load_index_with_mmap:
            raise NotImplementedError(
                "TODO: Implement mmap loading for ResidualEmbeddings"
            )
            if len(chunk_idxs) != 1:
                raise ValueError(
                    "Index must only have 1 chunk to load with memory mapping!"
                    "Use the colbert/utils/coalesce.py to prepare index for memory mapping."
                )

            print("#> Loading codes and residuals with memory mapping...")

            residuals_path = os.path.join(index_path, f"0.residuals.pt")
            codes_path = os.path.join(index_path, f"0.codes.pt")

            codes_size = get_codes_size(index_path, 0)
            storage = torch.IntStorage.from_file(
                filename=codes_path, shared=True, size=codes_size + 80
            )
            # Trim the header, which is 320 bytes, or 80x 32-byte ints
            codes = torch.IntTensor(storage)[80:]

            residuals_size, codes_size, packed_dim = get_residuals_size(index_path, 0)
            storage = torch.ByteStorage.from_file(
                filename=residuals_path, shared=True, size=residuals_size + 320
            )
            ret = torch.ByteTensor(storage)
            # Trim to 320-byte header
            ret = ret[320:]
            ret = torch.reshape(ret, (codes_size, packed_dim))
            residuals = ret
        else:
            print("#> Loading codes and residuals...")
            if use_cls:
                cls_codes = torch.empty(num_cls_embeddings, dtype=torch.int32)
                cls_residuals = torch.empty(
                    num_cls_embeddings, dim // 8 * nbits, dtype=torch.uint8
                )

            tok_codes = torch.empty(num_tok_embeddings, dtype=torch.int32)
            tok_residuals = torch.empty(
                num_tok_embeddings, dim // 8 * nbits, dtype=torch.uint8
            )
            if use_phrase:
                phrase_codes = torch.empty(num_phrase_embeddings, dtype=torch.int32)
                phrase_residuals = torch.empty(
                    num_phrase_embeddings, dim // 8 * nbits, dtype=torch.uint8
                )

            cls_codes_offset = 0
            tok_codes_offset = 0
            phrase_codes_offset = 0
            for chunk_idx in tqdm.tqdm(chunk_idxs):
                cls_chunk, tok_chunk, phrase_chunk = cls.load(index_path, chunk_idx)

                # cls
                if use_cls:
                    cls_codes_endpos = cls_codes_offset + cls_chunk.codes.size(0)
                    # Copy the values over to the allocated space
                    cls_codes[cls_codes_offset:cls_codes_endpos] = cls_chunk.codes
                    cls_residuals[cls_codes_offset:cls_codes_endpos] = (
                        cls_chunk.residuals
                    )
                    cls_codes_offset = cls_codes_endpos

                # tok
                tok_codes_endpos = tok_codes_offset + tok_chunk.codes.size(0)
                tok_codes[tok_codes_offset:tok_codes_endpos] = tok_chunk.codes
                tok_residuals[tok_codes_offset:tok_codes_endpos] = tok_chunk.residuals
                tok_codes_offset = tok_codes_endpos

                # phrase
                if use_phrase:
                    phrase_codes_endpos = phrase_codes_offset + phrase_chunk.codes.size(
                        0
                    )
                    phrase_codes[phrase_codes_offset:phrase_codes_endpos] = (
                        phrase_chunk.codes
                    )
                    phrase_residuals[phrase_codes_offset:phrase_codes_endpos] = (
                        phrase_chunk.residuals
                    )
                    phrase_codes_offset = phrase_codes_endpos

        cls_cls = cls(cls_codes, cls_residuals) if use_cls else None
        tok_cls = cls(tok_codes, tok_residuals)
        phrase_cls = cls(phrase_codes, phrase_residuals) if use_phrase else None
        return cls_cls, tok_cls, phrase_cls

    @classmethod
    def load(
        cls, index_path, chunk_idx
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cls_codes, tok_codes, phrase_codes = cls.load_codes(index_path, chunk_idx)
        cls_residuals, tok_residuals, phrase_residuals = cls.load_residuals(
            index_path, chunk_idx
        )

        if cls_codes is not None:
            cls_cls = cls(cls_codes, cls_residuals)
        else:
            cls_cls = None

        tok_cls = cls(tok_codes, tok_residuals)

        if phrase_codes is not None:
            phrase_cls = cls(phrase_codes, phrase_residuals)
        else:
            phrase_cls = None

        return cls_cls, tok_cls, phrase_cls

    @classmethod
    def load_codes(
        self, index_path: str, chunk_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get paths
        cls_codes_path = os.path.join(index_path, f"{chunk_idx}-cls.codes.pt")
        tok_codes_path = os.path.join(index_path, f"{chunk_idx}-tok.codes.pt")
        phrase_codes_path = os.path.join(index_path, f"{chunk_idx}-phrase.codes.pt")

        # Load codes
        tok_codes = torch.load(tok_codes_path, map_location="cpu")

        if os.path.exists(cls_codes_path):
            cls_codes = torch.load(cls_codes_path, map_location="cpu")
        else:
            cls_codes = None

        if os.path.exists(phrase_codes_path):
            phrase_codes = torch.load(phrase_codes_path, map_location="cpu")
        else:
            phrase_codes = None

        return cls_codes, tok_codes, phrase_codes

    @classmethod
    def load_residuals(
        self, index_path, chunk_idx
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cls_residuals_path = os.path.join(index_path, f"{chunk_idx}-cls.residuals.pt")
        tok_residuals_path = os.path.join(index_path, f"{chunk_idx}-tok.residuals.pt")
        phrase_residuals_path = os.path.join(
            index_path, f"{chunk_idx}-phrase.residuals.pt"
        )
        if os.path.exists(cls_residuals_path):
            cls_residuals = torch.load(cls_residuals_path, map_location="cpu")
        else:
            cls_residuals = None
        tok_residuals = torch.load(tok_residuals_path, map_location="cpu")
        if os.path.exists(phrase_residuals_path):
            phrase_residuals = torch.load(phrase_residuals_path, map_location="cpu")
        else:
            phrase_residuals = None
        return cls_residuals, tok_residuals, phrase_residuals

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
