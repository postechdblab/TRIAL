import itertools
from typing import *

import hkkang_utils.list as list_utils

MINCHUNKSIZE = 25000


class Corpus:
    def __init__(self) -> None:
        self.nranks = 1
        # TODO: Implement below
        self.data = None

    def __len__(self) -> int:
        return len(self.data)

    @property
    def chunksize(self) -> int:
        assert len(self), f"The corpus is empty."
        return min(MINCHUNKSIZE, len(self) // self.nranks)

    @property
    def num_chunks(self) -> int:
        return len(self) // self.chunksize

    def enumerate(self, rank: Optional[int] = None) -> Iterator[Tuple[int, str]]:
        for _, doc_idx, docs in self.enumerate_batches(rank=rank):
            for local_idx, doc in enumerate(docs):
                yield (doc_idx + local_idx, doc)

    def enumerate_chunk(
        self, rank: Optional[int] = None
    ) -> Iterator[Tuple[int, List[str]]]:
        for doc_idx in range(len(self.corpus), self.chunksize):
            chunk_idx = doc_idx // self.chunksize
            # yield if the chunk index is the same as the rank
            if rank is not None and chunk_idx % self.nranks == rank:
                yield (chunk_idx, doc_idx, self.data[doc_idx])
