import os
from typing import *

import hkkang_utils.data as data_utils
import hkkang_utils.file as file_utils
from omegaconf import DictConfig

MINCHUNKSIZE = 25000


@data_utils.dataclass
class Document:
    _id: int
    text: str
    title: str

    def __str__(self) -> str:
        return f"{self.title} | {self.text}"


class Corpus:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.data: List[Document] = [
            Document(**item) for item in file_utils.read_jsonl_file(self.corpus_path)
        ]
        self.word_indices = file_utils.read_pickle_file(self.word_range_path)
        self.phrase_indices = file_utils.read_pickle_file(self.phrase_range_path)

    def __len__(self) -> int:
        return len(self.data)

    def get_document(self, idx: int) -> Document:
        return self.data[idx]

    def get_document_by_id(self, pid: int) -> List[Document]:
        docs = [doc for doc in self.data if doc._id == pid]
        return docs

    @property
    def corpus_path(self) -> str:
        return os.path.join(
            self.cfg.dataset.dir_path,
            self.cfg.dataset.name,
            self.cfg.dataset.corpus_file,
        )

    @property
    def word_range_path(self) -> str:
        return os.path.join(
            self.cfg.dataset.dir_path,
            self.cfg.dataset.name,
            self.cfg.dataset.d_word_range_file,
        )

    @property
    def phrase_range_path(self) -> str:
        return os.path.join(
            self.cfg.dataset.dir_path,
            self.cfg.dataset.name,
            self.cfg.dataset.d_phrase_range_file,
        )

    def get_chunk_size(self, world_size: int) -> int:
        assert len(self), f"The corpus is empty."
        return min(MINCHUNKSIZE, len(self) // world_size)

    def get_num_chunks(self, world_size: int) -> int:
        return len(self) // self.get_chunk_size(world_size)

    def enumerate(
        self, rank: int = 0, world_size: int = 1
    ) -> Iterator[Tuple[int, Document]]:
        for _, doc_idx, docs, word_ranges, phrase_ranges in self.enumerate_chunk(
            rank=rank, world_size=world_size
        ):
            for local_idx, doc in enumerate(docs):
                word_range = word_ranges[local_idx] if word_ranges else None
                phrase_range = phrase_ranges[local_idx] if phrase_ranges else None
                yield (doc_idx + local_idx, doc, word_range, phrase_range)

    def enumerate_chunk(
        self, rank: int = 0, world_size: int = 1
    ) -> Iterator[Tuple[int, List[Document]]]:
        chunk_size = self.get_chunk_size(world_size)
        for doc_idx in range(0, len(self.data), chunk_size):
            chunk_idx = doc_idx // chunk_size
            # yield if the chunk index is the same as the rank
            if chunk_idx % world_size == rank:
                docs = self.data[doc_idx : doc_idx + chunk_size]
                word_ranges = (
                    self.word_indices[doc_idx : doc_idx + chunk_size]
                    if self.word_indices
                    else None
                )
                phrase_ranges = (
                    self.phrase_indices[doc_idx : doc_idx + chunk_size]
                    if self.phrase_indices
                    else None
                )
                yield (chunk_idx, doc_idx, docs, word_ranges, phrase_ranges)
