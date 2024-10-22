import logging
import os
from typing import *

import hkkang_utils.data as data_utils
import hkkang_utils.file as file_utils
import tqdm
from omegaconf import DictConfig

MINCHUNKSIZE = 25000

logger = logging.getLogger("Corpus")


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
        self._load_data()

    def __len__(self) -> int:
        return len(self.data)

    def _load_data(self) -> None:
        # Load corpus data
        logger.info(f"Loading corpus data from {self.corpus_path}..")
        corpus = file_utils.read_jsonl_file(self.corpus_path)
        logger.info(f"Loaded {len(corpus)} documents. Converting to Document objects..")
        self.data = [Document(**item) for item in tqdm.tqdm(corpus)]

        return None

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

    def get_chunk_size(self, world_size: int) -> int:
        assert len(self), f"The corpus is empty."
        return min(MINCHUNKSIZE, len(self) // world_size)

    def get_num_chunks(self, world_size: int) -> int:
        return len(self) // self.get_chunk_size(world_size)

    def enumerate(
        self, rank: int = 0, world_size: int = 1
    ) -> Iterator[Tuple[int, Document]]:
        for _, doc_idx, docs in self.enumerate_chunk(rank=rank, world_size=world_size):
            for local_idx, doc in enumerate(docs):
                yield (doc_idx + local_idx, doc)

    def enumerate_chunk(
        self, rank: int = 0, world_size: int = 1
    ) -> Iterator[Tuple[int, List[Document]]]:
        chunk_size = self.get_chunk_size(world_size)
        for doc_idx in range(0, len(self.data), chunk_size):
            chunk_idx = doc_idx // chunk_size
            # yield if the chunk index is the same as the rank
            if chunk_idx % world_size == rank:
                docs = self.data[doc_idx : doc_idx + chunk_size]
                yield (chunk_idx, doc_idx, docs)
