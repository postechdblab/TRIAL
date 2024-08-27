import abc
import functools
import logging
from typing import *

import hkkang_utils.data as data_utils
import numpy as np

from model.utils import Document

logger = logging.getLogger("BaseRetriever")


@data_utils.dataclass
class RetrievalResult:
    score: float
    doc: Document
    token_scores: np.ndarray = data_utils.field(default_factory=None)


class BaseRetriever:
    def __init__(self, corpus: Optional[List[str]] = None):
        self.corpus: List[str] = [] if corpus is None else corpus  # List of documents

    @functools.cached_property
    def titles(self) -> List[str]:
        title_text_split_keyword = " | "
        if title_text_split_keyword in self.corpus[1]:
            return [doc[: doc.index(title_text_split_keyword)] for doc in self.corpus]
        logger.debug(f"No title found in the corpus.")
        return []

    def get_doc_title(self, pid: int) -> str:
        """Get the title of a document by its passage id."""
        pid = int(pid)
        if len(self.titles) == 0:
            return ""
        return self.titles[pid]

    def get_doc_text(self, pid: int) -> str:
        """Get the text of a document by its passage id."""
        pid = int(pid)
        return self.corpus[pid]

    def retrieve(self, query: str, topk: int = 100) -> List[RetrievalResult]:
        """Retrieve top-k documents for a given query."""
        return self.retrieve_batch(queries=[query], topk=topk)[0]

    def calculate_score(
        self, query: str, pid: int = None, title: str = None, text: str = None
    ) -> RetrievalResult:
        """Get the retrieval score of a document for a given query."""
        return self.calculate_score_batch(
            queries=[query],
            pids=[pid] if pid else None,
            titles=[title] if title else None,
            doc_texts=[text] if text else None,
        )[0]

    def calculate_score_batch(
        self,
        queries: List[str],
        pids: List[int] = None,
        titles: List[str] = None,
        doc_texts: List[str] = None,
    ) -> List[RetrievalResult]:
        """Get the retrieval scores of a list of documents for a list of queries."""
        # Parse the arguments
        # Check one of pids, titles, and text is not None
        if pids is None and titles is None and doc_texts is None:
            raise ValueError("One of pids, titles, and text must be not None.")
        # Check the length of pids, titles, and text are the same
        if pids and pids[0] is not None:
            assert len(pids) == len(
                queries
            ), f"Length of pids and queries are different: {len(pids)} vs {len(queries)}"
            # Get document texts
            doc_texts = [self.get_doc_text(pid) for pid in pids]
            # Get document titles
            titles = [self.get_doc_title(pid) for pid in pids]
        elif titles and titles[0] is not None:
            assert len(titles) == len(
                queries
            ), f"Length of titles and queries are different: {len(titles)} vs {len(queries)}"
            # Get document texts
            doc_texts = [
                self.get_doc_text(self.titles.index(title)) for title in titles
            ]
            # Get document pids
            pids = [self.titles.index(title) for title in titles]
        elif doc_texts and doc_texts[0] is not None:
            assert len(doc_texts) == len(
                queries
            ), f"Length of text and queries are different: {len(doc_texts)} vs {len(queries)}"
            # Get document pids
            pids = [self.corpus.index(text) for text in doc_texts]
            # Get document titles
            titles = [self.titles[pid] for pid in pids]
        # Create documents
        docs = [
            Document(id=str(pid), title=title, text=text)
            for pid, title, text in zip(pids, titles, doc_texts)
        ]
        # Get retrieval scores
        return self.calculate_score_by_doc_batch(queries=queries, docs=docs)

    def rank_docs(
        self,
        query: str,
        doc_texts: List[Document] = None,
        pids: List[int] = None,
        titles: List[str] = None,
    ) -> List[Tuple[int, int]]:
        """Rerank a list of documents for a given query."""
        if pids:
            input_data = pids
        elif titles:
            input_data = titles
        elif doc_texts:
            input_data = doc_texts
        else:
            raise ValueError("One of pids, titles, and text must be given.")
        # Duplicate the query
        queries = [query] * len(input_data)
        # Get scores
        results: List[RetrievalResult] = self.calculate_score_batch(
            queries=queries, pids=pids, titles=titles, doc_texts=doc_texts
        )
        scores: List[float] = [result.score for result in results]

        # Generate output
        # Rank the documents by the scores
        return sorted(zip(scores, input_data), key=lambda x: x[0], reverse=True)

    # List of abstract methods to be implemented in a subclass
    @abc.abstractmethod
    def retrieve_batch(
        self, queries: List[str], topk: int = 100, return_scores: bool = False, **kwargs
    ) -> List[List[RetrievalResult]]:
        """Retrieve top-k documents for each query in a batch."""
        raise NotImplementedError("Implement this method in a subclass.")

    @abc.abstractmethod
    def calculate_score_by_doc_batch(
        self, queries: List[str], docs: List[Document]
    ) -> List[float]:
        """Get the retrieval score of a document for a given query."""
        raise NotImplementedError("Implement this method in a subclass.")

    @abc.abstractmethod
    def create_index(self, index_name: str, corpus_path: str) -> None:
        """Create an index for a given corpus."""
        raise NotImplementedError("Implement this method in a subclass.")
