import random
from typing import *

import numpy as np
import torch

from eagle.index.corpus import Corpus

TYPICAL_DOCLEN = 120


class Indexer:
    def __init__(self, cfg) -> None:
        # TODO: Implement below
        self.cfg = cfg
        self.rank = None
        self.corpus: Corpus = None
        self.encoder = None

    def _setup(self):
        """Calcuate and saves plan.json for the whole corpus."""
        pass

    @property
    def num_doc(self) -> int:
        return len(self.corpus)

    def sample_pids(self) -> Set[int]:
        num_sample_pids = min(
            1 + 16 * np.sqrt(TYPICAL_DOCLEN * self.num_doc), self.num_doc
        )
        return set(random.sample(range(self.num_doc), num_sample_pids))

    def sample_embeddings(self, sampled_pids: List[int]) -> None:
        # Extract documents
        local_sample = [
            passage
            for pid, passage in self.corpus.enumerate(rank=self.rank)
            if pid in sampled_pids
        ]
        # encode text
        local_sample_embs, doclens = self.encoder.encode_passages(local_sample)

    def encode_passage(self, passages: List[str]) -> Tuple[torch.Tensor, List[int]]:
        with torch.inference_mode():
            embs, doclens = [], []
            for passages_batch in list_utils.batch(passages, self.cfg.bsize):
                embs_, doclens_ = self.encoder.encode_passages(
                    passages_batch,
                    bsize=self.cfg.bsize,
                    showprogress=True,
                )
                embs.append(embs_)
                doclens.extend(doclens_)
            embs = torch.cat(embs)

        return embs, doclens
