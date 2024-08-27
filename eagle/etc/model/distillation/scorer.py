from typing import *


import tqdm

from colbert.infra import Run
from colbert.infra.launcher import Launcher
from colbert.utils.utils import flatten
from model import BaseRetriever


class Scorer:
    def __init__(
        self,
        queries: Dict,
        collection: Dict,
        model: BaseRetriever,
        maxlen: int = 180,
        bsize: int = 256,
    ):
        self.queries = queries
        self.collection = collection
        self.model = model

        self.maxlen = maxlen
        self.bsize = bsize

    def launch(self, qids, pids):
        launcher = Launcher(self._score_pairs_process, return_all=True)
        outputs = launcher.launch(Run().config, qids, pids)

        return flatten(outputs)

    def _score_pairs_process(self, config, qids, pids):
        assert len(qids) == len(pids), (len(qids), len(pids))
        share = 1 + len(qids) // config.nranks
        offset = config.rank * share
        endpos = (1 + config.rank) * share

        return self._score_pairs(
            qids[offset:endpos], pids[offset:endpos], show_progress=(config.rank < 1)
        )

    def _score_pairs(
        self, qids: List[str], pids: List[int], show_progress: bool = False
    ) -> List[float]:
        assert len(qids) == len(pids), (len(qids), len(pids))

        scores: List[float] = []
        for offset in tqdm.tqdm(
            range(0, len(qids), self.bsize), disable=(not show_progress)
        ):
            # Get the query and passages texts
            endpos = offset + self.bsize
            queries_: List[str] = [
                self.queries[str(qid)] for qid in qids[offset:endpos]
            ]
            passages_: List[str] = [self.collection[pid] for pid in pids[offset:endpos]]
            # Compute scores
            scores.extend(
                self.model.calculate_score_by_text_batch(
                    queries=queries_, doc_texts=passages_
                )
            )

        Run().print(f"Returning with {len(scores)} scores")

        return scores


# LONG-TERM TODO: This can be sped up by sorting by length in advance.
