from collections import defaultdict
from typing import *

import tqdm
import ujson

from colbert.data import Ranking
from colbert.infra import Run
from colbert.infra.provenance import Provenance
from colbert.utils.utils import print_message, zipstar
from model.distillation.scorer import Scorer


class RankingScorer:
    def __init__(self, scorer: Scorer, ranking: Union[Ranking, List]):
        self.scorer = scorer
        if type(ranking) == List:
            self.ranking = ranking
        else:
            self.ranking = ranking.tolist()
        self.__provenance = Provenance()

        print_message(f"#> Loaded ranking with {len(self.ranking)} qid--pid pairs!")

    def provenance(self):
        return self.__provenance

    def run(self):
        print_message(f"#> Starting..")

        qids, pids, *_ = zipstar(self.ranking)
        distillation_scores = self.scorer.launch(qids, pids)

        scores_by_qid = defaultdict(list)

        for qid, pid, score in tqdm.tqdm(zip(qids, pids, distillation_scores)):
            scores_by_qid[qid].append((pid, score))

        with Run().open("distillation_scores.jsonl", "w") as f:
            for qid in tqdm.tqdm(scores_by_qid):
                obj = (int(qid), *scores_by_qid[qid])
                f.write(ujson.dumps(obj) + "\n")

            output_path = f.name
            print_message(f"#> Saved the distillation_scores to {output_path}")

        with Run().open(f"{output_path}.meta", "w") as f:
            d = {}
            line = ujson.dumps(d, indent=4)
            f.write(line)

        return output_path
