import logging
import os
from typing import *

import hkkang_utils.file as file_utils
import hkkang_utils.slack as slack_utils
import tqdm

from colbert.data.ranking import Ranking
from colbert.distillation.ranking_scorer import RankingScorer
from colbert.distillation.scorer import Scorer
from colbert.infra import Run, RunConfig
from model.cross_encoder import MiniLLMRetriever
from model.dual_encoder import E5MistralRetriever
from scripts.retrieval import ColBERTRetriever
from scripts.utils import read_collection, read_queries

# Data path
DATASET_DIR = "/root/EAGLE/data"
collection_path = os.path.join(DATASET_DIR, "msmarco_old/collection.tsv")
train_query_path = os.path.join(DATASET_DIR, "msmarco_old/queries.train.tsv")
train_gold_path = os.path.join(DATASET_DIR, "msmarco_old/qrels.train.tsv")

# Retriever path
ROOT = "/root/EAGLE/experiments/"
EXPERIMENT = "msmarco_old"
INDEX = "msmarco.distillation.nbits=2"
# Model configs
SKIP_PADDING = False
DEBUG = False

logger = logging.getLogger("CreateDistillationData")

RANKING_CACHE_PATH = "/root/EAGLE/ranking.jsonl.pkl"
HARD_NEGATIVE_DATA_PATH = "/root/EAGLE/data/msmarco_old/train_data_nhards256.jsonl"


def main(
    teacher_name: str, ranking_cache_path: str = None, is_debug: bool = DEBUG
) -> None:
    with Run().context(
        RunConfig(
            nranks=1,
            experiment=f"msmarco_old/distillation",
            name=f"{teacher_name}_{INDEX}",
        )
    ):
        # Launch trained model and create Ranking object
        logger.info(f"Reading data...")
        query_dict = read_queries(train_query_path)
        collection = read_collection(collection_path)
        # Convert key type from str to int
        collection = {int(key): collection[key] for key in collection.keys()}
        # Sample if debug
        if is_debug:
            query_dict = {
                int(key): query_dict[key] for key in list(query_dict.keys())[:100]
            }

        # Get ranking object
        if HARD_NEGATIVE_DATA_PATH:
            logger.info(f"Loading hard negative data from {HARD_NEGATIVE_DATA_PATH}...")
            hard_negative_data = file_utils.read_jsonl_file(HARD_NEGATIVE_DATA_PATH)
            logger.info(
                f"Loaded {len(hard_negative_data)} queries from {HARD_NEGATIVE_DATA_PATH} !"
            )
            # Format data
            new_hard_negative_data = []
            for d in tqdm.tqdm(
                hard_negative_data, desc="Formatting hard negative data"
            ):
                qid = d[0]
                for i in range(0, min(len(d), 165)):
                    pid = d[i]
                    new_hard_negative_data.append([qid, pid])
            hard_negative_data = new_hard_negative_data
            ranking = Ranking(data=hard_negative_data)
        elif ranking_cache_path:
            logger.info(f"Loading ranking object from {ranking_cache_path}...")
            ranking = Ranking(path=ranking_cache_path)
            logger.info(
                f"Loaded {len(ranking.data)} queries from {ranking_cache_path} !"
            )
        else:
            # Initialize retriever
            logger.info(f"Initializing retriever...")
            retriever = ColBERTRetriever(
                root=ROOT, index=INDEX, experiment=EXPERIMENT, skip_padding=SKIP_PADDING
            )

            # Search for top-k passages for each query
            logger.info(f"Searching top-k passages for {len(query_dict)} query...")
            # TODO: Need to add the gold pid as the first pid in the ranking
            raise NotImplementedError(
                "Need to add the gold pid as the first pid in the ranking"
            )
            ranking = retriever.retrieve_as_ranking(queries_dict=query_dict)
            # Save ranking object
            logger.info(f"Saving ranking object...")
            ranking.save(new_path="ranking.jsonl")

        # Pass the Ranking object to RankingScorer
        logger.info(f"Initializing RankingScorer...")
        if teacher_name == "minillm":
            teacher = MiniLLMRetriever()
        elif teacher_name == "mistral":
            teacher = E5MistralRetriever()
        else:
            raise RuntimeError(f"Unknown teacher name: {teacher_name}")

        # Initialize RankingScorer
        scorer = Scorer(
            queries=query_dict,
            collection=collection,
            model=teacher,
            bsize=1024,
            maxlen=180,
        )
        ranking_scorer = RankingScorer(scorer=scorer, ranking=ranking)

        # launch it
        logger.info("Launching RankingScorer...")
        ranking_scorer.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    with slack_utils.notification(
        channel="question-answering",
        success_msg=f"Succeeded to create distillation data for msmarco!",
        error_msg=f"Falied to create distillation data for msmarco!",
    ):
        main(teacher_name="minillm")
    logger.info(f"Done!")
