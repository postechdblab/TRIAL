import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import logging
import os
from typing import *

import hkkang_utils.file as file_utils
import tqdm

from eagle.phrase.clean import unidecode_text
from scripts.analysis.utils import DATASET_NAMES

logger = logging.getLogger("CleanDatasetText")
DIR_PATH = "/root/EAGLE/data/"


def main() -> None:
    for dataset_name in tqdm.tqdm(DATASET_NAMES, desc="Datasets"):
        query_path = os.path.join(DIR_PATH, dataset_name, "queries.jsonl")
        corpus_path = os.path.join(DIR_PATH, dataset_name, "corpus.jsonl")

        logger.info(f"Reading {dataset_name} queries ...")
        queries = file_utils.read_jsonl_file(query_path)

        for query in tqdm.tqdm(queries, desc="Cleaning queries"):
            query["text"] = unidecode_text(query["text"])

        # Write the cleaned queries
        logger.info(f"Writing {dataset_name} queries ...")
        file_utils.write_jsonl_file(queries, query_path)

        logger.info(f"Reading {dataset_name} corpus ...")
        corpus = file_utils.read_jsonl_file(corpus_path)

        for doc in tqdm.tqdm(corpus, desc="Cleaning corpus"):
            doc["text"] = unidecode_text(doc["text"])

        # Write the cleaned corpus
        logger.info(f"Writing {dataset_name} corpus ...")
        file_utils.write_jsonl_file(corpus, corpus_path)

    logger.info("Done!")

    return None


if __name__ == "__main__":
    main()
