import argparse
import logging
import os
from typing import *

import hkkang_utils.file as file_utils

from colbert.noun_extraction.identify_noun import extract_nouns_indices_batch

logger = logging.getLogger("AnalyzeNounExtraction")

base_dir = "/root/ColBERT/data/"
hotpotqa_corpus_path = os.path.join(base_dir, "hotpotqa/collection.tsv")
hotpotqa_query_path = os.path.join(base_dir, "hotpotqa/dev.json")
msmarco_corpus_path = os.path.join(base_dir, "msmarco/collection.tsv")
msmarco_query_path = os.path.join(base_dir, "msmarco/queries.dev.tsv")


def load_data_to_analyze(dataset: str, examine_query: bool) -> List[str]:
    if dataset == "hotpotqa":
        if examine_query:
            data = file_utils.read_json_file(hotpotqa_query_path)
            data: List[str] = [item["question"] for item in data]
        else:
            data = file_utils.read_csv_file(
                hotpotqa_corpus_path, delimiter="\t", first_row_as_header=False
            )
            data: List[str] = [item[1] for item in data]
    elif dataset == "msmarco":
        if examine_query:
            data = file_utils.read_csv_file(
                msmarco_query_path, delimiter="\t", first_row_as_header=False
            )
            data: List[str] = [item[1] for item in data]
        else:
            data = file_utils.read_csv_file(
                msmarco_corpus_path, delimiter="\t", first_row_as_header=False
            )
            data: List[str] = [item[1] for item in data]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return data


def main(dataset: str, examine_query: bool, sample_num: int) -> None:
    # Load data
    logger.info(f"Loading {dataset} data...")
    data: List[str] = load_data_to_analyze(dataset, examine_query)
    logger.info(f"Loaded {len(data)} data.")
    if sample_num:
        data = data[:sample_num]
        logger.info(f"Sampled {len(data)} data.")

    logger.info(f"Extracting nouns...")
    noun_indices_batch: List[List[int]] = extract_nouns_indices_batch(data)

    for idx, noun_indices in enumerate(noun_indices_batch):
        datum = data[idx]
        logger.info(f"Idx {idx}: {datum}")
        nouns = [datum[i:j] for i, j in noun_indices]
        logger.info(f"Nouns: {nouns}\n")

    logger.info("\nDone!")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path for the collection.tsv file",
        choices=["msmarco", "hotpotqa"],
    )
    parser.add_argument(
        "--target",
        type=str,
        help="Total number of paritions",
        choices=["query", "passage"],
    )
    parser.add_argument(
        "--sample_num", type=int, help="Number of data to examine", default=100
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    args = parse_args()

    main(
        dataset=args.dataset,
        examine_query=args.target == "query",
        sample_num=args.sample_num,
    )
