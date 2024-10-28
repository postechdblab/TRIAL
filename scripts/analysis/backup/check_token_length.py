import argparse
import csv
import logging
import os
import sys
from typing import *

import hkkang_utils.file as file_utils
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from transformers import AutoTokenizer

from scripts.utils import BEIR_DATASET_NAMES

# Data path
DATASET_DIR = "/root/EAGLE/data"

logger = logging.getLogger("Evaluate")

csv.field_size_limit(sys.maxsize)


def read_in_beir_docs(dir_path: str, dataset_name: str) -> List[str]:
    path = os.path.join(dir_path, f"{dataset_name}/collection.tsv")
    data = file_utils.read_csv_file(path, delimiter="\t", first_row_as_header=True)
    return [d["text"] for d in data]


def read_in_beir_queries(dir_path: str, dataset_name: str) -> List[str]:
    qrels_path = os.path.join(dir_path, f"{dataset_name}/qrels.test.tsv")
    queries_path = os.path.join(dir_path, f"{dataset_name}/queries.jsonl")
    qrels = file_utils.read_csv_file(qrels_path, delimiter="\t")
    queries = file_utils.read_jsonl_file(queries_path)
    queries_dict = {q["_id"]: q["text"] for q in queries}
    # Get text
    queries_text: List[str] = []
    skipping_qids = []
    for item in qrels:
        qid = item["query-id"]
        if qid in queries_dict:
            queries_text.append(queries_dict[qid])
        else:
            skipping_qids.append(qid)
    if skipping_qids:
        logger.warning(f"Skipping qids: {skipping_qids}")
    return queries_text


def draw_histogram(
    token_lengths: List[int],
    dataset_name: str,
    suffix: str = None,
    output_dir: str = None,
) -> None:
    """Draw histogram and mark the average token length"""
    # Get title
    title = f"Token length distribution ({dataset_name})"
    if suffix:
        title += f" ({suffix})"
    # Get filename
    file_name = f"token_length_distribution_{dataset_name}"
    if suffix:
        file_name += f"_{suffix}"
    if output_dir:
        file_name = os.path.join(output_dir, file_name)
    # Draw histogram
    plt.hist(
        token_lengths,
        bins=min(100, max(token_lengths)),
        range=(0, min(max(token_lengths), 100)),
    )
    plt.title(title)
    plt.axvline(x=sum(token_lengths) / len(token_lengths), color="red")
    plt.xlabel("Token length")
    plt.ylabel("Frequency")
    plt.savefig(f"{file_name}.png")
    plt.close()


def get_token_stats(
    tokenizer, texts: List[str], max_token_threshold: int = None
) -> Tuple[int, int, float, float, int, List[int], int]:
    """Get the statistics of the number of tokens in the texts.

    :param tokenizer: Tokenizer
    :type tokenizer: _type_
    :param texts: List of texts
    :type texts: List[str]
    :return: Max, min, average, and std of tokens in the text. Also, return the number of texts with token length > max_token_threshold and the list of token lengths
    :rtype: Tuple[int, int, float, float, int, List[int]]
    """
    results = tokenizer(texts)
    input_ids = results["input_ids"]
    attention_mask = results["attention_mask"]

    # Get max token length
    max_text_tokens = max([sum(attention_mask[i]) for i in range(len(input_ids))])

    # Get min token length
    min_text_tokens = min([sum(attention_mask[i]) for i in range(len(input_ids))])

    # Get average token length
    sum_text_tokens = sum([sum(attention_mask[i]) for i in range(len(input_ids))])
    avg_text_tokens = sum_text_tokens / len(input_ids)

    # Get variance
    token_lengths = [sum(attention_mask[i]) for i in range(len(input_ids))]
    variance = np.var(token_lengths)

    # Get standard deviation
    std = np.std(token_lengths)

    input_toks = list(map(tokenizer.convert_ids_to_tokens, input_ids))
    # Count words breaking into multiple tokens
    q_with_word_to_multi_tokens_cnt = 0
    for i, toks in enumerate(input_toks):
        for tok in toks:
            if len(tok) > 2 and tok[:2] == "##":
                q_with_word_to_multi_tokens_cnt += 1
                break

    # Calculate the number of texts with token length > max_token_threshold
    num_bad_queries = 0
    if max_token_threshold:
        num_bad_queries = sum(
            [1 for length in token_lengths if length > max_token_threshold]
        )

    return (
        max_text_tokens,
        min_text_tokens,
        round(avg_text_tokens, 2),
        round(std, 2),
        num_bad_queries,
        token_lengths,
        q_with_word_to_multi_tokens_cnt,
    )


def main(
    dataset_name: str, analyze_doc: bool = False, target_max_len: int = 64
) -> None:
    """Get the max token length"""
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", model_max_length=512)

    # Read in text data
    logger.info(f"Reading data...")

    if analyze_doc:
        texts = read_in_beir_docs(DATASET_DIR, dataset_name)
        sample_size = 100000
        texts = texts[:sample_size]
    else:
        texts = read_in_beir_queries(DATASET_DIR, dataset_name)

    # Get the stats
    logger.info(f"Getting stats...")
    (
        max_text_length,
        min_text_length,
        avg_token_length,
        std,
        num_bad_queries,
        token_lengths,
        num_word_to_multi_toks,
    ) = get_token_stats(tokenizer, texts)

    logger.info(f"Dataset: {dataset_name} ({'document' if analyze_doc else 'query'})")
    logger.info(f"Total number of texts: {len(texts)}")
    logger.info(f"Max token length: {max_text_length}")
    logger.info(f"Average token length: {avg_token_length}")
    logger.info(f"standard deviation: {std}")
    logger.info(
        f"Number of texts with token length > {target_max_len}: {num_bad_queries}"
    )
    logger.info(
        f"Number of texts with word broken into multiple tokens: {num_word_to_multi_toks}"
    )

    # Draw histogram and mark the average token length
    draw_histogram(
        token_lengths, dataset_name, suffix="document" if analyze_doc else "query"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Start end-to-end test.")
    parser.add_argument(
        "--dataset",
        type=str,
        help="Name of the dataset to analyze",
        choices=BEIR_DATASET_NAMES + ["all"],
        default="trec-covid",
    )
    parser.add_argument(
        "--is_doc", action="store_true", help="Whether to analyze documents"
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    args = parse_args()
    target_max_len = 220 if args.is_doc else 32
    if args.dataset == "all":
        datasets = BEIR_DATASET_NAMES
    else:
        datasets = [args.dataset]

    for dataset in tqdm.tqdm(datasets, desc="Analyzing datasets"):
        main(
            dataset_name=dataset, analyze_doc=args.is_doc, target_max_len=target_max_len
        )

    logger.info(f"Done!")
