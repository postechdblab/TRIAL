import argparse
import logging
import os
from typing import *

import hkkang_utils.file as file_utils
import hkkang_utils.list as list_utils
import hkkang_utils.slack as slack_utils
import tqdm
from transformers import AutoTokenizer

from scripts.analysis.check_token_length import draw_histogram, get_token_stats
from scripts.evaluate.utils import load_dataset
from scripts.utils import BEIR_DATASET_NAMES, LOTTE_DATASET_NAMES

DATASET_DIR = "/root/EAGLE/data"
MAX_QUERY_TOKENS = 32
MAX_DOC_TOKENS = 220

logger = logging.getLogger("DatasetStats")


def main(
    dataset_dir: str,
    dataset_name: str,
    tokenizer,
    output_dir: str = None,
    draw_historgram: bool = False,
    max_q_tokens: int = MAX_QUERY_TOKENS,
    max_d_tokens: int = MAX_DOC_TOKENS,
) -> None:
    """Figure out:
    - Number of unqiue queries
    - Number of documents
    - Number of gold documents
    - Number of gold documents per query
    - The max, min, average and std of the number of query tokens
    - The max, min, average and std of the number of document tokens
    """
    # Load data
    corpus_path = os.path.join(dataset_dir, f"{dataset_name}/corpus.jsonl")
    corpus = file_utils.read_json_file(corpus_path, auto_detect_extension=True)
    data: List = load_dataset(
        dataset_dir=DATASET_DIR,
        dataset_name=dataset_name,
        return_unique=True,
    )

    # Number of queries
    num_queries = len(set(d[0] for d in data))
    num_of_every_documents = len(corpus)
    num_of_gold_documents = len(
        set(list_utils.do_flatten_list([item[2] for item in data]))
    )
    num_of_gold_documents_per_query = sum([len(item[2]) for item in data]) / num_queries

    # Get query token stats
    (
        q_max_text_tokens,
        q_min_text_tokens,
        q_avg_text_tokens,
        q_std,
        q_num_bad_queries,
        q_token_lengths,
        q_with_word_to_multi_tokens_cnt,
    ) = get_token_stats(
        tokenizer=tokenizer,
        texts=[item[1] for item in data],
        max_token_threshold=max_q_tokens,
    )

    # Get document token stats
    (
        d_max_text_tokens,
        d_min_text_tokens,
        d_avg_text_tokens,
        d_std,
        d_num_bad_queries,
        d_token_lengths,
        d_with_word_to_multi_tokens_cnt,
    ) = get_token_stats(
        tokenizer=tokenizer,
        texts=[item["text"] for item in corpus[:100000]],
        max_token_threshold=max_d_tokens,
    )

    # Print stats
    logger.info(f"Number of unique queries: {num_queries}")
    logger.info(f"Number of documents: {num_of_every_documents}")
    logger.info(f"Number of gold documents: {num_of_gold_documents}")
    logger.info(
        f"Number of gold documents per query: {round(num_of_gold_documents_per_query, 1)}"
    )
    logger.info(
        f"Query token stats: max={q_max_text_tokens}, min={q_min_text_tokens}, avg={q_avg_text_tokens}, std={q_std}, num_bad_queries={q_num_bad_queries}"
    )
    logger.info(
        f"Document token stats: max={d_max_text_tokens}, min={d_min_text_tokens}, avg={d_avg_text_tokens}, std={d_std}, num_bad_docs={d_num_bad_queries}\n"
    )

    if draw_historgram:
        draw_histogram(
            token_lengths=q_token_lengths,
            dataset_name=dataset_name,
            suffix="query",
            output_dir=output_dir,
        )
        draw_histogram(
            token_lengths=d_token_lengths,
            dataset_name=dataset_name,
            suffix="document",
            output_dir=output_dir,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Start end-to-end test.")
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset to evaluate",
        choices=[f"beir-{name}" for name in BEIR_DATASET_NAMES]
        + [f"lotte-{name}" for name in LOTTE_DATASET_NAMES]
        + ["all"],
        required=True,
    )
    parser.add_argument("--output_dir", type=str, help="Output directory", default=None)
    parser.add_argument(
        "--slack", action="store_true", help="Whether to send slack notification"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Set logging hanlders
    handlers = [logging.StreamHandler()]
    if args.output_dir:
        handlers.add(logging.FileHandler(f"Dataset_stat_{args.dataset}.log"))

    # Set logging config
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
        handlers=handlers,
    )

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    if args.dataset == "all":
        eval_datasets = BEIR_DATASET_NAMES
    else:
        eval_datasets = [args.dataset]

    with slack_utils.notification(
        channel="question-answering",
        success_msg=f"Succeeded to compute dataset stats for {args.dataset} dataset",
        error_msg=f"Falied to compute dataset stats for {args.dataset} dataset",
        disable=not args.slack,
    ):
        # Evaluate each dataset
        for dataset_name in tqdm.tqdm(eval_datasets, desc="Processing datasets"):
            logger.info(f"Computing stats for {dataset_name}...")
            main(
                dataset_dir=DATASET_DIR,
                dataset_name=dataset_name,
                tokenizer=tokenizer,
                output_dir=args.output_dir,
            )
    logger.info(f"Done!")
