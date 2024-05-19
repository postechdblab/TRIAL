import argparse
import logging
import os
from typing import *

import hkkang_utils.file as file_utils
import hkkang_utils.list as list_utils
import hkkang_utils.slack as slack_utils
import tqdm

from colbert.infra import ColBERTConfig
from colbert.modeling.tokenization.doc_tokenization import DocTokenizer
from colbert.modeling.tokenization.query_tokenization import QueryTokenizer
from colbert.modeling.tokenization.utils import get_phrase_indices
from colbert.noun_extraction.identify_noun import SpacyModel, Text
from colbert.noun_extraction.utils import unidecode_text
from scripts.analysis.check_token_length import draw_histogram, get_token_stats
from scripts.evaluate.utils import load_beir_data
from scripts.utils import BEIR_DATASET_NAMES

DATASET_DIR = "/root/EAGLE/data"

logger = logging.getLogger("PhraseStats")


def count_phrase_stats(
    phrases: List[Tuple[int, int]], explain: bool = True
) -> List[int]:
    """Count the number of phrases with length greater than 1"""
    filtered_phrases = []
    query_with_long_phrases = []
    len_of_long_phrase_per_query = []
    num_of_long_phrases_per_query = []
    for phrase_indices in tqdm.tqdm(phrases, desc="Counting phrases"):
        filtered_items = [j - i for i, j in phrase_indices if j - i > 1]
        num_of_long_phrase = len(filtered_items)
        len_of_long_phrase = (
            sum(filtered_items) / len(filtered_items) if num_of_long_phrase > 0 else 0
        )
        if filtered_items and any([item > 1 for item in filtered_items]):
            query_with_long_phrases.append(phrase_indices)
        if num_of_long_phrase > 0:
            filtered_phrases.append(phrase_indices)
            len_of_long_phrase_per_query.append(len_of_long_phrase)
            num_of_long_phrases_per_query.append(num_of_long_phrase)

    if explain:
        logger.info(
            f"Out of {len(phrases)} data, {len(filtered_phrases)} data have phrases length greater than 1"
        )
        logger.info(
            f"Length of phrases per data (among those with greater than 1): max_length={max(len_of_long_phrase_per_query)}, min_length={min(len_of_long_phrase_per_query)}, avg_length={sum(len_of_long_phrase_per_query)/len(len_of_long_phrase_per_query)}"
        )
        logger.info(
            f"Number of phrases per data (among those with greater than 1): max_num={max(num_of_long_phrases_per_query)}, min_num={min(num_of_long_phrases_per_query)}, avg_num={sum(num_of_long_phrases_per_query)/len(num_of_long_phrases_per_query)}"
        )
        logger.info(f"query_with_long_phrases: {len(query_with_long_phrases)}")
    return None


def main(
    dataset_name: str,
    q_tokenizer,
    d_tokenizer,
    output_dir: str = None,
    draw_historgram: bool = False,
    q_max_len: int = 32,
    d_max_len: int = 300,
) -> None:
    # Load documents
    colletion_path = os.path.join(DATASET_DIR, f"{dataset_name}/collection.tsv")
    logger.info(f"Loading {dataset_name} dataset collection: {colletion_path}...")
    collection = file_utils.read_csv_file(
        colletion_path, delimiter="\t", first_row_as_header=True, quotechar=None
    )
    logger.info(f"Loaded {len(collection)} documents from {colletion_path}")
    # Load data
    logger.info(f"Loading {dataset_name} dataset data...")
    data: List = load_beir_data(
        dataset_dir=DATASET_DIR,
        dataset_name=dataset_name,
        return_unique=True,
        collection=collection,
    )
    logger.info(f"Loaded {len(data)} data from {dataset_name} dataset")
    # Load phrase indices
    d_phrase_file_path = os.path.join(
        DATASET_DIR, f"{dataset_name}/collection.phrase_range.pkl"
    )
    logger.info(
        f"Loading document phrase indices from {dataset_name} dataset: {d_phrase_file_path}..."
    )
    d_phrases = file_utils.read_pickle_file(d_phrase_file_path)
    d_phrases = list(d_phrases.values())
    logger.info(f"Loaded {len(d_phrases)} document phrases from {d_phrase_file_path}")

    # Number of queries
    num_queries = len(set(d[0] for d in data))
    num_of_every_documents = len(collection)
    num_of_gold_documents = len(
        set(list_utils.do_flatten_list([item[2] for item in data]))
    )
    num_of_gold_documents_per_query = sum([len(item[2]) for item in data]) / num_queries

    # Get query token stats
    logger.info(f"Getting query token stats...")
    (
        q_max_text_tokens,
        q_min_text_tokens,
        q_avg_text_tokens,
        q_std,
        q_num_bad_queries,
        q_token_lengths,
    ) = get_token_stats(
        tokenizer=q_tokenizer.tok,
        texts=[item[1] for item in data],
        max_token_threshold=q_max_len,
    )

    # Get document token stats
    logger.info(f"Getting document token stats...")
    (
        d_max_text_tokens,
        d_min_text_tokens,
        d_avg_text_tokens,
        d_std,
        d_num_bad_queries,
        d_token_lengths,
    ) = get_token_stats(
        tokenizer=d_tokenizer.tok,
        texts=[item["text"] for item in collection[:100000]],
        max_token_threshold=d_max_len,
    )

    # Get query phrase stats
    logger.info(f"Getting query phrase stats...")
    queries = [
        unidecode_text(item[1]) for item in tqdm.tqdm(data, desc="Parsing queries")
    ]
    parsed_texts: List[Text] = SpacyModel()(queries, max_token_num=q_max_len)
    input_ids, attention_mask = q_tokenizer.tensorize(queries, bsize=len(queries))[0]
    q_phrases = get_phrase_indices(
        input_ids,
        attention_mask,
        q_tokenizer.tok,
        queries,
        parsed_texts,
        bsize=len(queries),
        all_noun_only=True,
    )[0]
    count_phrase_stats(q_phrases, explain=True)
    # num_of_noun_phrases_per_query = sum([len(q.all_noun_phrase_indices) for q in parsed_texts]) / len(parsed_texts)
    # logger.info(f"Number of noun phrases per query: {num_of_noun_phrases_per_query}")

    # Get document phrase stats
    logger.info(f"Getting document phrase stats...")
    count_phrase_stats(d_phrases, explain=True)

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
        choices=BEIR_DATASET_NAMES + ["all"],
        default="hotpotqa",
    )
    parser.add_argument("--output_dir", type=str, help="Output directory", default=None)
    parser.add_argument(
        "--slack", action="store_true", help="Whether to send slack notification"
    )
    parser.add_argument(
        "--histogram",
        action="store_true",
        help="Whether to draw histogram of token lengths",
    )
    parser.add_argument(
        "--q_max_len", type=int, help="Maximum token length for queries", default=32
    )
    parser.add_argument(
        "--d_max_len", type=int, help="Maximum token length for queries", default=300
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Set logging config
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    args = parse_args()

    # Initialize tokenizer
    # Load tokenizer
    config = ColBERTConfig()
    config.query_maxlen = args.q_max_len
    config.doc_maxlen = args.d_max_len
    config.checkpoint = "bert-base-uncased"
    q_tokenizer = QueryTokenizer(config)
    d_tokenizer = DocTokenizer(config)

    with slack_utils.notification(
        channel="question-answering",
        success_msg=f"Succeeded to compute dataset stats for {args.dataset} dataset",
        error_msg=f"Falied to compute dataset stats for {args.dataset} dataset",
        disable=not args.slack,
    ):
        main(
            dataset_name=args.dataset,
            q_tokenizer=q_tokenizer,
            d_tokenizer=d_tokenizer,
            output_dir=args.output_dir,
            q_max_len=args.q_max_len,
            d_max_len=args.d_max_len,
            draw_historgram=args.histogram,
        )
    logger.info(f"Done!")
