import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import json
import logging
import os
from typing import *

import hkkang_utils.file as file_utils
import hydra
import tqdm
from omegaconf import DictConfig

from eagle.tokenization.tokenizer import Tokenizer
from eagle.tokenization.tokenizers import Tokenizers
from scripts.analysis.utils import DATASET_NAMES, avg, avg_list_of_list

logger = logging.getLogger("TokenStats")


def get_stats_on_tokenization(
    tokenizer: Tokenizer, sentence: str
) -> Tuple[int, int, int, int]:
    # Split the sentence into words
    words = sentence.split()

    # Count the number of words that are broken down into multiple tokens
    token_ids_list = tokenizer(words)["input_ids"]

    # Post-process the token ids (Remove the special tokens)
    token_ids_list = [item[2:-1] for item in token_ids_list]

    # Statistics
    tok_num_for_broken_words: List[List[int]] = [
        len(item) for item in token_ids_list if len(item) > 1
    ]
    num_brokn_words = len(tok_num_for_broken_words)
    num_token_per_broken_words = avg(tok_num_for_broken_words)
    num_token_per_words = avg_list_of_list(token_ids_list)
    total_words = len(token_ids_list)

    return num_brokn_words, num_token_per_broken_words, num_token_per_words, total_words


def get_stats_for_dataset(
    tokenizer: Tokenizer, dataset: List[Dict[str, Any]], sample_num: int = 10000
) -> Dict[str, int]:
    # Get the statistics
    num_broken_words_list: List[int] = []
    num_token_per_broken_words_list: List[int] = []
    num_token_per_words_list: List[int] = []
    num_words_list: List[int] = []

    # Process each item in the dataset
    # Initialize tqdm with total set to sample_num
    with tqdm.tqdm(total=min(sample_num, len(dataset)), desc="Data") as pbar:
        cnt = 0
        for datum in dataset:
            # Break the loop if the sample number is reached
            if cnt >= sample_num:
                break

            # Process
            text = datum["text"]
            if isinstance(text, list):
                full_paragraph_text = " ".join(text)
            else:
                full_paragraph_text = text
            if len(full_paragraph_text) == 0:
                continue

            (
                num_brokn_words,
                num_token_per_broken_words,
                num_token_per_words,
                words_in_text,
            ) = get_stats_on_tokenization(tokenizer, full_paragraph_text)
            num_broken_words_list.append(num_brokn_words)
            num_token_per_broken_words_list.append(num_token_per_broken_words)
            num_token_per_words_list.append(num_token_per_words)
            num_words_list.append(words_in_text)

            cnt += 1
            pbar.update(1)  # Update the progress bar only when cnt is incremented

    # Return the stats
    return {
        "avg_num_broken_words_per_doc": avg(num_broken_words_list),
        "max_num_broken_words_per_doc": max(num_broken_words_list),
        "min_num_broken_words_per_doc": min(num_broken_words_list),
        "avg_tok_per_broken_words": avg(num_token_per_broken_words_list),
        "max_tok_per_broken_words": max(num_token_per_broken_words_list),
        "min_tok_per_broken_words": min(num_token_per_broken_words_list),
        "avg_tok_per_words": avg(num_token_per_words_list),
        "max_tok_per_words": max(num_token_per_words_list),
        "min_tok_per_words": min(num_token_per_words_list),
        "avg_num_of_words": avg(num_words_list),
        "max_num_of_words": max(num_words_list),
        "min_num_of_words": min(num_words_list),
        "total_samples": len(num_broken_words_list),
    }


def get_stats_for_all_dataset(
    cfg: DictConfig, dataset_names: List[str]
) -> Dict[str, Dict[str, int]]:
    dataset_query_paths = []
    dataset_corpus_paths = []
    for dataset_name in dataset_names:
        dir_path = os.path.join(cfg.dataset.dir_path, dataset_name)
        dataset_query_paths.append(os.path.join(dir_path, cfg.dataset.query_file))
        dataset_corpus_paths.append(os.path.join(dir_path, cfg.dataset.corpus_file))

    # Prepare tokenizers
    tokenizers = Tokenizers(
        q_cfg=cfg.tokenizers.query,
        d_cfg=cfg.tokenizers.document,
        model_name=cfg.model.backbone_name,
    )

    # Get the statistics for all datasets
    stats: Dict[str, Dict[str, int]] = {}
    for dataset_name, query_path, corpus_path in tqdm.tqdm(
        zip(dataset_names, dataset_query_paths, dataset_corpus_paths),
        desc="Dataset",
        total=len(dataset_names),
    ):
        logger.info(f"Loading the {dataset_name} query dataset from {query_path} ...")
        query_dataset: List = file_utils.read_json_file(
            query_path, auto_detect_extension=True
        )
        logger.info(
            f"Loading the {dataset_name} document dataset from {corpus_path} ..."
        )
        corpus_dataset: List = file_utils.read_json_file(
            corpus_path, auto_detect_extension=True
        )
        logger.info(f"Getting the token statistics for {dataset_name} query dataset")
        stats[f"{dataset_name}_query"] = get_stats_for_dataset(
            tokenizers.q_tokenizer, query_dataset
        )
        logger.info(f"Getting the token statistics for {dataset_name} document dataset")
        stats[f"{dataset_name}_doc"] = get_stats_for_dataset(
            tokenizers.d_tokenizer, corpus_dataset
        )

    return stats


@hydra.main(version_base=None, config_path="/root/EAGLE/config", config_name="config")
def main(cfg: DictConfig) -> None:
    all_stats = get_stats_for_all_dataset(cfg=cfg, dataset_names=DATASET_NAMES)

    # Get the dataset statistics
    logger.info(json.dumps(all_stats, indent=4))

    return None


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    main()
