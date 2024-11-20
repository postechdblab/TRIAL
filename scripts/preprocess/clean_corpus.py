import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import logging
import os
from typing import *

import hkkang_utils.file as file_utils
import hydra
import tqdm
from omegaconf import DictConfig

from scripts.analysis.utils import DATASET_NAMES

logger = logging.getLogger("CleanCorpus")


def clean_dataset(dir_path: str) -> None:
    """
    1. Cleaning logic: if the text is empty, remove the item from the corpus
    """

    # Get file paths
    src_file_path = os.path.join(dir_path, "corpus.jsonl.original")
    backup_file_path = os.path.join(dir_path, "corpus.jsonl.original")
    dst_file_path = os.path.join(dir_path, "corpus.jsonl")

    # Change the src_file_path if the original file does not exist
    has_backup_file = os.path.exists(src_file_path)
    if not has_backup_file:
        src_file_path = os.path.join(dir_path, "corpus.jsonl")

    # Read in the corpus
    logger.info(f"Reading the corpus from {src_file_path}")
    corpus_data: List[Dict] = file_utils.read_jsonl_file(src_file_path)
    logger.info(f"Read {len(corpus_data)} items from the corpus")

    # Clean the corpus
    filtered_corpus_data: List[Dict] = []
    for item in tqdm.tqdm(corpus_data, desc="Filtering"):
        if len(item["text"]) > 0:
            filtered_corpus_data.append(item)

    # Return if no items to remove
    if len(filtered_corpus_data) == len(corpus_data):
        logger.info(f"No items to remove.")
        return

    logger.info(
        f"Removed {len(corpus_data) - len(filtered_corpus_data)} items from the corpus"
    )

    # Backup the original file
    if not has_backup_file:
        logger.info(
            f"Backing up the original file {src_file_path} to {backup_file_path}"
        )
        file_utils.write_jsonl_file(
            corpus_data, os.path.join(dir_path, "corpus.jsonl.original")
        )
        logger.info(
            f"Backed up the original file {src_file_path} to {backup_file_path}"
        )

    # Write the original file
    logger.info(
        f"Writing the cleaned corpus ({len(filtered_corpus_data)} data) to {dst_file_path}"
    )
    file_utils.write_jsonl_file(filtered_corpus_data, dst_file_path)
    return None


@hydra.main(version_base=None, config_path="/root/EAGLE/config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Get the directory path for indices
    base_path = cfg.dataset.dir_path
    # Get all dataset names
    for dataset_name in tqdm.tqdm(DATASET_NAMES, desc="Datasets"):
        logger.info(f"Cleaning the corpus: {dataset_name} ...")
        dir_path: str = os.path.join(base_path, dataset_name)
        clean_dataset(dir_path=dir_path)
    logger.info(f"Done!")
    return None


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    main()
