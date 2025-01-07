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

logger = logging.getLogger("Convert_lotte_corpus_index")


def convert_corpus_index(dir_path: str) -> None:
    # Read in corpus file
    logger.info(f"Reading in corpus file: {dir_path} ...")
    corpus_file_path: str = os.path.join(dir_path, "corpus.jsonl")
    corpus_data: List[Dict] = file_utils.read_jsonl_file(corpus_file_path)
    # Conver the indices: 0-based to 1-based
    for data in corpus_data:
        data["_id"] = str(int(data["_id"]) + 1)
    # Write out the corpus file
    logger.info(f"Writing out the corpus file: {corpus_file_path} ...")
    file_utils.write_jsonl_file(corpus_data, corpus_file_path)

    # Read in dev file
    logger.info(f"Reading in dev file: {dir_path} ...")
    dev_file_path: str = os.path.join(dir_path, "dev.jsonl")
    dev_data: List[Dict] = file_utils.read_jsonl_file(dev_file_path)
    # Conver the indices: 0-based to 1-based
    for data in dev_data:
        data["answers"] = [item + 1 for item in data["answers"]]
    # Write out the dev file
    logger.info(f"Writing out the dev file: {dev_file_path} ...")
    file_utils.write_jsonl_file(dev_data, dev_file_path)
    return None


@hydra.main(version_base=None, config_path="/root/EAGLE/config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Get the directory path for indices
    base_path = cfg.dataset.dir_path
    # Get all dataset names
    for dataset_name in tqdm.tqdm(DATASET_NAMES, desc="Datasets"):
        if dataset_name.lower().startswith("lotte"):
            logger.info(f"Cleaning the corpus: {dataset_name} ...")
            dir_path: str = os.path.join(base_path, dataset_name)
            convert_corpus_index(dir_path=dir_path)
    logger.info(f"Done!")
    return None


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    main()
