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

logger = logging.getLogger("ValidateCorpusIndex")


def validate_corpus_index(dir_path: str) -> bool:
    # Read in the corpus file
    corpus_file_path: str = os.path.join(dir_path, "corpus.jsonl")
    corpus_data: List[Dict] = file_utils.read_jsonl_file(corpus_file_path)
    # Check if the indices are contiguous
    for idx, data in enumerate(tqdm.tqdm(corpus_data, desc="Idx"), start=1):
        if data["_id"] != idx:
            return False
    return True


@hydra.main(version_base=None, config_path="/root/EAGLE/config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Get the directory path for indices
    base_path = cfg.dataset.dir_path
    # Get all dataset names
    bad_datasets: List[str] = []
    for dataset_name in tqdm.tqdm(DATASET_NAMES, desc="Datasets"):
        logger.info(f"Validating the corpus: {dataset_name} ...")
        dir_path: str = os.path.join(base_path, dataset_name)
        is_good = validate_corpus_index(dir_path=dir_path)
        if not is_good:
            bad_datasets.append(dataset_name)
    logger.info(f"Bad datasets: {bad_datasets}")
    logger.info(f"Done!")
    return None


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    main()
