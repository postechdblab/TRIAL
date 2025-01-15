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


def fix_dev_index(dir_path: str) -> bool:
    """
    1.Fix the corpus index in the answers dic
    """
    # Read in the dev file
    dev_file_path: str = os.path.join(dir_path, "dev.jsonl")
    dev_data: List[Dict] = file_utils.read_jsonl_file(dev_file_path)
    # Fix the corpus index in the answers dict
    for data in tqdm.tqdm(dev_data, desc="Idx"):
        data["answers"] = [item + 1 for item in data["answers"]]
    # Save the new data
    file_utils.write_jsonl_file(dev_data, dev_file_path)
    return None


def fix_corpus_index(dir_path: str, do_add_offset: bool = True) -> bool:
    """
    1. Add 1 offset to the corpus index
    2. Add dummy data for the missing indices
    """
    # Read in the corpus file
    corpus_file_path: str = os.path.join(dir_path, "corpus.jsonl")
    corpus_data: List[Dict] = file_utils.read_jsonl_file(corpus_file_path)
    # Configs
    is_ok = False if do_add_offset else True
    add_offset = 1 if do_add_offset else 0
    # Fix the corpus index
    start_cnt = 1
    cnt = 0
    new_data: List[Dict] = []
    for data in tqdm.tqdm(corpus_data, desc="Idx"):
        if int(data["_id"]) != cnt + start_cnt:
            is_ok = False
            # Append dummy data
            for j in range(cnt, int(data["_id"])):
                new_data.append({"_id": j + 1, "text": [""], "title": ""})
                cnt += 1
        data["_id"] = cnt + 1
        new_data.append(data)
        cnt += 1
    # Save the new data
    if not is_ok:
        logger.info(f"Fixing the corpus index for {dir_path} ...")
        file_utils.write_jsonl_file(new_data, corpus_file_path)
    return is_ok


@hydra.main(version_base=None, config_path="/root/EAGLE/config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Get the directory path for indices
    base_path = cfg.dataset.dir_path
    # Get all dataset names
    bad_datasets: List[str] = []
    for dataset_name in tqdm.tqdm(DATASET_NAMES, desc="Datasets"):
        logger.info(f"Validating the corpus: {dataset_name} ...")
        dir_path: str = os.path.join(base_path, dataset_name)
        # fix_dev_index(dir_path=dir_path)
        is_good = fix_corpus_index(dir_path=dir_path)
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
