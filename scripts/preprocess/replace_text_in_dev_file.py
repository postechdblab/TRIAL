import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import logging
import os
from typing import *

import hkkang_utils.file as file_utils
import hydra
import tqdm
from omegaconf import DictConfig

from scripts.utils import BEIR_DATASET_NAMES

logger = logging.getLogger("ReplaceDevFile")


def replace_text_in_dev_file(dir_path: str) -> None:
    q_file_path = os.path.join(dir_path, "queries.jsonl")
    dev_file_path = os.path.join(dir_path, "dev.jsonl")
    # Read in the queries data
    logger.info(f"Reading queries from {q_file_path}")
    q_data: List = file_utils.read_jsonl_file(q_file_path)
    q_data: Dict[str, str] = {item["_id"]: item["text"] for item in q_data}
    logger.info(f"Reading dev data from {dev_file_path}")
    dev_data = file_utils.read_jsonl_file(dev_file_path)
    for item in tqdm.tqdm(dev_data, desc="Data"):
        # Replace the query in the dev data
        _id = item["id"]
        item["text"] = q_data[_id]
    # Write the dev data back to the file
    logger.info(f"Writing dev data to {dev_file_path}")
    file_utils.write_jsonl_file(dev_data, dev_file_path)
    return None


@hydra.main(version_base=None, config_path="/root/EAGLE/config", config_name="config")
def main(cfg: DictConfig) -> None:
    for dataset_name in tqdm.tqdm(BEIR_DATASET_NAMES, desc="Dataset"):
        if dataset_name == "msmarco":
            continue
        dir_path = os.path.join(cfg.dataset.dir_path, f"beir-{dataset_name}")
        replace_text_in_dev_file(dir_path)
    return None


if __name__ == "__main__":
    main()
