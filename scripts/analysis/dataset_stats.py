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

from scripts.analysis.utils import DATASET_NAMES

logger = logging.getLogger("DatasetStats")


def get_stats_for_dataset(
    base_path: str, dataset_name: str
) -> Dict[str, Union[int, float]]:
    dir_path = os.path.join(base_path, dataset_name)

    if not os.path.exists(dir_path):
        logger.warning(f"Directory {dir_path} does not exist.")
        return {}

    # Get stats for each subdirectory
    logger.info(f"Reading in {dataset_name}...")
    docs = file_utils.read_jsonl_file(os.path.join(dir_path, "corpus.jsonl"))
    num_docs = len(docs)

    return {"doc_count": num_docs}


def get_stats_for_all_dataset(
    cfg: DictConfig,
) -> Dict[str, Dict[str, Union[int, float]]]:
    base_path = cfg.dataset.dir_path
    all_stats: Dict[str, Dict[str, Union[int, float]]] = {}
    for dataset_name in tqdm.tqdm(DATASET_NAMES):
        stats = get_stats_for_dataset(base_path=base_path, dataset_name=dataset_name)
        all_stats[dataset_name] = stats
    return all_stats


@hydra.main(version_base=None, config_path="/root/EAGLE/config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    1. For each dataset:
    2. Show the number of cluster, embeddings in each cluster, and total number of embeddings.
    3. Show the size of the embedding and total size of the index.
    """
    all_stats = get_stats_for_all_dataset(cfg=cfg)
    logger.info(json.dumps(all_stats, indent=4))
    return None


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    main()
