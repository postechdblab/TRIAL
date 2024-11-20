import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import json
import logging
import os
from typing import *

import hkkang_utils.file as file_utils
import hydra
from omegaconf import DictConfig

from scripts.analysis.utils import DATASET_NAMES

logger = logging.getLogger("IndexStats")


def get_stat_for_index(path: str) -> Dict[str, Union[int, float]]:
    # Get the path for the plan file
    plan_file = os.path.join(path, "plan.json")
    if not os.path.exists(plan_file):
        logger.warning(f"File {plan_file} does not exist.")
        return {}

    # Read the plan file and get the number of clusters
    plan_info: Dict[str, Any] = file_utils.read_json_file(plan_file)
    num_clusters = plan_info["num_partitions"]

    # Get all files ending with .metadata.json
    num_docs = 0
    num_embeddings = 0
    metadata_files = file_utils.get_files_in_directory(
        path, filter_func=lambda x: x.endswith(".metadata.json")
    )
    for metadata_file in metadata_files:
        metadata_path = os.path.join(path, metadata_file)
        metadata_info = file_utils.read_json_file(metadata_path)
        num_docs += metadata_info["num_passages"]
        num_embeddings += metadata_info["num_tok_embeddings"]
    avg_emb_in_cluster = num_embeddings // num_clusters if num_clusters > 0 else 0

    # Get the directory memory size
    dir_size = file_utils.get_directory_size(path)

    # Return
    return {
        "num_clusters": num_clusters,
        "num_docs": num_docs,
        "num_embeddings": num_embeddings,
        "avg_emb_in_cluster": avg_emb_in_cluster,
        "memory_size": dir_size,
    }


def get_stat_for_dataset(
    base_path: str, dataset_name: str
) -> Dict[str, Dict[str, Union[int, float]]]:
    # Get the directory path for indices
    dir_path = os.path.join(base_path, dataset_name)

    # Check if directory exists
    if not os.path.exists(dir_path):
        logger.warning(f"Directory {dir_path} does not exist.")
        return {}

    # Get stats for each subdirectory
    all_stats: Dict[str, Dict[str, Union[int, float]]] = {}
    for model_name in os.listdir(dir_path):
        stats: Dict[str, Any] = get_stat_for_index(
            path=os.path.join(dir_path, model_name)
        )
        all_stats[f"{dataset_name}_{model_name}"] = stats

    return all_stats


def get_stats_for_all_dataset(
    cfg: DictConfig,
) -> Dict[str, Dict[str, Union[int, float]]]:
    # Get the directory path for indices
    base_path = cfg.indexing.dir_path
    # Get all dataset names
    all_stats: Dict[str, Dict[str, Union[int, float]]] = {}
    for dataset_name in DATASET_NAMES:
        stats = get_stat_for_dataset(base_path=base_path, dataset_name=dataset_name)
        all_stats.update(stats)
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
