import logging
import os
import warnings
from datetime import timedelta
from typing import *

import hkkang_utils.slack as slack_utils
import hydra
import lightning as L
import torch
import torch.multiprocessing as mp
from omegaconf import DictConfig

from eagle.index.registry import INDEXER_REGISTRY, BaseIndexer
from eagle.utils import add_global_configs, set_random_seed

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger("Indexing")


def multi_process_indexing(
    rank: int, cfg: DictConfig, world_size: int, indexer_module: BaseIndexer
) -> None:
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )

    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)

        # Initialize the process group
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        torch.distributed.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method="env://",
            world_size=world_size,
            timeout=timedelta(hours=5),
        )

        # Set default device
        torch.cuda.set_device(rank)

        # Create and call indexer
        indexer = indexer_module(cfg=cfg, rank=rank, world_size=world_size)
        indexer()

        return None


@hydra.main(version_base=None, config_path="/root/EAGLE/config", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg: DictConfig = add_global_configs(cfg, exclude_keys=["args"])

    with slack_utils.notification(
        channel="question-answering",
        success_msg=f"Succeeded to train NewRetriever!",
        error_msg=f"Falied to train NewRetriever!",
        disable=not ("use_slack" in cfg and cfg.use_slack),
    ):
        # Set random seeds
        set_random_seed(seed=cfg._global.seed)

        # Set random seeds
        L.seed_everything(cfg._global.seed, workers=True)

        # Get configs
        world_size = torch.cuda.device_count()

        # Get indexer module
        indexer_module = INDEXER_REGISTRY[cfg.model.name]

        mp.spawn(
            multi_process_indexing,
            args=(cfg, world_size, indexer_module),
            nprocs=world_size,
            join=True,
        )
        print("Done!")
        logger.info("Training completed successfully!")


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    main()
