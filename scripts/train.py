import logging
import os
from datetime import timedelta
from typing import *

import git
import hkkang_utils.slack as slack_utils
import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from pytorch_lightning.strategies import DDPStrategy

from eagle.dataset import NewDataModule
from eagle.model import LightningNewModel
from eagle.utils import add_config, add_global_configs

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision("high")

logger = logging.getLogger("PL_Trainer")


@hydra.main(version_base=None, config_path="/root/EAGLE/config", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg: DictConfig = add_global_configs(cfg)
    # Get git hash
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    add_config(cfg, "git_hash", sha)

    # Set random seeds
    seed_everything(cfg._global.seed, workers=True)

    # Calculate configs
    device_cnt = torch.cuda.device_count()
    log_every_n_steps = (
        cfg.training.logging_steps * cfg.training.gradient_accumulation_steps
    )
    val_check_interval = (
        10
        if cfg._global.is_debug
        else cfg.training.val_check_interval_by_step
        * cfg.training.gradient_accumulation_steps
    )
    default_root_dir = (
        os.path.join(cfg._global.root_dir, "debug", cfg._global.tag)
        if cfg._global.is_debug
        else os.path.join(cfg._global.root_dir, cfg._global.tag)
    )
    train_batch_num = 532752 / cfg.training.per_device_train_batch_size / device_cnt

    # Load data module and model
    data_module = NewDataModule(cfg)
    model = LightningNewModel(cfg=cfg, train_batch_num=train_batch_num)

    # Trainer initialization with your training args
    trainer = pl.Trainer(
        deterministic=True,
        max_epochs=1,
        num_sanity_val_steps=2,
        profiler="simple",
        accelerator="gpu",
        devices=device_cnt,
        precision=cfg.training.precision,
        log_every_n_steps=log_every_n_steps,
        val_check_interval=val_check_interval,
        default_root_dir=default_root_dir,
        strategy=DDPStrategy(
            timeout=timedelta(hours=5), static_graph=True, gradient_as_bucket_view=True
        ),
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            ModelSummary(max_depth=-1),
            ModelCheckpoint(
                dirpath=default_root_dir,
                monitor="val_NDCG@10",
                mode="max",
                save_top_k=1,
                save_last=True,
            ),
        ],
    )
    if cfg.training.resume_ckpt_path:
        logger.info(f"Resuming from checkpoint: {cfg.training.resume_ckpt_path}")

    trainer.fit(model, datamodule=data_module, ckpt_path=cfg.training.resume_ckpt_path)


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    with slack_utils.notification(
        channel="question-answering",
        success_msg=f"Succeeded to train NewRetriever!",
        error_msg=f"Falied to train NewRetriever!",
    ):
        main()
    logger.info("Training completed successfully!")
