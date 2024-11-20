import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import logging
import os
from typing import *

import hydra
import lightning as L
import torch
from omegaconf import DictConfig

from eagle.dataset.pl_module.contrastive_module import ContrastiveDataModule
from eagle.dataset.pl_module.inference_module import InferenceDataModule
from eagle.model import LightningNewModel
from eagle.utils import add_global_configs, set_random_seed
from scripts.utils import check_argument

logger = logging.getLogger("Evaluate")


def full_retrieval(
    cfg: DictConfig,
    ckpt_path: str,
    is_analyze: bool,
    use_oracle_candidate: bool = False,
) -> None:
    # Load data module and model
    data_module = InferenceDataModule(cfg)

    # Load index
    index_dir_path = os.path.join(
        cfg.indexing.dir_path, cfg.dataset.name, cfg._global.tag
    )
    logger.info(f"Index directory path: {index_dir_path}")

    # Load trained model
    assert ckpt_path, "Please provide the path to the checkpoint"
    model = LightningNewModel(
        cfg=cfg,
        index_dir_path=index_dir_path,
        use_oracle_candidate=use_oracle_candidate,
    )

    trainer = L.Trainer(
        deterministic=True,
        accelerator="cuda",
        devices=torch.cuda.device_count(),
        strategy="ddp",
    )
    trainer.test(model, datamodule=data_module, ckpt_path=ckpt_path)
    return None


def reranking(cfg: DictConfig, ckpt_path: str, is_analyze: bool) -> None:
    # Load data module and model
    data_module = ContrastiveDataModule(cfg, skip_train=True)

    # Load trained model
    assert ckpt_path, "Please provide the path to the checkpoint"
    model = LightningNewModel(cfg=cfg)
    trainer = L.Trainer(
        deterministic=True,
        accelerator="cuda",
        devices=torch.cuda.device_count(),
        strategy="ddp",
    )
    trainer.test(model, datamodule=data_module, ckpt_path=ckpt_path)
    return None


def check_arguments(cfg: DictConfig) -> DictConfig:
    check_argument(
        cfg.args,
        name="mode",
        arg_type=str,
        choices=["reranking", "retrieval"],
        is_requried=True,
        help="mode should be 'reranking', or 'retrieval'",
    )
    check_argument(
        cfg.args,
        name="use_slack",
        arg_type=bool,
        help="Whether to use slack notification",
    )
    check_argument(
        cfg.args,
        name="use_oracle_candidate",
        arg_type=bool,
        help="Wheter to include oracle in candidate list when doing retrieval",
    )
    check_argument(
        cfg.args, name="ckpt_path", arg_type=str, help="Path to the checkpoint"
    )
    return cfg.args


@hydra.main(version_base=None, config_path="/root/EAGLE/config", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg: DictConfig = add_global_configs(cfg, exclude_keys=["args"])

    # Set random seeds
    L.seed_everything(cfg._global.seed, workers=True)
    args = check_arguments(cfg)

    with torch.no_grad():
        # Check arguments
        if args.mode == "retrieval":
            full_retrieval(
                cfg,
                ckpt_path=args.ckpt_path,
                is_analyze=False,
                use_oracle_candidate=args.use_oracle_candidate,
            )
        elif args.mode == "reranking":
            reranking(cfg, ckpt_path=args.ckpt_path, is_analyze=False)
        else:
            raise ValueError(f"Invalid mode: {args.mode}")

        logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    set_random_seed()
    main()
