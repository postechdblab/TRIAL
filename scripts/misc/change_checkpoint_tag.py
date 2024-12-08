from typing import *

import torch
from omegaconf import DictConfig
import logging

logger = logging.getLogger("TagConverter")


def check_and_replace_tag(ckpt: Dict, orig_tag: str, new_tag: str) -> Dict:
    for key, value in ckpt.items():
        if isinstance(value, (dict, DictConfig)):
            ckpt[key] = check_and_replace_tag(value, orig_tag, new_tag)
        if key == "tag":
            assert value == orig_tag, f"Tag mismatch: {value} != {orig_tag}"
            ckpt[key] = new_tag
    return ckpt


def main():
    # Configs
    ckpt_path = "/root/EAGLE/runs/tmp/eagle_relation_lambda_1_distill_from_author/best_model.ckpt"
    orig_tag = "eagle_weights_q_relation_distill_from_author"
    new_tag = "eagle_relation_lambda_1_distill_from_author"

    # Load the checkpoint
    print(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, weights_only=False)
    # Replace the tag name from orig_tag to new_tag
    ckpt = check_and_replace_tag(ckpt["hyper_parameters"], orig_tag, new_tag)
    print(f"Replaced tag from {orig_tag} to {new_tag}")
    # save the cleaned checkpoint
    torch.save(ckpt, ckpt_path)
    print(f"Converted checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
