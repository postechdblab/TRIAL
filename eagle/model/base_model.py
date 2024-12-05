import abc
import logging
from collections import defaultdict
from typing import *

import torch
from omegaconf import DictConfig
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.nodes import AnyNode
from transformers import AutoModel

from eagle.tokenization.tokenizers import Tokenizers
from eagle.utils import add_config

# Add DictConfig to the safe globals for torch serialization
torch.serialization.add_safe_globals(
    [DictConfig, ContainerMetadata, Any, dict, defaultdict, AnyNode, Metadata]
)

logger = logging.getLogger("BaseModel")


class BaseModel(torch.nn.Module):
    def __init__(self, cfg: DictConfig, tokenizers: Tokenizers) -> None:
        super().__init__()
        self.cfg = cfg
        self.tokenizers = tokenizers
        backbone_name = cfg.override_backbone_name
        if backbone_name is None:
            backbone_name = cfg.backbone_name
        self.llm = self.__create_backbone_model(
            backbone_name, vocab_num=tokenizers.vocab_num
        )

    def __create_backbone_model(self, name: str, vocab_num: int) -> torch.nn.Module:
        # Load pretrained backbone model
        model = AutoModel.from_pretrained(
            name,
            device_map=torch.device("cpu"),
        )

        # Resize the token embeddings
        model.resize_token_embeddings(vocab_num)

        # Remove redundant layers
        if "bert-" in name:
            model.pooler = None
        if "t5" in name:
            model.decoder = None
            model = model.encoder

        return model

    def load_checkpoint(self) -> None:
        if self.cfg.ckpt_path:
            logger.info(f"Loading model checkpoint from {self.cfg.ckpt_path}")
            loaded_params = torch.load(
                self.cfg.ckpt_path, map_location="cpu", weights_only=True
            )["state_dict"]
            # Remove "model." from the keys
            renamed_params = {}
            prefix_to_remove = "model."
            prefix_to_remove2 = "model._orig_mod."
            for k, v in loaded_params.items():
                # Remove the prefix
                if k.startswith(prefix_to_remove2):
                    k = k[len(prefix_to_remove2) :]
                elif k.startswith(prefix_to_remove):
                    k = k[len(prefix_to_remove) :]
                else:
                    raise ValueError(f"Cannot find {prefix_to_remove} in {k}")
                # Save the renamed params
                renamed_params[k] = v
            # Replace the parameters
            found_params = []
            for name, param in self.named_parameters():
                # Check the name exists in the loaded params
                if name not in renamed_params:
                    logger.warning(
                        f"Cannot find {name} in the loaded params. Skipping.."
                    )
                    continue
                assert (
                    name in renamed_params
                ), f"Cannot find {name} in the loaded params"
                # Check the dtype, shape, and device
                assert (
                    param.dtype == renamed_params[name].dtype
                ), f"Type mismatch: {name} {param.dtype} vs {renamed_params[name].dtype}"
                assert (
                    param.shape == renamed_params[name].shape
                ), f"Shape mismatch: {name} {param.shape} vs {renamed_params[name].shape}"
                assert (
                    param.device == renamed_params[name].device
                ), f"Device mismatch: {name} {param.device} vs {renamed_params[name].device}"
                param.data = renamed_params[name]
                found_params.append(name)
            not_found_params = set(renamed_params.keys()) - set(found_params)
            assert (
                len(not_found_params) == 0
            ), f"Cannot find {not_found_params} in the model"
            logger.info(
                f"Updated {len(found_params)} parameters instances from the checkpoint"
            )
            add_config(self.cfg, key="ckpt_path", value=None)
        return None

    @abc.abstractmethod
    def encode_passage(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Implement this method in the derived class")

    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Implement this method in the derived class")

    @abc.abstractmethod
    def encode_documents(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Implement this method in the derived class")

    def eval(self, *args, **kwargs) -> None:
        for att_name in dir(self):
            att = getattr(self, att_name)
            if (
                isinstance(att, torch.nn.Module)
                or isinstance(att, torch.nn.ModuleList)
                or isinstance(att, torch.nn.ModuleDict)
                or isinstance(att, torch.nn.ParameterList)
                or isinstance(att, torch.nn.ParameterDict)
                or isinstance(att, torch.nn.Parameter)
                or isinstance(att, torch.Tensor)
            ):
                att.eval(*args, **kwargs)

    def train(self, *args, **kwargs) -> None:
        for att_name in dir(self):
            att = getattr(self, att_name)
            if (
                isinstance(att, torch.nn.Module)
                or isinstance(att, torch.nn.ModuleList)
                or isinstance(att, torch.nn.ModuleDict)
                or isinstance(att, torch.nn.ParameterList)
                or isinstance(att, torch.nn.ParameterDict)
                or isinstance(att, torch.nn.Parameter)
                or isinstance(att, torch.Tensor)
            ):
                att.train(*args, **kwargs)
