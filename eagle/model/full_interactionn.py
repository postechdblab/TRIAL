from typing import *

import torch
from omegaconf import DictConfig
from transformers import AutoModel

from eagle.tokenizer import Tokenizers


class CrossEncoder(torch.nn.Module):
    def __init__(self, cfg: DictConfig, tokenizers: Tokenizers) -> None:
        super().__init__()
        self.cfg = cfg
        self.q_special_tok_ids = tokenizers.q_special_tok_ids
        self.d_special_tok_ids = tokenizers.d_special_tok_ids
        self.punct_tok_ids = tokenizers.punct_tok_ids
        self.q_maxlen = tokenizers.q_maxlen
        # Configs
        self.nway = cfg.nway
        # Backbone model
        self.llm = self.__create_backbone_model(
            cfg.llm.name, vocab_num=tokenizers.vocab_num
        )
        # Projection layer
        self.tok_projection_layer = torch.nn.Linear(
            self.llm.config.hidden_size, cfg.out_dim, bias=False
        )

    def forward(
        self,
        q_tok_ids: torch.Tensor,
        q_tok_att_mask: torch.Tensor,
        q_scatter_indices: torch.Tensor,
        q_tok_mask: torch.Tensor,
        q_phrase_mask: torch.Tensor,
        doc_tok_ids: torch.Tensor,
        doc_tok_att_mask: torch.Tensor,
        doc_scatter_indices: torch.Tensor,
        doc_tok_mask: torch.Tensor,
        doc_phrase_mask: torch.Tensor,
        fine_grained_label: Optional[torch.Tensor] = None,
        fine_grained_label_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        distillation_scores: Optional[torch.Tensor] = None,
        is_inference: Optional[bool] = False,
        is_analyze: Optional[bool] = False,
        **kwargs,
    ) -> Dict[str, Any]:
        # Configs
        bsize, nway, dim = doc_tok_ids.shape
        ib_nhard = nway // bsize
        is_eval = labels is not None
        is_use_fine_grained_loss = fine_grained_label is not None
        assert (
            q_tok_ids.size(0) == bsize
        ), f"Batch size mismatch: {q_tok_ids.size(0)} != {bsize}"
        # Encode
        self.compute_scores()

    @property
    def q_skiplist(self) -> List[int]:
        return self.q_special_tok_ids

    @property
    def d_skiplist(self) -> List[int]:
        return self.d_special_tok_ids
        # return self.d_special_tok_ids + self.punct_tok_ids

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

    def compute_scores(
        self,
    ) -> torch.Tensor:
        pass
