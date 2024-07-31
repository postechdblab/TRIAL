from typing import *

import torch
from omegaconf import DictConfig
from transformers import AutoModel

from eagle.tokenizer import Tokenizer
from eagle.model.objective import compute_loss


class CrossEncoder(torch.nn.Module):
    def __init__(self, cfg: DictConfig, tokenizers: Tokenizer) -> None:
        super().__init__()
        self.cfg = cfg
        # Configs
        self.nway = cfg.nway
        # Backbone model
        self.llm = self.__create_backbone_model(
            cfg.backbone_name, vocab_num=tokenizers.vocab_num
        )
        self.score_projection_layer = torch.nn.Linear(self.llm.config.hidden_size, 1)

    def forward(
        self,
        tok_ids: torch.Tensor,
        tok_att_mask: torch.Tensor,
        ib_tok_ids: torch.Tensor,
        ib_tok_att_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        distillation_scores: Optional[torch.Tensor] = None,
        is_inference: Optional[bool] = False,
        is_analyze: Optional[bool] = False,
        **kwargs,
    ) -> Dict[str, Any]:
        # Configs
        bsize, nway, dim = tok_ids.shape
        ib_nhard = nway // bsize
        is_eval = labels is not None

        # Encode
        pred_scores = self.compute_scores(tok_ids, tok_att_mask)
        device = pred_scores.device

        # Compute loss
        loss, intra_loss, inter_loss, kl_loss = compute_loss(
            scores=intra_scores,
            ib_scores=inter_scores,
            distillation_scores=distillation_scores,
            bsize=bsize,
            nway=nway,
            ib_nhard=ib_nhard,
            device=device,
            intra_loss_coeff=self.intra_loss_coeff,
            inter_loss_coeff=self.inter_loss_coeff,
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
        self, tok_ids: torch.Tensor, tok_att_mask: torch.Tensor
    ) -> torch.Tensor:
        # Compuate intra scores

        # Compute inter scores

        # Encode
        encoded_tok_vectors = self.llm(
            input_ids=tok_ids,
            attention_mask=tok_att_mask,
        ).last_hidden_state

        # Project to scores
        scores = self.score_projection_layer(encoded_tok_vectors[:, 0, :])

        return scores

    def compute_intra_scores(
        self,
        tok_ids: torch.Tensor,
        tok_att_mask: torch.Tensor,
        nway: int,
        ib_nhard: int,
    ) -> torch.Tensor:
        return scores

    def compute_inter_scores(
        self,
        tok_ids: torch.Tensor,
        tok_att_mask: torch.Tensor,
        nway: int,
        ib_nhard: int,
    ) -> torch.Tensor:
        return scores
