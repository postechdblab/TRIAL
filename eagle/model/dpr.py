import logging
from typing import *

import torch
from omegaconf import DictConfig

from eagle.model.base_model import BaseModel
from eagle.model.objective import compute_loss
from eagle.tokenization.tokenizer import Tokenizer

logger = logging.getLogger("DPR")


class DPR(BaseModel):
    def __init__(self, cfg: DictConfig, tokenizers: Tokenizer) -> None:
        super().__init__(cfg=cfg, tokenizers=tokenizers)
        self.score_projection_layer = torch.nn.Linear(self.llm.config.hidden_size, 1)

        # TODO: Move the post processing to the base class
        self.load_checkpoint()

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
        bsize = len(kwargs["q_tok_ids"])
        nway = tok_ids.shape[0] // bsize
        _, dim = tok_ids.shape
        ib_nhard = nway // bsize
        is_eval = labels is not None

        # Encode
        intra_pred_scores = self.compute_scores(tok_ids, tok_att_mask)
        ib_inter_pred_scores = self.compute_scores(ib_tok_ids, ib_tok_att_mask)
        device = intra_pred_scores.device

        # Compute loss
        loss, intra_loss, inter_loss, kl_loss = compute_loss(
            scores=intra_pred_scores,
            ib_scores=ib_inter_pred_scores,
            distillation_scores=distillation_scores,
            bsize=bsize,
            nway=nway,
            ib_nhard=ib_nhard,
            device=device,
        )
        return_dict = {
            "loss": loss,
            "intra_loss": intra_loss,
            "inter_loss": inter_loss,
            "kl_loss": kl_loss,
        }
        if is_eval:
            return return_dict, intra_pred_scores.reshape(bsize, -1)

        return return_dict

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
