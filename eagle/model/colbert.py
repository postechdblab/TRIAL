import logging
from typing import *

import torch
from omegaconf import DictConfig

from eagle.model.base_model import BaseModel
from eagle.model.objective import (
    compute_loss,
    doc_indices_for_ib_loss,
    get_target_scale_tensor,
)
from eagle.search.algorithm import compute_sum_maxsim
from eagle.tokenization import Tokenizer

logger = logging.getLogger("ColBERT")


class ColBERT(BaseModel):
    def __init__(self, cfg: DictConfig, tokenizers: Tokenizer) -> None:
        super().__init__(cfg=cfg, tokenizers=tokenizers)
        # Configs
        self.q_special_tok_ids = tokenizers.q_tokenizer.special_toks_ids
        self.d_special_tok_ids = tokenizers.d_tokenizer.special_toks_ids
        self.punct_tok_ids = tokenizers.q_tokenizer.punctuations
        self.q_maxlen = tokenizers.q_tokenizer.cfg.max_len
        self.intra_loss_coeff = cfg.intra_loss_coeff
        self.inter_loss_coeff = cfg.inter_loss_coeff

        # Projection layers
        self.tok_projection_layer = torch.nn.Linear(
            self.llm.config.hidden_size,
            cfg.out_dim,
            bias=True,
        )

        # TODO: Move the post processing to the base class
        self.load_checkpoint()

    def forward(
        self,
        q_tok_ids: torch.Tensor,
        q_tok_att_mask: torch.Tensor,
        q_tok_mask: torch.Tensor,
        doc_tok_ids: torch.Tensor,
        doc_tok_att_mask: torch.Tensor,
        doc_tok_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        distillation_scores: Optional[torch.Tensor] = None,
        is_inference: Optional[bool] = False,
        is_analyze: Optional[bool] = False,
        **kwargs,
    ) -> torch.Tensor:
        # Configs
        bsize, nway, dim = doc_tok_ids.shape
        ib_nhard = nway // bsize
        is_eval = labels is not None
        doc_tok_mask = doc_tok_mask.view(-1, doc_tok_mask.shape[-1]).unsqueeze(-1)
        assert (
            q_tok_ids.shape[0] == bsize
        ), f"Batch size is not consistent: {q_tok_ids.shape[0]} vs {bsize}"
        # Encode
        (
            q_encoded,
            q_tok_projected,
            q_tok_scale_factor,
        ) = self.encode_q_text(
            tok_ids=q_tok_ids,
            att_mask=q_tok_att_mask,
            tok_mask=q_tok_mask,
        )
        d_tok_projected = self.encode_d_text(
            tok_ids=doc_tok_ids,
            att_mask=doc_tok_att_mask,
            nway=nway,
        )

        intra_scores, inter_scores, _, _ = self.compute_scores(
            q_encoded=q_tok_projected,
            d_encoded=d_tok_projected,
            q_weight=None,
            q_scale_factor=q_tok_scale_factor,
            q_mask=q_tok_mask,
            d_mask=doc_tok_mask,
            d_weight_intra=None,
            d_weight_inter=None,
            nway=nway,
            ib_nhard=ib_nhard,
        )

        # Compute loss
        device = intra_scores.device
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

        # Initialize return dictionary
        return_dict = {
            "loss": loss,
            "intra_loss": intra_loss,
            "inter_loss": inter_loss,
            "kl_loss": 0 if kl_loss is None else kl_loss,
        }
        if is_eval:
            return return_dict, intra_scores.reshape(bsize, -1)
        return return_dict

    def encode_text(
        self,
        tok_ids: torch.Tensor,
        att_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # LLM encoding
        encoded_tok_vectors = self.llm(
            tok_ids, attention_mask=att_mask
        ).last_hidden_state

        # Perform projection for token
        projected_tok_vectors = self.tok_projection_layer(encoded_tok_vectors)

        return (
            encoded_tok_vectors,
            projected_tok_vectors,
        )

    def encode_q_text(
        self,
        tok_ids: torch.Tensor,
        att_mask: torch.Tensor,
        tok_mask: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        (
            encoded_tok_vectors,
            projected_tok_vectors,
        ) = self.encode_text(tok_ids, att_mask)
        dtype = projected_tok_vectors.dtype

        # Compute normalization scale for each query
        token_scale_factor = self.get_scale_factor(mask=tok_mask)

        # Normalize
        projected_tok_vectors = torch.nn.functional.normalize(
            projected_tok_vectors, p=2, dim=2
        )
        if projected_tok_vectors.dtype != dtype:
            projected_tok_vectors = projected_tok_vectors.to(dtype)

        return (
            encoded_tok_vectors,
            projected_tok_vectors,
            token_scale_factor,
        )

    def encode_d_text(
        self,
        tok_ids: torch.Tensor,
        att_mask: torch.Tensor,
        nway: int = None,
    ) -> torch.Tensor:
        # Configs
        if len(tok_ids.shape) == 3:
            bsize, ndoc, max_len = tok_ids.shape
            nhard = nway // bsize
            tok_ids_combined = tok_ids.view(-1, max_len)
            att_mask_combined = att_mask.view(-1, max_len)
        elif len(tok_ids.shape) == 2:
            bsize, max_len = tok_ids.shape
            nhard = 1
            tok_ids_combined = tok_ids
            att_mask_combined = att_mask
        (
            encoded_tok_vectors,
            projected_tok_vectors,
        ) = self.encode_text(tok_ids_combined, att_mask_combined)

        dtype = projected_tok_vectors.dtype

        # Normalize
        projected_tok_vectors = torch.nn.functional.normalize(
            projected_tok_vectors, p=2, dim=2
        )
        if projected_tok_vectors.dtype != dtype:
            projected_tok_vectors = projected_tok_vectors.to(dtype)

        return projected_tok_vectors

    def compute_scores(
        self,
        q_encoded: torch.Tensor,
        d_encoded: torch.Tensor,
        q_weight: Optional[torch.Tensor],
        q_scale_factor: Optional[torch.Tensor],
        q_mask: Optional[torch.Tensor],
        d_mask: Optional[torch.Tensor],
        d_weight_intra: Optional[torch.Tensor],
        d_weight_inter: Optional[torch.Tensor],
        nway: int,
        ib_nhard: int,
        return_max_scores: bool = False,
        return_entire_scores: bool = False,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]
    ]:
        bsize = q_encoded.shape[0]
        # Apply weights
        if q_weight is not None:
            q_encoded = q_encoded * q_weight

        # Perform Maxsim
        intra_scores, intra_q_max_scores, intra_qd_scores, _ = (
            self.compute_intra_scores(
                q_encoded,
                q_mask=q_mask,
                d_encoded=d_encoded,
                d_weight=d_weight_intra,
                d_mask=d_mask,
                nway=nway,
                return_max_scores=return_max_scores,
                return_entire_scores=return_entire_scores,
            )
        )

        # For optimizing the memory usage
        inter_scores, inter_q_max_scores, inter_qd_scores, _ = (
            self.compute_inter_scores(
                q_encoded,
                q_mask=q_mask,
                d_encoded=d_encoded,
                d_weight=d_weight_inter,
                d_mask=d_mask,
                nway=nway,
                ib_nhard=ib_nhard,
                return_max_scores=return_max_scores,
                return_entire_scores=return_entire_scores,
            )
        )

        # Apply scale factor
        if q_scale_factor is not None:
            intra_scale_factors = q_scale_factor.repeat_interleave(nway)
            intra_scores = intra_scores * intra_scale_factors
            inter_scale_factors = q_scale_factor.repeat_interleave(
                ib_nhard * (bsize - 1) + 1
            )
            inter_scores = inter_scores * inter_scale_factors

        return intra_scores, inter_scores, intra_q_max_scores, intra_qd_scores

    def compute_intra_scores(
        self,
        q_encoded: torch.Tensor,
        q_mask: torch.Tensor,
        d_encoded: torch.Tensor,
        d_weight: torch.Tensor,
        d_mask: Optional[torch.Tensor],
        nway: int,
        return_max_scores: bool = False,
        return_entire_scores: bool = False,
    ) -> torch.Tensor:
        # Perform Maxsim
        if d_weight is not None:
            d_encoded = d_encoded * d_weight
        q_encoded = q_encoded.repeat_interleave(nway, dim=0)
        if q_mask is not None:
            q_mask = q_mask.repeat_interleave(nway, dim=0)
        return compute_sum_maxsim(
            q_encoded=q_encoded,
            k_encoded=d_encoded,
            q_mask=q_mask,
            k_mask=d_mask,
            return_max_scores=return_max_scores,
            return_element_wise_scores=return_entire_scores,
        )

    def compute_inter_scores(
        self,
        q_encoded: torch.Tensor,
        q_mask: torch.Tensor,
        d_encoded: torch.Tensor,
        d_weight: torch.Tensor,
        d_mask: Optional[torch.Tensor],
        nway: int,
        ib_nhard: int,
        return_max_scores: bool = False,
        return_entire_scores: bool = False,
    ) -> torch.Tensor:
        # Compute the scores for the ib_loss
        bsize = q_encoded.shape[0]
        repeat_num = ib_nhard * (bsize - 1) + 1
        q_encoded = q_encoded.repeat_interleave(repeat_num, dim=0)
        if q_mask is not None:
            q_mask = q_mask.repeat_interleave(repeat_num, dim=0)
        # Get the indices
        d_indices_tensor: torch.Tensor = doc_indices_for_ib_loss(
            bsize, nway, ib_nhard, return_as_tensor=True, device=d_encoded.device
        )
        d_encoded = d_encoded[d_indices_tensor]
        if d_mask is not None:
            d_mask = d_mask[d_indices_tensor]
        # Apply weights if exists
        if d_weight is not None:
            d_encoded = d_encoded * d_weight
        # Compute the scores
        return compute_sum_maxsim(
            q_encoded=q_encoded,
            k_encoded=d_encoded,
            q_mask=q_mask,
            k_mask=d_mask,
            return_max_scores=return_max_scores,
            return_element_wise_scores=return_entire_scores,
        )

    def get_valid_num(self, mask: torch.Tensor) -> torch.Tensor:
        num_non_valid_tokens = mask.sum(dim=1)
        target_scale = get_target_scale_tensor(
            target_scale=mask.shape[1],
            b_size=num_non_valid_tokens.shape[0],
            device=num_non_valid_tokens.device,
            dtype=num_non_valid_tokens.dtype,
        )
        num_valid_tokens = target_scale - num_non_valid_tokens
        return num_valid_tokens

    def get_scale_factor(self, mask: torch.Tensor) -> torch.Tensor:
        num_valid_tokens = self.get_valid_num(mask == 0)
        return self.q_maxlen / num_valid_tokens
