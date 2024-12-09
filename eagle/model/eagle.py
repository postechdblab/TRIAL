import logging
from typing import *

import hkkang_utils.list as list_utils
import torch
import tqdm
from omegaconf import DictConfig
from torch.nn.utils.rnn import pad_sequence

from eagle.dataset.corpus import Document
from eagle.dataset.utils import get_mask
from eagle.model.base_model import BaseModel
from eagle.model.objective import compute_loss, doc_indices_for_ib_loss
from eagle.model.utils import (
    _sort_by_length,
    _split_into_batches,
    aggregate_vectors_with_indices,
    get_scale_factor,
    get_valid_num,
    get_weight_layer,
    l1_regularization,
    l2_regularization,
)
from eagle.search.algorithm import token_interaction_with_relation
from eagle.tokenization.tokenizers import Tokenizers

logger = logging.getLogger("EAGLE")


class EAGLE(BaseModel):
    def __init__(self, cfg: DictConfig, tokenizers: Tokenizers) -> None:
        super().__init__(cfg=cfg, tokenizers=tokenizers)
        self.q_special_tok_ids = tokenizers.q_tokenizer.special_toks_ids
        self.d_special_tok_ids = tokenizers.d_tokenizer.special_toks_ids
        self.punct_tok_ids = tokenizers.q_tokenizer.punctuations
        self.q_maxlen = tokenizers.q_tokenizer.cfg.max_len

        # Ideas
        self.is_use_q_weight = cfg.is_use_q_weight
        self.is_use_d_weight = cfg.is_use_d_weight
        self.w_regularize_strategy = cfg.w_regularize_strategy
        self.q_weight_strategy = cfg.q_weight_strategy
        self.d_weight_strategy = cfg.d_weight_strategy
        self.q_weight_coeff = cfg.q_weight_coeff
        self.d_weight_coeff = cfg.d_weight_coeff
        self.intra_loss_coeff = cfg.intra_loss_coeff
        self.inter_loss_coeff = cfg.inter_loss_coeff
        self.sim_type = cfg.sim_type
        self.use_attn_for_phrase_encoding = False
        self.use_multi_doc_granularity = (
            cfg.use_multi_doc_granularity
            if "use_multi_doc_granularity" in cfg
            else False
        )
        self.use_phrase_level = (
            cfg.use_phrase_level if "use_phrase_level" in cfg else False
        )
        self.use_sum_when_training = (
            cfg.use_sum_when_training if "use_sum_when_training" in cfg else False
        )
        self.relation_scale_factor = (
            cfg.relation_scale_factor if "relation_scale_factor" in cfg else 0
        )

        # Layers to encode the query into phrase level
        if self.use_attn_for_phrase_encoding:
            self.attn_for_phrase_embedding = torch.nn.MultiheadAttention(
                embed_dim=self.llm.config.hidden_size,
                num_heads=8,
                batch_first=True,
            )
            self.phrase_projection_layer = None
        else:
            self.attn_for_phrase_embedding = None
            self.phrase_projection_layer = torch.nn.Linear(cfg.out_dim, cfg.out_dim)

        # Projection layers
        self.tok_projection_layer = torch.nn.Linear(
            self.llm.config.hidden_size,
            cfg.out_dim,
            bias=True,
        )

        # Pooling for phrase level embeddings
        self.reduce_strategy = cfg.reduce_strategy

        # Layers to predict the weights (i.e., importance)
        self.q_weight_layer = self.__create_q_weight_layer(
            input_dim=cfg.out_dim,
            intermediate_dim=cfg.out_dim,
        )
        self.d_weight_layer = self.__create_d_weight_layer(
            input_dim=cfg.out_dim,
            intermediate_dim=cfg.out_dim,
        )
        self.d_weight_layer_norm = (
            torch.nn.LayerNorm(cfg.out_dim) if self.is_use_d_weight else None
        )

        # Cross-attention layer for interacting query and documents
        self.cross_att_layer = self.__create_cross_att_layer()

        # Regularization for the weights (i.e., importance)
        self.regularization = self.__create_regularization_func(
            strategy=cfg.w_regularize_strategy
        )
        self.relation_encoder = None
        if self.use_relation:
            # Related to the relation between tokens
            self.relation_encoder = torch.nn.Sequential(
                torch.nn.Linear(cfg.out_dim * 2, cfg.out_dim),
                torch.nn.LayerNorm(cfg.out_dim),
                torch.nn.Mish(),
                torch.nn.Linear(cfg.out_dim, cfg.out_dim),
            )

        # TODO: Move the post processing to the base class
        self.load_checkpoint()

    @property
    def use_relation(self) -> bool:
        return self.relation_scale_factor > 0

    @property
    def q_skiplist(self) -> List[int]:
        return self.q_special_tok_ids

    @property
    def d_skiplist(self) -> List[int]:
        return self.d_special_tok_ids
        # return self.d_special_tok_ids + self.punct_tok_ids

    def __create_cross_att_layer(self) -> torch.nn.Module:
        if self.is_use_d_weight:
            return torch.nn.MultiheadAttention(
                embed_dim=self.cfg.out_dim,
                num_heads=8,
                batch_first=True,
            )
        return None

    def __create_q_weight_layer(
        self, input_dim: int, intermediate_dim: int
    ) -> torch.nn.Module:
        if self.is_use_q_weight:
            return get_weight_layer(
                strategy=self.q_weight_strategy,
                input_dim=input_dim,
                intermediate_dim=intermediate_dim,
                out_dim=1,
            )
        return None

    def __create_d_weight_layer(
        self, input_dim: int, intermediate_dim: int
    ) -> torch.nn.Module:
        if self.is_use_d_weight:
            return get_weight_layer(
                strategy=self.d_weight_strategy,
                input_dim=input_dim,
                intermediate_dim=intermediate_dim,
                out_dim=1,
            )
        return None

    def __create_regularization_func(
        self, strategy: str
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        if strategy == "l1":
            return l1_regularization
        elif strategy == "l2":
            return l2_regularization
        raise ValueError(f"Unsupported regularization strategy: {strategy}")

    def forward(
        self,
        q_tok_ids: torch.Tensor,
        q_tok_att_mask: torch.Tensor,
        q_phrase_scatter_indices: torch.Tensor,
        q_sent_start_indices: List[List[int]],
        q_tok_mask: torch.Tensor,
        q_phrase_mask: torch.Tensor,
        q_sent_mask: torch.Tensor,
        doc_tok_ids: torch.Tensor,
        doc_tok_att_mask: torch.Tensor,
        doc_phrase_scatter_indices: torch.Tensor,
        doc_sent_start_indices: List[List[List[int]]],
        doc_tok_mask: torch.Tensor,
        doc_phrase_mask: torch.Tensor,
        doc_sent_mask: torch.Tensor,
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
        assert (
            q_tok_ids.shape[0] == bsize
        ), f"Batch size is not consistent: {q_tok_ids.shape[0]} vs {bsize}"

        # Encode
        (
            q_encoded,
            q_tok_projected,
            q_phrase_projected,
            q_sent_projected,
            q_tok_weight,
            q_phrase_weight,
            q_sent_weight,
            q_tok_scale_factor,
            q_phrase_scale_factor,
            q_sent_scale_factor,
        ) = self.encode_q_text(
            tok_ids=q_tok_ids,
            att_mask=q_tok_att_mask,
            tok_mask=q_tok_mask,
            phrase_mask=q_phrase_mask,
            sent_mask=q_sent_mask,
            phrase_scatter_indices=q_phrase_scatter_indices,
            sent_start_indices=q_sent_start_indices,
        )
        (
            d_tok_projected,
            d_phrase_projected,
            d_sent_projected,
            d_tok_weight_intra,
            d_tok_weight_inter,
            d_phrase_weight_intra,
            d_phrase_weight_inter,
            d_sent_weight_intra,
            d_sent_weight_inter,
        ) = self.encode_d_text(
            tok_ids=doc_tok_ids,
            att_mask=doc_tok_att_mask,
            tok_mask=doc_tok_mask,
            phrase_mask=doc_phrase_mask,
            sent_mask=doc_sent_mask,
            scatter_indices=doc_phrase_scatter_indices,
            sent_start_indices=doc_sent_start_indices,
            q_vectors=q_tok_projected,
            q_mask=q_tok_mask,
            nway=nway,
            is_inference=is_inference,
        )

        (
            intra_scores,
            inter_scores,
            intra_qd_inner_scores,
            intra_qd_outer_scores,
            intra_selected_d_weights,
        ) = self.compute_scores(
            q_sent=q_sent_projected,
            q_phrase=q_phrase_projected,
            q_tok=q_tok_projected,
            d_sent=d_sent_projected,
            d_phrase=d_phrase_projected,
            d_tok=d_tok_projected,
            q_tok_weight=q_tok_weight,
            q_phrase_weight=q_phrase_weight,
            d_tok_weight_intra=d_tok_weight_intra,
            d_tok_weight_inter=d_tok_weight_inter,
            d_phrase_weight_intra=d_phrase_weight_intra,
            d_phrase_weight_inter=d_phrase_weight_inter,
            d_sent_weight_inter=d_sent_weight_inter,
            d_sent_weight_intra=d_sent_weight_intra,
            q_scatter_indices=q_phrase_scatter_indices,
            q_tok_scale_factor=q_tok_scale_factor,
            q_phrase_scale_factor=q_phrase_scale_factor,
            q_sent_scale_factor=q_sent_scale_factor,
            is_inference=is_inference,
            return_element_wise_scores=is_analyze,
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

        # Compute weight regularization for query
        q_weight_reg_term = 0
        if q_tok_weight is not None:
            q_weight_reg_term = self.regularization(q_tok_weight)
            loss = loss + self.q_weight_coeff * q_weight_reg_term

        # Compute weight regularization for document
        d_weight_reg_term = 0
        if d_tok_weight_intra is not None:
            d_weight_reg_term = self.regularization(d_tok_weight_intra)
            if d_tok_weight_inter is not None:
                d_weight_reg_term = d_weight_reg_term + self.regularization(
                    d_tok_weight_inter
                )
            loss = loss + self.d_weight_coeff * d_weight_reg_term

        # Initialize return dictionary
        return_dict = {
            "loss": loss,
            "intra_loss": intra_loss,
            "inter_loss": inter_loss,
            "kl_loss": 0 if kl_loss is None else kl_loss,
        }

        # Analyze query weights
        q_weight_ratio = 0
        q_weight_var = 0
        if q_weight_reg_term:
            num_valid = get_valid_num(q_tok_mask)
            q_tok_weight = q_tok_weight.masked_fill(q_tok_mask.unsqueeze(-1) == True, 0)
            q_weight_ratio = q_tok_weight.sum() / num_valid.sum()
            q_weight_var = q_tok_weight[q_tok_mask == False].var()
        # Analyze document weights
        d_weight_intra_ratio = 0
        d_weight_inter_ratio = 0
        d_weight_intra_var = 0
        d_weight_inter_var = 0
        if d_tok_weight_intra is not None:
            reshaped_doc_tok_mask = doc_tok_mask.reshape(
                -1, doc_tok_mask.shape[-1]
            ).unsqueeze(-1)
            d_mask_intra = reshaped_doc_tok_mask
            d_weight_intra_masked = d_tok_weight_intra * d_mask_intra
            d_weight_intra_ratio = d_weight_intra_masked.sum() / d_mask_intra.sum()
            # Compute variance
            d_weight_intra_var = d_weight_intra_masked[d_mask_intra == False].var()
            if d_tok_weight_inter is not None:
                d_indices = doc_indices_for_ib_loss(
                    bsize,
                    nway,
                    ib_nhard,
                    return_as_tensor=True,
                    device=doc_tok_mask.device,
                )
                d_mask_inter = reshaped_doc_tok_mask[d_indices]
                d_weight_inter_masked = d_tok_weight_inter * d_mask_inter
                d_weight_inter_ratio = d_weight_inter_masked.sum() / d_mask_inter.sum()
                d_weight_inter_var = d_weight_inter_masked[d_mask_inter == False].var()

        # Append more log information
        return_dict["avg_intra_scores"] = intra_scores.mean().item()
        return_dict["avg_inter_scores"] = inter_scores.mean().item()
        return_dict["q_weight_reg"] = q_weight_reg_term
        return_dict["d_weight_reg"] = d_weight_reg_term
        return_dict["q_weight_ratio"] = q_weight_ratio
        return_dict["d_weight_intra_ratio"] = d_weight_intra_ratio
        return_dict["d_weight_inter_ratio"] = d_weight_inter_ratio
        return_dict["q_weight_var"] = q_weight_var
        return_dict["d_weight_intra_var"] = d_weight_intra_var
        return_dict["d_weight_inter_var"] = d_weight_inter_var

        if is_inference:
            # Weights
            return_dict["q_weight"] = q_tok_weight
            return_dict["d_weight_intra"] = d_tok_weight_intra
            return_dict["d_weight_inter"] = d_tok_weight_inter
            # Mask
            return_dict["q_mask"] = q_tok_mask
            return_dict["d_mask"] = doc_tok_mask

        # Append weights
        if is_eval:
            return return_dict, intra_scores.reshape(bsize, -1)
        elif is_analyze:
            return_dict["intra_qd_inner_scores"] = intra_qd_inner_scores
            return_dict["intra_qd_outer_scores"] = intra_qd_outer_scores
            return_dict["intra_q_weights"] = q_tok_weight
            return_dict["intra_selected_d_weights"] = intra_selected_d_weights
        return return_dict

    def encode_text(
        self,
        tok_ids: torch.Tensor,
        att_mask: torch.Tensor,
        phrase_scatter_indices: torch.Tensor = None,
        sent_start_indices: List[List[int]] = None,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]
    ]:
        # LLM encoding
        encoded_tok_vectors = self.llm(
            tok_ids, attention_mask=att_mask
        ).last_hidden_state

        # Perform lower-dimension projection in token-level
        projected_tok_vectors = self.tok_projection_layer(encoded_tok_vectors)

        # Handle phrase embeddings
        encoded_phrase_vectors = None
        projected_phrase_vectors = None
        # Encode phrase-level embeddings
        if self.use_phrase_level and phrase_scatter_indices is not None:
            if self.use_attn_for_phrase_encoding:
                # TODO: Make different phrases into a batch
                # TODO: Add positional encoding
                pass
            else:
                if projected_tok_vectors.shape[1] != phrase_scatter_indices.shape[1]:
                    raise RuntimeError("Shape mismatch")
                encoded_phrase_vectors = aggregate_vectors_with_indices(
                    src_tensor=projected_tok_vectors,
                    scatter_indices=phrase_scatter_indices,
                    reduce=self.reduce_strategy,
                )
                projected_phrase_vectors = encoded_phrase_vectors
                # projected_phrase_vectors = self.phrase_projection_layer(
                #     encoded_phrase_vectors
                # )

        # Handle sentence-level embeddings
        projected_sent_vectors = None
        if sent_start_indices is not None:
            projected_sent_vectors = []
            for b_idx, start_indices in enumerate(sent_start_indices):
                projected_sent_vectors.append(
                    projected_tok_vectors[b_idx, start_indices]
                )
            projected_sent_vectors = pad_sequence(
                projected_sent_vectors, batch_first=True
            )

        return (
            encoded_tok_vectors,
            projected_tok_vectors,
            projected_phrase_vectors,
            projected_sent_vectors,
        )

    def encode_q_text(
        self,
        tok_ids: torch.Tensor,
        att_mask: torch.Tensor,
        tok_mask: torch.Tensor,
        phrase_mask: torch.Tensor = None,
        sent_mask: torch.Tensor = None,
        phrase_scatter_indices: Optional[torch.Tensor] = None,
        sent_start_indices: Optional[List[List[int]]] = None,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        (
            encoded_tok_vectors,
            projected_tok_vectors,
            projected_phrase_vectors,
            projected_sent_vectors,
        ) = self.encode_text(
            tok_ids, att_mask, phrase_scatter_indices, sent_start_indices
        )
        # Dynamic configs
        dtype = projected_tok_vectors.dtype
        is_create_phrase_vectors = (
            self.use_phrase_level and projected_phrase_vectors is not None
        )
        is_create_sent_vectors = projected_sent_vectors is not None

        # Mask paddings
        projected_tok_vectors.masked_fill_(tok_mask.unsqueeze(-1) == True, 0)
        if is_create_phrase_vectors:
            projected_phrase_vectors.masked_fill_(phrase_mask.unsqueeze(-1) == True, 0)
        if is_create_sent_vectors:
            projected_sent_vectors.masked_fill_(sent_mask.unsqueeze(-1) == True, 0)

        # Weights
        tok_weights = None
        phrase_weights = None
        sentence_weights = None
        if self.is_use_q_weight:
            tok_weights = self.q_weight_layer(projected_tok_vectors)
            if is_create_phrase_vectors:
                phrase_weights = self.q_weight_layer(projected_phrase_vectors)
            if is_create_sent_vectors:
                sentence_weights = self.q_weight_layer(projected_sent_vectors)

        # Compute normalization scale for each query
        token_scale_factor = get_scale_factor(mask=tok_mask, q_maxlen=self.q_maxlen)

        sentence_scale_factor = None
        if is_create_sent_vectors:
            sentence_scale_factor = torch.full(
                size=(projected_sent_vectors.shape[:-1]),
                fill_value=self.q_maxlen,
                dtype=projected_sent_vectors.dtype,
                device=tok_ids.device,
            )

        phrase_scale_factor = None
        if is_create_phrase_vectors:
            phrase_scale_factor = get_scale_factor(
                mask=phrase_mask, q_maxlen=self.q_maxlen
            )

        # Normalize
        projected_tok_vectors = torch.nn.functional.normalize(
            projected_tok_vectors, p=2, dim=2
        )
        if projected_tok_vectors.dtype != dtype:
            projected_tok_vectors = projected_tok_vectors.to(dtype)
        if is_create_sent_vectors:
            projected_sent_vectors = torch.nn.functional.normalize(
                projected_sent_vectors, p=2, dim=2
            )
            if projected_sent_vectors.dtype != dtype:
                projected_sent_vectors = projected_sent_vectors.to(dtype)
        if is_create_phrase_vectors:
            projected_phrase_vectors = torch.nn.functional.normalize(
                projected_phrase_vectors, p=2, dim=2
            )
            if projected_phrase_vectors.dtype != dtype:
                projected_phrase_vectors = projected_phrase_vectors.to(dtype)

        return (
            encoded_tok_vectors,
            projected_tok_vectors,
            projected_phrase_vectors,
            projected_sent_vectors,
            tok_weights,
            phrase_weights,
            sentence_weights,
            token_scale_factor,
            phrase_scale_factor,
            sentence_scale_factor,
        )

    def encode_d_text(
        self,
        tok_ids: torch.Tensor,
        att_mask: torch.Tensor,
        tok_mask: torch.Tensor,
        phrase_mask: torch.Tensor = None,
        sent_mask: torch.Tensor = None,
        scatter_indices: torch.Tensor = None,
        sent_start_indices: List[List[List[int]]] = None,
        q_vectors: torch.Tensor = None,
        q_mask: torch.Tensor = None,
        nway: int = None,
        is_encoding: bool = False,
        is_inference: bool = False,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        # Configs
        if len(tok_ids.shape) == 3:
            bsize, ndoc, max_tok_len = tok_ids.shape
            if phrase_mask is not None:
                _, _, max_phrase_len = phrase_mask.shape
                phrase_mask_combined = phrase_mask.view(-1, max_phrase_len)
            if sent_mask is not None:
                _, _, max_sent_len = sent_mask.shape
                sent_mask_combined = sent_mask.view(-1, max_sent_len)
            nhard = nway // bsize
            tok_ids_combined = tok_ids.view(-1, max_tok_len)
            att_mask_combined = att_mask.view(-1, max_tok_len)
            tok_mask_combined = tok_mask.view(-1, max_tok_len)
            if sent_start_indices is not None:
                sent_start_indices = list_utils.do_flatten_list(sent_start_indices)
        elif len(tok_ids.shape) == 2:
            bsize, max_len = tok_ids.shape
            nhard = 1
            tok_ids_combined = tok_ids
            att_mask_combined = att_mask
            tok_mask_combined = tok_mask
            if phrase_mask is not None:
                phrase_mask_combined = phrase_mask
            if sent_mask is not None:
                sent_mask_combined = sent_mask

        # Encode text
        (
            encoded_tok_vectors,
            projected_tok_vectors,
            projected_phrase_vectors,
            projected_sent_vectors,
        ) = self.encode_text(
            tok_ids_combined, att_mask_combined, scatter_indices, sent_start_indices
        )

        # Dynamic configs
        dtype = projected_tok_vectors.dtype
        is_create_phrase_vectors = (
            self.use_phrase_level and projected_phrase_vectors is not None
        )
        is_create_sent_vectors = projected_sent_vectors is not None

        # Apply mask
        projected_tok_vectors = projected_tok_vectors.masked_fill(
            tok_mask_combined.unsqueeze(-1) == True, 0
        )
        if is_create_phrase_vectors:
            projected_phrase_vectors = projected_phrase_vectors.masked_fill(
                phrase_mask_combined.unsqueeze(-1) == True, 0
            )
        if is_create_sent_vectors:
            projected_sent_vectors = projected_sent_vectors.masked_fill(
                sent_mask_combined.unsqueeze(-1) == True, 0
            )

        # Create weight using q_vetors
        tok_weights_intra = None
        tok_weights_inter = None
        phrase_weights_intra = None
        phrase_weights_inter = None
        sent_weights_intra = None
        sent_weights_inter = None
        if self.is_use_d_weight and not is_encoding:
            # Further encode with q_vectors for intra-example weights
            q_vectors_intra = q_vectors.repeat_interleave(nway, dim=0)
            q_mask_intra = q_mask.repeat_interleave(nway, dim=0)
            # Perform cross-attention
            cross_encoded_tok_vectors_intra, cross_attn_weights_intra = (
                self.cross_att_layer(
                    projected_tok_vectors,
                    q_vectors_intra,
                    q_vectors_intra,
                    key_padding_mask=q_mask_intra.squeeze(-1),
                )
            )
            # Add and normalize
            cross_encoded_tok_vectors_intra = (
                cross_encoded_tok_vectors_intra + projected_tok_vectors
            )
            cross_encoded_tok_vectors_intra = self.d_weight_layer_norm(
                cross_encoded_tok_vectors_intra
            )
            # Predict token-level weights
            tok_weights_intra = self.d_weight_layer(cross_encoded_tok_vectors_intra)
            # Predict phrase-level weights
            if is_create_phrase_vectors:
                phrase_weights_intra = aggregate_vectors_with_indices(
                    src_tensor=tok_weights_intra,
                    scatter_indices=scatter_indices,
                    reduce=self.reduce_strategy,
                )
            # Get sentence-level weights
            if is_create_sent_vectors:
                sent_weights_intra = []
                for b_idx, start_indices in enumerate(sent_start_indices):
                    sent_weights_intra.append(tok_weights_intra[b_idx, start_indices])
                sent_weights_intra = pad_sequence(sent_weights_intra, batch_first=True)
            # Further encode with q_vectors for inter-example weights (only for training and evaluation)
            if not is_inference:
                repeat_n = nhard * (bsize - 1) + 1
                q_vectors_inter = q_vectors.repeat_interleave(repeat_n, dim=0)
                q_mask_inter = q_mask.repeat_interleave(repeat_n, dim=0)
                # Get indices for the inter-examples
                doc_indices: List[int] = doc_indices_for_ib_loss(
                    bsize,
                    nway,
                    nhard,
                    return_as_tensor=True,
                    device=encoded_tok_vectors.device,
                )
                selected_encoded_tok_vectors = projected_tok_vectors[doc_indices]
                # Perform cross-attention
                cross_encoded_tok_vectors_inter, cross_attn_weights_inter = (
                    self.cross_att_layer(
                        query=selected_encoded_tok_vectors,
                        key=q_vectors_inter,
                        value=q_vectors_inter,
                        key_padding_mask=q_mask_inter.squeeze(-1),
                    )
                )
                # Add and normalize
                cross_encoded_tok_vectors_inter = (
                    cross_encoded_tok_vectors_inter + selected_encoded_tok_vectors
                )
                cross_encoded_tok_vectors_inter = self.d_weight_layer_norm(
                    cross_encoded_tok_vectors_inter
                )
                # Predict the weights
                tok_weights_inter = self.d_weight_layer(cross_encoded_tok_vectors_inter)
                if is_create_phrase_vectors:
                    selected_scatter_indices = scatter_indices[doc_indices]
                    phrase_weights_inter = aggregate_vectors_with_indices(
                        src_tensor=tok_weights_inter,
                        scatter_indices=selected_scatter_indices,
                        reduce=self.reduce_strategy,
                    )
                if is_create_sent_vectors:
                    sent_weights_inter = []
                    selected_sent_start_indices = [
                        sent_start_indices[d_idx] for d_idx in doc_indices
                    ]
                    for b_idx, start_indices in enumerate(selected_sent_start_indices):
                        sent_weights_inter.append(
                            tok_weights_inter[b_idx, start_indices]
                        )
                    sent_weights_inter = pad_sequence(
                        sent_weights_inter, batch_first=True
                    )

        # Normalize
        projected_tok_vectors = torch.nn.functional.normalize(
            projected_tok_vectors, p=2, dim=2
        )
        if projected_tok_vectors.dtype != dtype:
            projected_tok_vectors = projected_tok_vectors.to(dtype)
        if is_create_sent_vectors:
            projected_sent_vectors = torch.nn.functional.normalize(
                projected_sent_vectors, p=2, dim=2
            )
            if projected_sent_vectors.dtype != dtype:
                projected_sent_vectors = projected_sent_vectors.to(dtype)
        if is_create_phrase_vectors:
            projected_phrase_vectors = torch.nn.functional.normalize(
                projected_phrase_vectors, p=2, dim=2
            )
            if projected_phrase_vectors.dtype != dtype:
                projected_phrase_vectors = projected_phrase_vectors.to(dtype)

        return (
            projected_tok_vectors,
            projected_phrase_vectors,
            projected_sent_vectors,
            tok_weights_intra,
            tok_weights_inter,
            phrase_weights_intra,
            phrase_weights_inter,
            sent_weights_intra,
            sent_weights_inter,
        )

    def compute_scores(
        self,
        q_sent: torch.Tensor,
        q_phrase: torch.Tensor,
        q_tok: torch.Tensor,
        d_sent: torch.Tensor,
        d_phrase: torch.Tensor,
        d_tok: torch.Tensor,
        q_tok_weight: torch.Tensor,
        q_phrase_weight: torch.Tensor,
        d_tok_weight_intra: torch.Tensor,
        d_tok_weight_inter: torch.Tensor,
        d_phrase_weight_intra: torch.Tensor,
        d_phrase_weight_inter: torch.Tensor,
        d_sent_weight_inter: torch.Tensor,
        d_sent_weight_intra: torch.Tensor,
        q_scatter_indices: torch.Tensor,
        q_tok_scale_factor: torch.Tensor,
        q_phrase_scale_factor: torch.Tensor,
        q_sent_scale_factor: torch.Tensor,
        is_inference: bool = False,
        return_element_wise_scores: bool = False,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]
    ]:
        # Configurations
        bsize, q_tok_len, q_dim = q_tok.shape
        nway = d_tok.shape[0] // bsize
        ib_nhard = nway // bsize
        repeat_num_for_inter = ib_nhard * (bsize - 1) + 1
        d_inter_indices: torch.Tensor = doc_indices_for_ib_loss(
            bsize,
            nway,
            ib_nhard,
            return_as_tensor=True,
            device=d_tok.device,
        )

        # Prepare intra scores
        q_vecs_intra = q_tok.repeat_interleave(nway, dim=0)
        q_scale_factors_intra = q_tok_scale_factor.repeat_interleave(nway, dim=0)

        q_weight_intra = None
        if q_tok_weight is not None:
            q_weight_intra = q_tok_weight.repeat_interleave(nway, dim=0)
        q_scatter_indices_intra = None
        if self.use_phrase_level and q_scatter_indices is not None:
            q_scatter_indices_intra = q_scatter_indices.repeat_interleave(nway, dim=0)

        # Compute intra scores
        if self.use_multi_doc_granularity:
            (
                intra_sim_scores,
                intra_sim_elementwise_scores,
                intra_sim_selected_d_weights,
            ) = self.multi_granularity_interaction()
        else:
            (
                intra_sim_scores,
                intra_sim_elementwise_scores,
                intra_sim_selected_d_weights,
            ) = token_interaction_with_relation(
                q_tok=q_vecs_intra,
                q_tok_weight=q_weight_intra,
                d_tok=d_tok,
                q_scale_factors=q_scale_factors_intra,
                relation_encoder=self.relation_encoder,
                relation_scale_factor=self.relation_scale_factor,
                return_element_wise_scores=return_element_wise_scores,
            )

        if not is_inference:
            # Prepare inter scores
            q_vecs_inter = q_tok.repeat_interleave(repeat_num_for_inter, dim=0)
            q_scale_factors_inter = q_tok_scale_factor.repeat_interleave(
                repeat_num_for_inter, dim=0
            )
            q_weight_inter = None
            if q_tok_weight is not None:
                q_weight_inter = q_tok_weight.repeat_interleave(
                    repeat_num_for_inter, dim=0
                )
            q_scatter_indices_inter = None
            if self.use_phrase_level and q_scatter_indices is not None:
                q_scatter_indices_inter = q_scatter_indices.repeat_interleave(
                    repeat_num_for_inter, dim=0
                )
            selected_d_vecs_inter: torch.Tensor = d_tok[d_inter_indices]

            # Compute inter scores
            if self.use_multi_doc_granularity:
                (
                    inter_sim_scores,
                    inter_sim_elementwise_scores,
                    inter_sim_selected_d_weights,
                ) = self.multi_granularity_interaction()
            else:
                (
                    inter_sim_scores,
                    inter_sim_elementwise_scores,
                    inter_sim_selected_d_weights,
                ) = token_interaction_with_relation(
                    q_tok=q_vecs_inter,
                    q_tok_weight=q_weight_inter,
                    d_tok=selected_d_vecs_inter,
                    q_scale_factors=q_scale_factors_inter,
                    relation_encoder=self.relation_encoder,
                    relation_scale_factor=self.relation_scale_factor,
                    return_element_wise_scores=return_element_wise_scores,
                )

        return (
            intra_sim_scores,
            inter_sim_scores,
            intra_sim_elementwise_scores,
            inter_sim_elementwise_scores,
            intra_sim_selected_d_weights,
        )

    def multi_granularity_interaction(
        self, *args, **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        raise NotImplementedError("TODO: Implement multi-granularity interaction")

    def compute_outer_sim(
        self,
        q_vecs: torch.Tensor,
        q_weights: torch.Tensor,
        q_scatter_indices: torch.Tensor,
        q_scale_factors: torch.Tensor,
        d_vecs: torch.Tensor,
        d_weights: torch.Tensor,
        return_element_wise_scores: bool = False,
        use_sum_instead_of_max: bool = False,
    ) -> torch.Tensor:
        """Compute vector similarity in token-level, and then aggregate to phrase-level"""

        # Compute maxsim
        max_q_scores, element_wise_scores, selected_d_weights = self.compute_maxsim(
            q_vecs=q_vecs,
            q_weights=q_weights,
            d_vecs=d_vecs,
            d_weights=d_weights,
            return_element_wise_scores=return_element_wise_scores,
            use_sum_instead_of_max=use_sum_instead_of_max,
        )

        # Aggregate token-level vector similarity to phrase-level
        if q_scatter_indices is None:
            max_q_aggregated = max_q_scores
        else:
            max_q_aggregated = aggregate_vectors_with_indices(
                src_tensor=max_q_scores,
                scatter_indices=q_scatter_indices,
                reduce=self.reduce_strategy,
            )

        # Aggregate phrase-level scores to query-level scores
        q_scores = max_q_aggregated.sum(dim=1)

        # Scale the query scores
        q_scores = q_scores * q_scale_factors

        return q_scores, element_wise_scores, selected_d_weights

    def compute_inner_sim(
        self,
        q_vecs: torch.Tensor,
        q_weights: torch.Tensor,
        q_scale_factors: torch.Tensor,
        d_vecs: torch.Tensor,
        d_weights: torch.Tensor,
        return_element_wise_scores: bool = False,
        use_sum_instead_of_max: bool = False,
    ) -> torch.Tensor:
        """Compute vector similarity in phrase-level"""

        # Compute maxsim
        max_q_scores, element_wise_scores, selected_d_weights = self.compute_maxsim(
            q_vecs=q_vecs,
            q_weights=q_weights,
            d_vecs=d_vecs,
            d_weights=d_weights,
            return_element_wise_scores=return_element_wise_scores,
            use_sum_instead_of_max=use_sum_instead_of_max,
        )

        # Aggregate phrase-level scores to query-level scores
        q_scores = max_q_scores.sum(dim=1)

        # Scale the query scores
        q_scores = q_scores * q_scale_factors

        return q_scores, element_wise_scores, selected_d_weights

    def compute_maxsim(
        self,
        q_vecs: torch.Tensor,
        d_vecs: torch.Tensor,
        q_weights: torch.Tensor = None,
        d_weights: torch.Tensor = None,
        return_element_wise_scores: bool = False,
        use_sum_instead_of_max: bool = False,
    ) -> torch.Tensor:
        if return_element_wise_scores:
            # Compute similarity scores for each q vectors and d vectors
            element_wise_scores = d_vecs @ q_vecs.transpose(-2, -1)
            # Apply weights
            element_wise_scores_original = element_wise_scores.clone()
            if q_weights is not None:
                element_wise_scores = element_wise_scores * q_weights.transpose(1, 2)
            if d_weights is not None:
                element_wise_scores = element_wise_scores * d_weights
            # Find the maximum similarity scores for each query vectors
            if use_sum_instead_of_max:
                max_scores_info = element_wise_scores.sum(dim=1)
            else:
                max_scores_info = element_wise_scores.max(dim=1)
            max_q_scores = max_scores_info.values
            max_q_indices = max_scores_info.indices
            # Select the corresponding weights from d_weights
            if d_weights is None:
                selected_d_weights = None
            else:
                # Ensure correct advanced indexing
                batch_size, num_queries = max_q_indices.size()
                arange_batch = torch.arange(
                    batch_size, device=max_q_indices.device
                ).unsqueeze(1)
                selected_d_weights = d_weights[arange_batch, max_q_indices]
        else:
            # Apply d weights
            if d_weights is not None:
                d_vecs = d_vecs * d_weights
            # Compute similarity scores for each q vectors and d vectors
            element_wise_scores = d_vecs @ q_vecs.transpose(-2, -1)
            if use_sum_instead_of_max:
                max_q_scores = element_wise_scores.sum(dim=1)
            else:
                max_q_scores = element_wise_scores.max(dim=1).values
            # Apply q weights
            if q_weights is not None:
                max_q_scores = max_q_scores * q_weights.squeeze()
            # Dummy return values
            element_wise_scores_original = None
            selected_d_weights = None

        return max_q_scores, element_wise_scores_original, selected_d_weights

    def encode_documents(
        self,
        documents: List[Document],
        bsize: int = 512,
        show_progress: bool = False,
        truncation=False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.inference_mode():
            # Configs
            device = self.llm.device

            # Tokenize the documents
            result = self.tokenizers.d_tokenizer.tokenize_batch(
                documents,
                truncation=truncation,
                padding=True,
                return_tensors="pt",
            )
            ids, att_mask = result["input_ids"], result["attention_mask"]

            # Create mask
            tok_mask = get_mask(
                input_ids=ids, skip_ids=self.tokenizers.skip_tok_ids
            ).bool()

            # Save the original order
            all_tok_ids = ids
            all_tok_mask = tok_mask

            # Sort by length
            ids, att_mask, indices, reverse_indices = _sort_by_length(
                ids, att_mask, descending=True
            )
            tok_mask = tok_mask[indices]

            # Encode the documents
            all_tok_embs: List[torch.Tensor] = []
            for input_ids, attention_mask, token_mask, _ in tqdm.tqdm(
                _split_into_batches(ids, att_mask, tok_mask, bsize=bsize),
                disable=not show_progress,
            ):
                # Assumption: Attention mask is applied for every token in the input_ids (from left-to-right)
                local_max_tok_len = attention_mask.sum(1).max().item()
                # Sample the input_ids and attention_mask
                input_ids = input_ids[:, :local_max_tok_len]
                attention_mask = attention_mask[:, :local_max_tok_len]
                token_mask = token_mask[:, :local_max_tok_len]
                # TODO: Need to change the below codes
                tmp_result = self.encode_d_text(
                    tok_ids=input_ids.to(device),
                    att_mask=attention_mask.to(device),
                    tok_mask=token_mask.to(device),
                    is_encoding=True,
                )
                tok_embs = tmp_result[0]
                all_tok_embs.append(tok_embs.cpu().half())

            # Concatenate by padding the embeddings
            # Here, All_tok_embs shape was (bsize, n_tok, dim)
            all_tok_embs = list_utils.do_flatten_list(
                [t.unbind() for t in all_tok_embs]
            )
            all_tok_embs = pad_sequence(all_tok_embs, batch_first=True, padding_value=0)

            # Convert back to the original order
            all_tok_embs = all_tok_embs[reverse_indices]

        return all_tok_ids, all_tok_embs, all_tok_mask
