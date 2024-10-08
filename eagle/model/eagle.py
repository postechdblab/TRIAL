import logging
from typing import *

import hkkang_utils.list as list_utils
import torch
from omegaconf import DictConfig
from torch.nn.utils.rnn import pad_sequence

from eagle.model.base_model import BaseModel
from eagle.model.compiled_tensor_op import l1_regularization, l2_regularization
from eagle.model.objective import (
    compute_loss,
    doc_indices_for_ib_loss,
    get_target_scale_tensor,
)
from eagle.model.utils import aggregate_vectors_with_indices, get_weight_layer
from eagle.tokenization import Tokenizers

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

        # TODO: Move the post processing to the base class
        self.load_checkpoint()

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

        intra_scores, inter_scores = self.multi_granularity_interaction(
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
            num_valid = self.get_valid_num(q_tok_mask)
            q_tok_weight = q_tok_weight.masked_fill(q_tok_mask.unsqueeze(-1) == 0, 0)
            q_weight_ratio = q_tok_weight.sum() / num_valid.sum()
            q_weight_var = q_tok_weight[q_tok_mask == 1].var()
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
            d_weight_intra_var = d_weight_intra_masked[d_mask_intra == 1].var()
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
                d_weight_inter_var = d_weight_inter_masked[d_mask_inter == 1].var()

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
        return return_dict

    def encode_text(
        self,
        tok_ids: torch.Tensor,
        att_mask: torch.Tensor,
        phrase_scatter_indices: torch.Tensor = None,
        sent_start_indices: List[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # LLM encoding
        encoded_tok_vectors = self.llm(
            tok_ids, attention_mask=att_mask
        ).last_hidden_state

        # Perform lower-dimension projection in token-level
        projected_tok_vectors = self.tok_projection_layer(encoded_tok_vectors)

        # Encode phrase-level embeddings
        if self.use_attn_for_phrase_encoding:
            # TODO: Make different phrases into a batch
            # TODO: Add positional encoding
            encoded_phrase_vectors = None
            projected_phrase_vectors = None
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

        # Get sentence-level embeddings
        projected_sent_vectors = []
        for b_idx, start_indices in enumerate(sent_start_indices):
            projected_sent_vectors.append(projected_tok_vectors[b_idx, start_indices])
        projected_sent_vectors = pad_sequence(projected_sent_vectors, batch_first=True)

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
        dtype = projected_tok_vectors.dtype

        # Mask paddings
        projected_tok_vectors.masked_fill_(tok_mask.unsqueeze(-1) == 0, 0)
        projected_phrase_vectors.masked_fill_(phrase_mask.unsqueeze(-1) == 0, 0)
        projected_sent_vectors.masked_fill_(sent_mask.unsqueeze(-1) == 0, 0)

        # Weights
        tok_weights = None
        phrase_weights = None
        sentence_weights = None
        if self.is_use_q_weight:
            tok_weights = self.q_weight_layer(projected_tok_vectors)
            if projected_phrase_vectors is not None:
                phrase_weights = self.q_weight_layer(projected_phrase_vectors)
            if projected_sent_vectors is not None:
                sentence_weights = self.q_weight_layer(projected_sent_vectors)

        # Compute normalization scale for each query
        token_scale_factor = self.get_scale_factor(mask=tok_mask)

        sentence_scale_factor = None
        if projected_sent_vectors is not None:
            sentence_scale_factor = torch.full(
                size=(projected_sent_vectors.shape[:-1]),
                fill_value=self.q_maxlen,
                dtype=projected_sent_vectors.dtype,
                device=tok_ids.device,
            )

        phrase_scale_factor = None
        if projected_phrase_vectors is not None:
            phrase_scale_factor = self.get_scale_factor(mask=phrase_mask)

        # Normalize
        projected_tok_vectors = torch.nn.functional.normalize(
            projected_tok_vectors, p=2, dim=2
        )
        if projected_tok_vectors.dtype != dtype:
            projected_tok_vectors = projected_tok_vectors.to(dtype)
        if projected_sent_vectors is not None:
            projected_sent_vectors = torch.nn.functional.normalize(
                projected_sent_vectors, p=2, dim=2
            )
            if projected_sent_vectors.dtype != dtype:
                projected_sent_vectors = projected_sent_vectors.to(dtype)
        if projected_phrase_vectors is not None:
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
        phrase_mask: torch.Tensor,
        sent_mask: torch.Tensor,
        scatter_indices: torch.Tensor = None,
        sent_start_indices: List[List[List[int]]] = None,
        q_vectors: torch.Tensor = None,
        q_mask: torch.Tensor = None,
        nway: int = None,
        is_encoding: bool = False,
        is_inference: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Configs
        if len(tok_ids.shape) == 3:
            bsize, ndoc, max_tok_len = tok_ids.shape
            _, _, max_phrase_len = phrase_mask.shape
            _, _, max_sent_len = sent_mask.shape
            nhard = nway // bsize
            tok_ids_combined = tok_ids.view(-1, max_tok_len)
            att_mask_combined = att_mask.view(-1, max_tok_len)
            tok_mask_combined = tok_mask.view(-1, max_tok_len)
            phrase_mask_combined = phrase_mask.view(-1, max_phrase_len)
            sent_mask_combined = sent_mask.view(-1, max_sent_len)
            sent_start_indices = list_utils.do_flatten_list(sent_start_indices)
        elif len(tok_ids.shape) == 2:
            bsize, max_len = tok_ids.shape
            nhard = 1
            tok_ids_combined = tok_ids
            att_mask_combined = att_mask
            tok_mask_combined = tok_mask
            phrase_mask_combined = phrase_mask
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

        dtype = projected_tok_vectors.dtype

        # Apply mask
        projected_tok_vectors = projected_tok_vectors.masked_fill(
            tok_mask_combined.unsqueeze(-1) == 0, 0
        )
        projected_phrase_vectors = projected_phrase_vectors.masked_fill(
            phrase_mask_combined.unsqueeze(-1) == 0, 0
        )
        projected_sent_vectors = projected_sent_vectors.masked_fill(
            sent_mask_combined.unsqueeze(-1) == 0, 0
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
            if projected_phrase_vectors is not None:
                phrase_weights_intra = aggregate_vectors_with_indices(
                    src_tensor=tok_weights_intra,
                    scatter_indices=scatter_indices,
                    reduce=self.reduce_strategy,
                )
            # Get sentence-level weights
            if projected_sent_vectors is not None:
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
                if projected_phrase_vectors is not None:
                    selected_scatter_indices = scatter_indices[doc_indices]
                    phrase_weights_inter = aggregate_vectors_with_indices(
                        src_tensor=tok_weights_inter,
                        scatter_indices=selected_scatter_indices,
                        reduce=self.reduce_strategy,
                    )
                if projected_sent_vectors is not None:
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
        if projected_sent_vectors is not None:
            projected_sent_vectors = torch.nn.functional.normalize(
                projected_sent_vectors, p=2, dim=2
            )
            if projected_sent_vectors.dtype != dtype:
                projected_sent_vectors = projected_sent_vectors.to(dtype)
        if projected_phrase_vectors is not None:
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

    def get_valid_num(self, mask: torch.Tensor) -> torch.Tensor:
        """Get the number of valid tokens for each query
        :param mask with 1 as valid and 0 as non-valid token (Shape: [bsize, num_toks])
        :type mask: torch.Tensor
        :return: num_valid_tokens Shape: [bsize]
        :rtype: torch.Tensor
        """
        num_valid_tokens = mask.sum(dim=1)
        num_non_valid_tokens = mask.shape[1] - num_valid_tokens
        target_scale = get_target_scale_tensor(
            target_scale=mask.shape[1],
            b_size=num_non_valid_tokens.shape[0],
            device=num_non_valid_tokens.device,
            dtype=num_non_valid_tokens.dtype,
        )
        num_valid_tokens = target_scale - num_non_valid_tokens
        return num_valid_tokens

    def get_scale_factor(self, mask: torch.Tensor) -> torch.Tensor:
        """Get the scale factor for normalization
        :param mask: Shape: [bsize, num_toks]
        :type mask: torch.Tensor
        :return: scale factor Shape: [bsize]
        :rtype: torch.Tensor
        """
        num_valid_tokens = self.get_valid_num(mask)
        return self.q_maxlen / num_valid_tokens

    def multi_granularity_interaction(
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
        d_sent_weight_intra: torch.Tensor,
        d_sent_weight_inter: torch.Tensor,
        q_scatter_indices: torch.Tensor,
        q_tok_scale_factor: Optional[torch.Tensor] = None,
        q_phrase_scale_factor: Optional[torch.Tensor] = None,
        q_sent_scale_factor: Optional[torch.Tensor] = None,
        is_inference: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Configs
        bsize, q_tok_len, q_dim = q_tok.shape
        bsize, q_phrase_len, q_dim = q_phrase.shape
        _, sent_inter_max_len, _ = d_sent_weight_inter.shape
        nway = d_tok.shape[0] // bsize
        ib_nhard = nway // bsize
        repeat_num_for_inter = ib_nhard * (bsize - 1) + 1
        d_inter_indices: torch.Tensor = doc_indices_for_ib_loss(
            bsize,
            nway,
            ib_nhard,
            return_as_tensor=True,
            device=d_sent.device,
        )

        # Combine document vectors and weights for intra and inter comparison
        d_vecs_intra = torch.cat([d_sent, d_phrase, d_tok], dim=1)
        d_weights_intra = torch.cat(
            [d_sent_weight_intra, d_phrase_weight_intra, d_tok_weight_intra], dim=1
        )
        d_weights_inter = None
        if not is_inference:
            d_vecs_inter = torch.cat(
                [d_sent[:, :sent_inter_max_len, :], d_phrase, d_tok], dim=1
            )
            d_weights_inter = torch.cat(
                [d_sent_weight_inter, d_phrase_weight_inter, d_tok_weight_inter], dim=1
            )

        # Compute similarity scores
        assert self.sim_type in [
            "combination",
            "outer_agg",
            "inner_agg",
        ], f"Invalid sim_type: {self.sim_type}"
        if self.sim_type in ["combination", "outer_agg"]:
            # Repeat the query vectors for intra comparison
            q_vecs_intra = q_tok.repeat_interleave(nway, dim=0)
            q_weight_intra = q_tok_weight.repeat_interleave(nway, dim=0)
            q_scatter_indices_intra = q_scatter_indices.repeat_interleave(nway, dim=0)
            q_scale_factors_intra = q_tok_scale_factor.repeat_interleave(nway, dim=0)

            # Repeat the query vectors for inter comparison
            if not is_inference:
                q_vecs_inter = q_tok.repeat_interleave(repeat_num_for_inter, dim=0)
                q_weight_inter = q_tok_weight.repeat_interleave(
                    repeat_num_for_inter, dim=0
                )
                q_scatter_indices_inter = q_scatter_indices.repeat_interleave(
                    repeat_num_for_inter, dim=0
                )
                q_scale_factors_inter = q_tok_scale_factor.repeat_interleave(
                    repeat_num_for_inter, dim=0
                )
                selected_d_vecs_inter: torch.Tensor = d_vecs_inter[d_inter_indices]

            # Compute intra query scores through outer aggregation (i.e., token-level vector similarity)
            intra_sim_scores_outer = self.compute_outer_sim(
                q_vecs=q_vecs_intra,
                q_weights=q_weight_intra,
                q_scatter_indices=q_scatter_indices_intra,
                q_scale_factors=q_scale_factors_intra,
                d_vecs=d_vecs_intra,
                d_weights=d_weights_intra,
            )

            # Compute inter query scores through outer aggregation (i.e., token-level vector similarity)
            if not is_inference:
                inter_sim_scores_outer = self.compute_outer_sim(
                    q_vecs=q_vecs_inter,
                    q_weights=q_weight_inter,
                    q_scatter_indices=q_scatter_indices_inter,
                    q_scale_factors=q_scale_factors_inter,
                    d_vecs=selected_d_vecs_inter,
                    d_weights=d_weights_inter,
                )

        if self.sim_type in ["combination", "inner_agg"]:
            # Repeat the query vectors for intra comparison
            q_vecs_intra = q_phrase.repeat_interleave(nway, dim=0)
            q_weight_intra = q_phrase_weight.repeat_interleave(nway, dim=0)
            q_scale_factors_intra = q_phrase_scale_factor.repeat_interleave(nway, dim=0)

            # Repeat the query vectors for inter comparison
            if not is_inference:
                q_vecs_inter = q_phrase.repeat_interleave(repeat_num_for_inter, dim=0)
                q_weight_inter = q_phrase_weight.repeat_interleave(
                    repeat_num_for_inter, dim=0
                )
                q_scale_factors_inter = q_phrase_scale_factor.repeat_interleave(
                    repeat_num_for_inter, dim=0
                )
                selected_d_vecs_inter: torch.Tensor = d_vecs_inter[d_inter_indices]

            # Compute intra query scores through inner aggregation (i.e., phrase-level vector similarity)
            intra_sim_scores_inner = self.compute_inner_sim(
                q_vecs=q_vecs_intra,
                q_weights=q_weight_intra,
                q_scale_factors=q_scale_factors_intra,
                d_vecs=d_vecs_intra,
                d_weights=d_weights_intra,
            )

            # Compute inter query scores through inner aggregation (i.e., phrase-level vector similarity)
            if not is_inference:
                inter_sim_scores_inner = self.compute_inner_sim(
                    q_vecs=q_vecs_inter,
                    q_weights=q_weight_inter,
                    q_scale_factors=q_scale_factors_inter,
                    d_vecs=selected_d_vecs_inter,
                    d_weights=d_weights_inter,
                )

        # Compute the final scores
        inter_sim_scores = None
        if self.sim_type == "combination":
            intra_sim_scores = torch.max(intra_sim_scores_outer, intra_sim_scores_inner)
            if not is_inference:
                inter_sim_scores = torch.max(
                    inter_sim_scores_outer, inter_sim_scores_inner
                )
        elif self.sim_type == "outer_agg":
            intra_sim_scores = intra_sim_scores_outer
            if not is_inference:
                inter_sim_scores = inter_sim_scores_outer
        else:
            intra_sim_scores = intra_sim_scores_inner
            if not is_inference:
                inter_sim_scores = inter_sim_scores_inner

        return intra_sim_scores, inter_sim_scores

    def compute_outer_sim(
        self,
        q_vecs: torch.Tensor,
        q_weights: torch.Tensor,
        q_scatter_indices: torch.Tensor,
        q_scale_factors: torch.Tensor,
        d_vecs: torch.Tensor,
        d_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Compute vector similarity in token-level, and then aggregate to phrase-level"""

        # Compute maxsim
        max_q_scores = self.compute_maxsim(
            q_vecs=q_vecs,
            q_weights=q_weights,
            d_vecs=d_vecs,
            d_weights=d_weights,
        )

        # Aggregate token-level vector similarity to phrase-level
        max_q_aggregated = aggregate_vectors_with_indices(
            src_tensor=max_q_scores,
            scatter_indices=q_scatter_indices,
            reduce=self.reduce_strategy,
        )

        # Aggregate phrase-level scores to query-level scores
        q_scores = max_q_aggregated.sum(dim=1)

        # Scale the query scores
        q_scores = q_scores * q_scale_factors

        return q_scores

    def compute_inner_sim(
        self,
        q_vecs: torch.Tensor,
        q_weights: torch.Tensor,
        q_scale_factors: torch.Tensor,
        d_vecs: torch.Tensor,
        d_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Compute vector similarity in phrase-level"""

        # Compute maxsim
        max_q_scores = self.compute_maxsim(
            q_vecs=q_vecs,
            q_weights=q_weights,
            d_vecs=d_vecs,
            d_weights=d_weights,
        )

        # Aggregate phrase-level scores to query-level scores
        q_scores = max_q_scores.sum(dim=1)

        # Scale the query scores
        q_scores = q_scores * q_scale_factors

        return q_scores

    def compute_maxsim(
        self,
        q_vecs: torch.Tensor,
        d_vecs: torch.Tensor,
        q_weights: torch.Tensor = None,
        d_weights: torch.Tensor = None,
    ) -> torch.Tensor:
        # Apply d weights
        if d_weights is not None:
            d_vecs = d_vecs * d_weights
        # Compute similarity scores for each q vectors and d vectors
        element_wise_scores = d_vecs @ q_vecs.transpose(-2, -1)
        max_q_scores = element_wise_scores.max(dim=1).values
        # Apply q weights
        if q_weights is not None:
            max_q_scores = max_q_scores * q_weights.squeeze()
        return max_q_scores
