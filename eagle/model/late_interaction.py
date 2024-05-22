import logging
from typing import *

import torch
from accelerate import Accelerator
from omegaconf import DictConfig
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModel, BitsAndBytesConfig

from eagle.model.compiled_tensor_op import l1_regularization, l2_regularization
from eagle.model.objective import (compute_fine_grained_loss, compute_loss,
                                   doc_indices_for_ib_loss,
                                   get_target_scale_tensor)
from eagle.model.utils import (get_vectors_from_ranges, get_weight_layer,
                               modify_execution_device, modify_grad)
from eagle.search.algorithm import compute_sum_maxsim
from eagle.tokenizer import NewTokenizer
from eagle.utils import handle_old_ckpt

logger = logging.getLogger("NewModel")


class NewModel(torch.nn.Module):
    def __init__(
        self, cfg: DictConfig, q_tokenizer: NewTokenizer, d_tokenizer: NewTokenizer
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.q_special_tok_ids = q_tokenizer.special_toks_ids
        self.d_special_tok_ids = d_tokenizer.special_toks_ids
        self.punct_tok_ids = q_tokenizer.punctuations
        self.q_maxlen = q_tokenizer.cfg.max_len
        # Configs
        self.nway = cfg.nway
        # Ideas
        self.is_use_q_weight = cfg.is_use_q_weight
        self.is_use_d_weight = cfg.is_use_d_weight
        self.is_d_independent = cfg.is_d_independent
        self.w_regularize_strategy = cfg.w_regularize_strategy
        self.q_weight_strategy = cfg.q_weight_strategy
        self.d_weight_strategy = cfg.d_weight_strategy
        self.q_weight_coeff = cfg.q_weight_coeff
        self.d_weight_coeff = cfg.d_weight_coeff
        self.intra_loss_coeff = cfg.intra_loss_coeff
        self.inter_loss_coeff = cfg.inter_loss_coeff
        self.fine_grained_loss_coeff = handle_old_ckpt(cfg, "fine_grained_loss_coeff")
        self.is_use_dynamic_granularity_coeff = handle_old_ckpt(
            cfg, "is_use_dynamic_granularity_coeff"
        )
        self.is_only_phrase_score = handle_old_ckpt(cfg, "is_only_phrase_score")
        # About quantization
        self.is_use_lora = handle_old_ckpt(cfg, "is_use_lora")
        self.is_use_quantization = handle_old_ckpt(cfg, "is_use_quantization")
        self.quantization_bit = handle_old_ckpt(cfg, "quantization_bit")
        self.granularity_level = handle_old_ckpt(cfg, "granularity_level")
        # Backbone model (The attribute name should be llm to be compatible with the optimizer in LightningModule)
        self.llm = self.__create_backbone_model(
            name=cfg.name, vocab_num=len(q_tokenizer)
        )

        # Projection layers
        self.tok_projection_layer = torch.nn.Linear(
            self.llm.config.hidden_size, cfg.out_dim
        )
        self.cls_projection_layer = self.__create_linear_layers_for_multi_granularity(
            input_dim=self.llm.config.hidden_size,
            intermediate_dim=self.llm.config.hidden_size // 2,
            out_dim=cfg.out_dim,
        )
        self.phrase_projection_layer = (
            self.__create_linear_layers_for_multi_granularity(
                input_dim=self.llm.config.hidden_size,
                intermediate_dim=self.llm.config.hidden_size // 2,
                out_dim=cfg.out_dim,
                force=self.is_only_phrase_score,
            )
        )
        self.score_granularity_coeff_layer = (
            torch.nn.Sequential(
                torch.nn.Linear(self.llm.config.hidden_size, 3), torch.nn.Softmax(dim=1)
            )
            if self.is_use_dynamic_granularity_coeff
            else None
        )

        # Pooling for phrase level embeddings
        self.reduce_strategy = cfg.reduce_strategy

        # Layers to predict the weights (i.e., importance)
        self.q_weight_layer = self.__create_q_weight_layer(
            input_dim=self.llm.config.hidden_size,
            intermediate_dim=cfg.out_dim,
        )
        self.d_weight_layer = self.__create_d_weight_layer(
            input_dim=self.llm.config.hidden_size,
            intermediate_dim=cfg.out_dim,
        )
        self.d_weight_layer_norm = torch.nn.LayerNorm(self.llm.config.hidden_size) if self.is_use_d_weight else None
        
        # Cross-attention layer for interacting query and documents
        self.cross_att_layer = self.__create_cross_att_layer()

        # Regularization for the weights (i.e., importance)
        self.regularization = self.__create_regularization_func(
            strategy=cfg.w_regularize_strategy
        )

    @property
    def q_skiplist(self) -> List[int]:
        return self.q_special_tok_ids

    @property
    def d_skiplist(self) -> List[int]:
        return self.d_special_tok_ids
        # return self.d_special_tok_ids + self.punct_tok_ids

    @property
    def is_use_multi_granularity(self) -> bool:
        return (
            self.granularity_level in ["word", "phrase"]
            and not self.is_only_phrase_score
        )

    def __create_backbone_model(self, name: str, vocab_num: int) -> torch.nn.Module:
        # Load the BERT-base-uncased model configuration
        quantization_config = None
        if self.is_use_quantization:
            if self.quantization_bit == 4:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=False,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            elif self.quantization_bit == 8:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            else:
                raise ValueError(
                    f"Unsupported quantization bit: {self.quantization_bit}"
                )

        # Load model and resize the token embeddings
        # model = AutoModel.from_pretrained(name, quantization_config=quantization_config, device_map=device_map)
        model = AutoModel.from_pretrained(
            name,
            quantization_config=quantization_config,
            device_map=torch.device("cpu"),
        )
        model.resize_token_embeddings(vocab_num)
        if "bert-" in name:
            model.pooler = None
        if "t5" in name:
            model.decoder = None
            model = model.encoder

        # Prepare model for kbit training with lora
        if self.is_use_quantization and self.is_use_lora:
            for name, param in model.named_parameters():
                # freeze base model's layers
                param.requires_grad = False

        # Add lora layer
        if self.is_use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                target_modules=["query", "value", "dense"],
                r=8,
                lora_alpha=32,
                lora_dropout=0.05,
            )
            model = get_peft_model(model, lora_config)
        return model

    def __create_cross_att_layer(self) -> torch.nn.Module:
        if self.is_use_d_weight and not self.is_d_independent:
            return torch.nn.MultiheadAttention(
                embed_dim=self.llm.config.hidden_size,
                num_heads=self.llm.config.num_attention_heads,
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

    def __create_linear_layers_for_multi_granularity(
        self, input_dim: int, intermediate_dim: int, out_dim: int, force: bool = False
    ) -> torch.nn.Module:
        if self.is_use_multi_granularity or force:
            return torch.nn.Linear(input_dim, out_dim)
        return None

    def __create_regularization_func(
        self, strategy: str
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        if strategy == "l1":
            return l1_regularization
        elif strategy == "l2":
            return l2_regularization
        raise ValueError(f"Unsupported regularization strategy: {strategy}")

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
        distil_scores: Optional[torch.Tensor] = None,
        is_inference: Optional[bool] = False,
        is_analyze: Optional[bool] = False,
        **kwargs,
    ) -> Dict[str, Any]:
        # To set right gpu device for DDP training using quantization
        if self.is_use_quantization:
            process_idx = Accelerator().process_index
            if self.llm._hf_hook.execution_device != process_idx:
                print(
                    f"Device changed from {self.llm._hf_hook.execution_device} to {process_idx}"
                )
                modify_execution_device(self, process_idx)
                self.cuda(torch.device(f"cuda:{process_idx}"))

        # Configs
        bsize, nway, dim = doc_tok_ids.shape
        ib_nhard = nway // bsize
        is_eval = labels is not None
        is_use_fine_grained_loss = fine_grained_label is not None
        assert (
            q_tok_ids.shape[0] == bsize
        ), f"Batch size is not consistent: {q_tok_ids.shape[0]} vs {bsize}"
        # Encode
        (
            q_encoded,
            q_cls_projected,
            q_tok_projected,
            q_phrase_projected,
            q_tok_weight,
            q_phrase_weight,
            q_cls_scale_factor,
            q_tok_scale_factor,
            q_phrase_scale_factor,
        ) = self.encode_q_text(
            tok_ids=q_tok_ids,
            att_mask=q_tok_att_mask,
            tok_mask=q_tok_mask,
            phrase_mask=q_phrase_mask,
            scatter_indices=q_scatter_indices,
        )
        (
            d_cls_projected,
            d_tok_projected,
            d_phrase_projected,
            d_weight_intra,
            d_weight_inter,
        ) = self.encode_d_text(
            tok_ids=doc_tok_ids,
            att_mask=doc_tok_att_mask,
            tok_mask=doc_tok_mask,
            phrase_mask=doc_phrase_mask,
            scatter_indices=doc_scatter_indices,
            q_vectors=q_encoded,
            q_mask=q_tok_mask,
            nway=nway,
            is_eval=is_eval,
        )
        dtype = q_tok_projected.dtype

        # Compute coefficient for each granularity
        if self.is_use_dynamic_granularity_coeff:
            coeff = self.score_granularity_coeff_layer(q_encoded[:, 0]).unsqueeze(1)
            # coeff = coeff.to(dtype)
            cls_coeff, tok_coeff, phrase_coeff = (
                coeff[:, :, 0],
                coeff[:, :, 1],
                coeff[:, :, 2],
            )
            # Apply coefficient by changing the scale factor
            q_cls_scale_factor = q_cls_scale_factor * cls_coeff
            q_tok_scale_factor = q_tok_scale_factor * tok_coeff
            q_phrase_scale_factor = q_phrase_scale_factor * phrase_coeff
        elif self.is_only_phrase_score:
            cls_coeff = tok_coeff = phrase_coeff = None
        else:
            cls_coeff = tok_coeff = phrase_coeff = None
        # Compute scores with cls
        if q_cls_projected is not None:
            (
                cls_intra_scores,
                cls_inter_scores,
                cls_intra_q_max_scores,
                cls_intra_qd_scores,
            ) = self.compute_scores(
                q_encoded=q_cls_projected,
                d_encoded=d_cls_projected,
                q_weight=None,
                q_scale_factor=q_cls_scale_factor,
                q_mask=None,
                d_weight_intra=d_weight_intra,
                d_weight_inter=d_weight_inter,
                d_mask=None,
                nway=nway,
                ib_nhard=ib_nhard,
                return_max_scores=is_use_fine_grained_loss,
                return_entire_scores=is_analyze,
            )
        # Compute scores with token
        if not self.is_only_phrase_score:
            (
                tok_intra_scores,
                tok_inter_scores,
                tok_intra_q_max_scores,
                tok_intra_qd_scores,
            ) = self.compute_scores(
                q_encoded=q_tok_projected,
                q_weight=q_tok_weight,
                q_scale_factor=q_tok_scale_factor,
                q_mask=q_tok_mask,
                d_encoded=d_tok_projected,
                d_weight_intra=d_weight_intra,
                d_weight_inter=d_weight_inter,
                d_mask=doc_tok_mask,
                nway=nway,
                ib_nhard=ib_nhard,
                return_max_scores=is_use_fine_grained_loss,
                return_entire_scores=is_analyze,
            )
        # Compute scores with phrase
        if q_phrase_projected is not None:
            (
                phrase_intra_scores,
                phrase_inter_scores,
                phrase_intra_q_max_scores,
                phrase_intra_qd_scores,
            ) = self.compute_scores(
                q_encoded=q_phrase_projected,
                q_weight=q_phrase_weight,
                q_scale_factor=q_phrase_scale_factor,
                q_mask=q_phrase_mask,
                d_encoded=d_phrase_projected,
                d_weight_intra=d_weight_intra,
                d_weight_inter=d_weight_inter,
                d_mask=doc_phrase_mask,
                nway=nway,
                ib_nhard=ib_nhard,
                return_max_scores=is_use_fine_grained_loss,
                return_entire_scores=is_analyze,
            )

        # Sum scores across different granularity
        if self.is_only_phrase_score:
            intra_scores = 0
            inter_scores = 0
        else:
            intra_scores = tok_intra_scores
            inter_scores = tok_inter_scores
        if q_cls_projected is not None:
            intra_scores = intra_scores + cls_intra_scores
            inter_scores = inter_scores + cls_inter_scores
        if q_phrase_projected is not None:
            intra_scores = intra_scores + phrase_intra_scores
            inter_scores = inter_scores + phrase_inter_scores

        # Compute loss
        device = intra_scores.device
        loss, intra_loss, inter_loss = compute_loss(
            scores=intra_scores,
            ib_scores=inter_scores,
            bsize=bsize,
            nway=nway,
            ib_nhard=ib_nhard,
            device=device,
            intra_loss_coeff=self.intra_loss_coeff,
            inter_loss_coeff=self.inter_loss_coeff,
        )

        # Additional loss term
        fine_grained_loss = 0
        if is_use_fine_grained_loss:
            tok_intra_q_max_scores = tok_intra_q_max_scores.view(bsize, nway, -1)[
                :, 1:, :
            ].reshape(bsize * (nway - 1), -1)
            tok_intra_q_max_scores = tok_intra_q_max_scores[
                fine_grained_label_mask
            ]  # Filter those with no false negative?
            # Create a new tensor to avoid preventing flow of the gradient in the original tensor
            if tok_intra_q_max_scores.requires_grad:
                tok_intra_q_max_scores = tok_intra_q_max_scores.clone().float()
                # Do not flow the gradient to the false negative tokens (so that the score does not increase for the negative documents)
                modify_grad(tok_intra_q_max_scores, (fine_grained_label != 0))
            fine_grained_loss = compute_fine_grained_loss(
                scores=tok_intra_q_max_scores, labels=fine_grained_label
            )
            # loss = loss + ((self.fine_grained_loss_coeff / bsize) * fine_grained_loss)
            loss = loss + (self.fine_grained_loss_coeff * fine_grained_loss)

        # Compute weight regularization for query
        q_weight_reg_term = 0
        if q_tok_weight is not None:
            q_weight_reg_term = self.regularization(q_tok_weight)
            loss = loss + self.q_weight_coeff * q_weight_reg_term

        # Compute weight regularization for document
        d_weight_reg_term = 0
        if d_weight_intra is not None:
            d_weight_reg_term = self.regularization(d_weight_intra)
            if d_weight_inter is not None:
                d_weight_reg_term = d_weight_reg_term + self.regularization(
                    d_weight_inter
                )
            loss = loss + self.d_weight_coeff * d_weight_reg_term

        # Initialize return dictionary
        return_dict = {
            "loss": loss,
            "intra_loss": intra_loss,
            "inter_loss": inter_loss,
            "fine_grained_loss": fine_grained_loss,
        }

        if self.is_use_dynamic_granularity_coeff:
            return_dict["cls_coeff"] = cls_coeff.mean()
            return_dict["tok_coeff"] = tok_coeff.mean()
            return_dict["phrase_coeff"] = phrase_coeff.mean()

        # Analyze query weights
        q_weight_ratio = 0
        q_weight_var = 0
        if q_weight_reg_term:
            num_valid = self.get_valid_num(q_tok_mask)
            q_tok_weight = q_tok_weight.masked_fill(q_tok_mask, 0)
            q_weight_ratio = q_tok_weight.sum() / num_valid.sum()
            q_weight_var = q_tok_weight[q_tok_mask==0].var()
        # Analyze document weights
        d_weight_intra_ratio = 0
        d_weight_inter_ratio = 0
        d_weight_intra_var = 0
        d_weight_inter_var = 0
        if d_weight_intra is not None:
            d_mask_intra = doc_tok_mask
            d_weight_intra_masked = d_weight_intra * d_mask_intra
            d_weight_intra_ratio = d_weight_intra_masked.sum() / d_mask_intra.sum()
            # Compute variance
            d_weight_intra_var = d_weight_intra_masked[d_mask_intra == 0].var()
            if d_weight_inter is not None:
                d_indices = doc_indices_for_ib_loss(
                    bsize,
                    nway,
                    ib_nhard,
                    return_as_tensor=True,
                    device=doc_tok_mask.device,
                )
                d_mask_inter = doc_tok_mask[d_indices]
                d_weight_inter_masked = d_weight_inter * d_mask_inter
                d_weight_inter_ratio = d_weight_inter_masked.sum() / d_mask_inter.sum()
                d_weight_inter_var = d_weight_inter_masked[d_mask_inter == 0].var()
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
            return_dict["d_weight_intra"] = d_weight_intra
            return_dict["d_weight_inter"] = d_weight_inter
            # Mask
            return_dict["q_mask"] = q_tok_mask
            return_dict["d_mask"] = doc_tok_mask
        if is_analyze:
            return_dict["tok_intra_qd_scores"] = tok_intra_qd_scores
            return_dict["q_tok_weight"] = q_tok_weight
            return_dict["d_tok_weight_intra"] = d_weight_intra
        # Append weights
        if is_eval:
            return return_dict, intra_scores.reshape(bsize, -1)
        return return_dict

    def encode_text(
        self,
        tok_ids: torch.Tensor,
        att_mask: torch.Tensor,
        scatter_indices: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # LLM encoding
        encoded_tok_vectors = self.llm(
            tok_ids, attention_mask=att_mask
        ).last_hidden_state
        if self.is_use_multi_granularity:
            # Embedding to coarse level
            encoded_phrase_vectors = get_vectors_from_ranges(
                tensors=encoded_tok_vectors,
                scatter_indices=scatter_indices,
                reduce=self.reduce_strategy,
            )

            # Perform projection for cls and phrase
            projected_cls_vectors = self.cls_projection_layer(
                encoded_tok_vectors[:, 0:1]
            )
            projected_phrase_vectors = self.phrase_projection_layer(
                encoded_phrase_vectors
            )
        elif self.is_only_phrase_score:
            # Embedding to coarse level
            encoded_phrase_vectors = get_vectors_from_ranges(
                tensors=encoded_tok_vectors,
                scatter_indices=scatter_indices,
                reduce=self.reduce_strategy,
            )
            projected_phrase_vectors = self.phrase_projection_layer(
                encoded_phrase_vectors
            )
            projected_cls_vectors = None
        else:
            # Dummy values
            projected_cls_vectors = projected_phrase_vectors = None

        # Perform projection for token
        projected_tok_vectors = self.tok_projection_layer(encoded_tok_vectors)

        return (
            encoded_tok_vectors,
            projected_cls_vectors,
            projected_tok_vectors,
            projected_phrase_vectors,
        )

    def encode_q_text(
        self,
        tok_ids: torch.Tensor,
        att_mask: torch.Tensor,
        tok_mask: torch.Tensor,
        phrase_mask: torch.Tensor,
        scatter_indices: torch.Tensor,
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
            projected_cls_vectors,
            projected_tok_vectors,
            projected_phrase_vectors,
        ) = self.encode_text(tok_ids, att_mask, scatter_indices)
        dtype = projected_tok_vectors.dtype

        # Mask
        # if not self.is_only_phrase_score:
        #     projected_tok_vectors.masked_fill_(tok_mask, 0)
        # if projected_phrase_vectors is not None:
        #     projected_tok_vectors.masked_fill_(phrase_mask, 0)

        # Weights
        tok_weights = None
        phrase_weights = None
        if self.is_use_q_weight:
            tok_weights = self.q_weight_layer(encoded_tok_vectors)
            if projected_phrase_vectors is not None:
                raise NotImplementedError(
                    "Weights for phrase vectors are not supported yet"
                )

        # Compute normalization scale for each query
        token_scale_factor = self.get_scale_factor(mask=tok_mask)

        cls_scale_factor = None
        if projected_cls_vectors is not None:
            cls_scale_factor = torch.full(
                size=(projected_cls_vectors.shape[:-1]),
                fill_value=self.q_maxlen,
                dtype=projected_cls_vectors.dtype,
                device=tok_ids.device,
            )

        phrase_scale_factor = None
        if projected_phrase_vectors is not None:
            phrase_scale_factor = self.get_scale_factor(mask=phrase_mask)

        # Normalize
        if not self.is_only_phrase_score:
            projected_tok_vectors = torch.nn.functional.normalize(
                projected_tok_vectors, p=2, dim=2
            )
            if projected_tok_vectors.dtype != dtype:
                projected_tok_vectors = projected_tok_vectors.to(dtype)
        if projected_cls_vectors is not None:
            projected_cls_vectors = torch.nn.functional.normalize(
                projected_cls_vectors, p=2, dim=2
            )
            if projected_cls_vectors.dtype != dtype:
                projected_cls_vectors = projected_cls_vectors.to(dtype)
        if projected_phrase_vectors is not None:
            projected_phrase_vectors = torch.nn.functional.normalize(
                projected_phrase_vectors, p=2, dim=2
            )
            if projected_phrase_vectors.dtype != dtype:
                projected_phrase_vectors = projected_phrase_vectors.to(dtype)

        return (
            encoded_tok_vectors,
            projected_cls_vectors,
            projected_tok_vectors,
            projected_phrase_vectors,
            tok_weights,
            phrase_weights,
            cls_scale_factor,
            token_scale_factor,
            phrase_scale_factor,
        )

    def encode_d_text(
        self,
        tok_ids: torch.Tensor,
        att_mask: torch.Tensor,
        tok_mask: torch.Tensor,
        phrase_mask: torch.Tensor,
        scatter_indices: torch.Tensor = None,
        q_vectors: torch.Tensor = None,
        q_mask: torch.Tensor = None,
        nway: int = None,
        is_eval: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Configs
        if len(tok_ids.shape) == 3:
            bsize, ndoc, max_len = tok_ids.shape
            nhard = nway // bsize
            tok_ids_combined = tok_ids.view(-1, max_len)
            att_mask_combined = att_mask.view(-1, max_len)
        elif len(tok_ids.shape) == 2:
            assert (
                self.is_use_d_weight is False
            ), "Document weights are not supported for single document encoding"
            bsize, max_len = tok_ids.shape
            nhard = 1
            tok_ids_combined = tok_ids
            att_mask_combined = att_mask
        (
            encoded_tok_vectors,
            projected_cls_vectors,
            projected_tok_vectors,
            projected_phrase_vectors,
        ) = self.encode_text(tok_ids_combined, att_mask_combined, scatter_indices)

        dtype = projected_tok_vectors.dtype

        # Mask
        # if not self.is_only_phrase_score:
        #     projected_tok_vectors.masked_fill_(tok_mask, 0)
        # if projected_phrase_vectors is not None:
        #     projected_phrase_vectors.masked_fill_(phrase_mask, 0)

        # Create weight using q_vetors
        weights_intra = None
        weights_inter = None
        if self.is_use_d_weight:
            if projected_phrase_vectors is not None:
                raise NotImplementedError(
                    "Weights for phrase vectors are not supported yet"
                )
            if self.is_d_independent:
                weights_intra = self.d_weight_layer(encoded_tok_vectors)
                if not is_eval:
                    doc_indices: List[int] = doc_indices_for_ib_loss(
                        bsize,
                        nway,
                        nhard,
                        return_as_tensor=True,
                        device=encoded_tok_vectors.device,
                    )
                    selected_encoded_tok_vectors = encoded_tok_vectors[doc_indices]
                    weights_inter = self.d_weight_layer(selected_encoded_tok_vectors)
            else:
                # Further encode with q_vectors for intra-example weights
                q_vectors_intra = q_vectors.repeat_interleave(nway, dim=0)
                q_mask_intra = q_mask.repeat_interleave(nway, dim=0)
                # Perform cross-attention
                cross_encoded_tok_vectors_intra, cross_attn_weights_intra = (
                    self.cross_att_layer(
                        encoded_tok_vectors,
                        q_vectors_intra,
                        q_vectors_intra,
                        key_padding_mask=q_mask_intra.squeeze(-1),
                    )
                )
                # Add and normalize
                cross_encoded_tok_vectors_intra = cross_encoded_tok_vectors_intra + encoded_tok_vectors
                cross_encoded_tok_vectors_intra = self.d_weight_layer_norm(cross_encoded_tok_vectors_intra)

                weights_intra = self.d_weight_layer(cross_encoded_tok_vectors_intra)
                if not is_eval:
                    # Further encode with q_vectors for inter-example weights
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
                    selected_encoded_tok_vectors = encoded_tok_vectors[doc_indices]
                    # Perform cross-attention
                    cross_encoded_tok_vectors_inter, cross_attn_weights_inter = (
                        self.cross_att_layer(
                            selected_encoded_tok_vectors,
                            q_vectors_inter,
                            q_vectors_inter,
                            key_padding_mask=q_mask_inter.squeeze(-1),
                        )
                    )
                    # Add and normalize
                    cross_encoded_tok_vectors_inter = cross_encoded_tok_vectors_inter + selected_encoded_tok_vectors
                    cross_encoded_tok_vectors_inter = self.d_weight_layer_norm(cross_encoded_tok_vectors_inter)
                    # Predict the weights
                    weights_inter = self.d_weight_layer(cross_encoded_tok_vectors_inter)

        # Normalize
        if not self.is_only_phrase_score:
            projected_tok_vectors = torch.nn.functional.normalize(
                projected_tok_vectors, p=2, dim=2
            )
            if projected_tok_vectors.dtype != dtype:
                projected_tok_vectors = projected_tok_vectors.to(dtype)
        if projected_cls_vectors is not None:
            projected_cls_vectors = torch.nn.functional.normalize(
                projected_cls_vectors, p=2, dim=2
            )
            if projected_cls_vectors.dtype != dtype:
                projected_cls_vectors = projected_cls_vectors.to(dtype)
        if projected_phrase_vectors is not None:
            projected_phrase_vectors = torch.nn.functional.normalize(
                projected_phrase_vectors, p=2, dim=2
            )
            if projected_phrase_vectors.dtype != dtype:
                projected_phrase_vectors = projected_phrase_vectors.to(dtype)

        return (
            projected_cls_vectors,
            projected_tok_vectors,
            projected_phrase_vectors,
            weights_intra,
            weights_inter,
        )

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsize = q_encoded.shape[0]
        # Apply weights
        if q_weight is not None:
            q_encoded = q_encoded * q_weight

        # Perform Maxsim
        intra_scores, intra_q_max_scores, intra_qd_scores = self.compute_intra_scores(
            q_encoded,
            q_mask=q_mask,
            d_encoded=d_encoded,
            d_weight=d_weight_intra,
            d_mask=d_mask,
            nway=nway,
            return_max_scores=return_max_scores,
            return_entire_scores=return_entire_scores,
        )

        # For optimizing the memory usage
        inter_scores, inter_q_max_scores, inter_qd_scores = self.compute_inter_scores(
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
        target_scale = get_target_scale_tensor(target_scale=mask.shape[1], b_size=num_non_valid_tokens.shape[0], device=num_non_valid_tokens.device, dtype=num_non_valid_tokens.dtype)
        num_valid_tokens = target_scale - num_non_valid_tokens
        return num_valid_tokens

    def get_scale_factor(self, mask: torch.Tensor) -> torch.Tensor:
        num_valid_tokens = self.get_valid_num(mask)
        return self.q_maxlen / num_valid_tokens


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )

    import hydra

    from eagle.tokenizer import NewTokenizer

    @hydra.main(
        version_base=None, config_path="/root/EAGLE/config", config_name="config"
    )
    def main(cfg: DictConfig) -> None:
        # Initialize tokenizer
        q_tokenizer = NewTokenizer(cfg=cfg.q_tokenizer)
        d_tokenizer = NewTokenizer(cfg=cfg.d_tokenizer)
        token_skiplist = list(set(q_tokenizer.skiplist + d_tokenizer.skiplist))
        assert len(q_tokenizer) == len(
            d_tokenizer
        ), f"Tokenizers have different sizes: {len(q_tokenizer)} vs {len(d_tokenizer)}"

        # Load model
        logger.info("Initialize model!")
        model = NewModel(
            cfg=cfg.model,
            token_num=len(q_tokenizer),
            skiplist=token_skiplist,
            punct_list=q_tokenizer.punctuations,
        )

        # Create dummy data
        dummy_bs = 4
        q_text = ["This is a test query"] * dummy_bs
        d_text = ["This is a test document"] * dummy_bs * cfg.model.nway
        # Tokenize
        q_tokens = q_tokenizer(q_text)
        d_tokens = d_tokenizer(d_text)

        loss_dic = model(
            q_tok_ids=q_tokens["input_ids"],
            q_tok_att_mask=q_tokens["attention_mask"],
            doc_tok_ids=d_tokens["input_ids"],
            doc_tok_att_mask=d_tokens["attention_mask"],
        )
        return loss_dic

    main()
