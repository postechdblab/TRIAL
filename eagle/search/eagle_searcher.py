from typing import *

import hkkang_utils.time as time_utils
import torch
from omegaconf import DictConfig

from eagle.model.base_model import BaseModel
from eagle.search.base_searcher import BaseSearcher
from eagle.search.plaid import PLAID


class EAGLESearcher(BaseSearcher):
    def __init__(
        self,
        model: BaseModel,
        cfg: DictConfig,
        index_dir_path: str,
    ) -> None:
        super().__init__(cfg=cfg, model=model)
        self.index_dir_path = index_dir_path
        self.plaid = PLAID(
            index_path=self.index_dir_path,
            indexer_name=model.cfg.name,
            d_cross_attention_layer=self.model.cross_att_layer,
            d_weight_project_layer=self.model.d_weight_layer,
            d_weight_layer_norm=self.model.d_weight_layer_norm,
            relation_encoder=self.model.relation_encoder,
            relation_scale_factor=model.cfg.relation_scale_factor,
        )
        self.timer_encodings = time_utils.Timer(
            class_name=self.__class__.__name__, func_name="encode"
        )

    def search(
        self,
        q_tok_ids: torch.Tensor,
        q_tok_att_mask: torch.Tensor,
        q_tok_mask: torch.Tensor,
        pos_doc_indices: List[List[int]] = None,
        **Kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[List, List, List]]]:
        # Config
        bsize = q_tok_ids.size(0)

        # Preprocess the pids (decrease by 1 due to 0-based indexing)
        with self.timer.pause():
            if pos_doc_indices is not None:
                pos_doc_indices = self.preprocess_doc_indices(pos_doc_indices)

        # Encode query
        with self.timer_encodings.measure():
            result = self.model.encode_q_text(
                tok_ids=q_tok_ids, att_mask=q_tok_att_mask, tok_mask=q_tok_mask
            )
            q_tok_projected = result[1]
            q_tok_weight = result[4]

        # Perform search one-by-one
        all_pids = []
        all_scores = []
        all_intermediate_pids = []
        all_qd_scores = []
        for b_idx in range(bsize):
            # Retrieve pids and scores
            query_tok = q_tok_projected[b_idx]
            mask = q_tok_mask[b_idx]
            weight = q_tok_weight[b_idx]

            # Perform retrieval
            (
                retrieved_pids,
                scores,
                qd_scores,
                intermediate_pids,
            ) = self.plaid(
                query_tok=query_tok,
                tok_weight=weight,
                mask=mask,
                gold_doc_ids=(
                    None if pos_doc_indices is None else pos_doc_indices[b_idx]
                ),
                return_intermediate_pids=True,
            )
            retrieved_pids = retrieved_pids.cpu()

            # Increase the values of pids by 1 (0-based indexing)
            with self.timer.pause():
                retrieved_pids = self.postprocess_doc_indices(retrieved_pids)
                stage_1_pids = self.postprocess_doc_indices(intermediate_pids[0])
                stage_2_pids = self.postprocess_doc_indices(intermediate_pids[1])
                stage_3_pids = self.postprocess_doc_indices(intermediate_pids[2])

            # Aggregate results
            all_pids.append(retrieved_pids)
            all_scores.append(scores.cpu())
            all_intermediate_pids.append((stage_1_pids, stage_2_pids, stage_3_pids))
            all_qd_scores.append(qd_scores)

        return (
            all_pids,
            all_scores,
            all_qd_scores,
            all_intermediate_pids,
        )
