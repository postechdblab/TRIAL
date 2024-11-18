from typing import *

import torch
from omegaconf import DictConfig

from eagle.model.colbert import ColBERT
from eagle.search.base_searcher import BaseSearcher
from eagle.search.plaid import PLAID


class ColBERTSearcher(BaseSearcher):
    def __init__(self, cfg: DictConfig, model: ColBERT, index_dir_path: str) -> None:
        super().__init__(cfg=cfg, model=model)
        self.index_dir_path = index_dir_path
        self.plaid = PLAID(index_path=self.index_dir_path, indexer_name=model.cfg.name)

    def serach(
        self,
        q_tok_ids: torch.Tensor,
        q_tok_att_mask: torch.Tensor,
        q_tok_mask: torch.Tensor,
        pos_doc_indices: List[List[int]] = None,
        **Kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[List, List, List]]]:
        # Configs
        bsize = q_tok_ids.size(0)

        # Preprocess the pids (decrease by 1 due to 0-based indexing)
        if pos_doc_indices is not None:
            pos_doc_indices = torch.tensor(pos_doc_indices, dtype=torch.int64)
            pos_doc_indices = pos_doc_indices - torch.ones_like(pos_doc_indices)

        # Encode query
        q_tok_projected, q_tok_scale_factor = self.model.encode_q_text(
            tok_ids=q_tok_ids, att_mask=q_tok_att_mask, tok_mask=q_tok_mask
        )

        # Perform search one-by-one
        all_pids = []
        all_scores = []
        all_intermediate_pids = []
        all_qd_scores = []
        for b_idx in range(bsize):
            # Retrieve pids and scores
            query_tok = q_tok_projected[b_idx]
            mask = q_tok_mask[b_idx]

            # Perform retrieval
            retrieved_pids, scores, qd_scores, intermediate_pids = self.plaid(
                query_tok=query_tok,
                mask=mask,
                gold_doc_ids=(
                    None if pos_doc_indices is None else pos_doc_indices[b_idx]
                ),
                return_intermediate_pids=True,
            )
            # Increase the values of pids by 1 (0-based indexing)
            retrieved_pids = retrieved_pids.cpu()
            retrieved_pids = retrieved_pids + torch.ones_like(retrieved_pids)
            stage_1_pids = [item + 1 for item in intermediate_pids[0]]
            stage_2_pids = [item + 1 for item in intermediate_pids[1]]
            stage_3_pids = [item + 1 for item in intermediate_pids[2]]

            # Aggregate results
            all_pids.append(retrieved_pids)
            all_scores.append(scores.cpu())
            all_intermediate_pids.append((stage_1_pids, stage_2_pids, stage_3_pids))
            all_qd_scores.append(qd_scores)

        return all_pids, all_scores, all_qd_scores, all_intermediate_pids
