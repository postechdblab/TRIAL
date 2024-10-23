from typing import *

import torch
from omegaconf import DictConfig

from eagle.model.colbert import ColBERT
from eagle.search import PLAID
from eagle.search.base_searcher import BaseSearcher


class ColBERTSearcher(BaseSearcher):
    def __init__(self, cfg: DictConfig, model: ColBERT, index_dir_path: str) -> None:
        super().__init__(cfg=cfg, model=model)
        self.index_dir_path = index_dir_path
        self.plaid = PLAID(index_path=self.index_dir_path, indexer_name=model.cfg.name)

    def __call__(
        self,
        q_tok_ids: torch.Tensor,
        q_tok_att_mask: torch.Tensor,
        q_tok_mask: torch.Tensor,
        pos_doc_indices: List[List[int]] = None,
        **Kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[List, List, List]]]:
        # Configs
        bsize = q_tok_ids.size(0)

        # Encode query
        q_tok_projected, q_tok_scale_factor = self.model.encode_q_text(
            tok_ids=q_tok_ids, att_mask=q_tok_att_mask, tok_mask=q_tok_mask
        )

        # Perform search one-by-one
        all_pids = []
        all_scores = []
        all_intermediate_pids = []
        for b_idx in range(bsize):
            # Retrieve pids and scores
            query_tok = q_tok_projected[b_idx]
            mask = q_tok_mask[b_idx]

            # Perform retrieval
            retrieved_pids, scores, intermediate_pids = self.plaid(
                query_tok=query_tok,
                mask=mask,
                gold_doc_ids=pos_doc_indices[b_idx] if pos_doc_indices else None,
                return_intermediate_pids=True,
            )
            # Aggregate results
            all_pids.append(retrieved_pids.cpu())
            all_scores.append(scores.cpu())
            all_intermediate_pids.append(intermediate_pids)

        return all_pids, all_scores, all_intermediate_pids
