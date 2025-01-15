from typing import *

import hkkang_utils.list as list_utils
import torch

from eagle.index.codecs.registry import CODEC_STRIDED_REGISTRY
from eagle.index.index_loader import IndexLoader
from eagle.search.algorithm import (
    compute_sum_maxsim,
    reduce_element_wise_relevance_scores,
    token_interaction_with_relation,
)
from eagle.search.strided_tensor import StridedTensor
import hkkang_utils.time as time_utils


class PLAID:
    def __init__(
        self,
        index_path: str,
        indexer_name: str,
        ncells: int = 2,
        ndocs: int = 256,
        centroid_threshold: float = 0.45,
        skip_stage2: bool = False,
        d_cross_attention_layer: torch.nn.Module = None,
        d_weight_project_layer: torch.nn.Module = None,
        d_weight_layer_norm: torch.nn.Module = None,
        relation_encoder: torch.nn.Module = None,
        relation_scale_factor: float = 1.0,
        agg_in_phrase_level: bool = False,
    ) -> None:
        self.index = IndexLoader(index_path=index_path)
        self.ndocs = ndocs
        self.ncells = ncells
        self.skip_stage2 = skip_stage2
        self.centroid_threshold = centroid_threshold
        # For computing document weight dynamically
        self.d_cross_attention_layer = d_cross_attention_layer
        self.d_weight_project_layer = d_weight_project_layer
        self.d_weight_layer_norm = d_weight_layer_norm
        # For relation-based scoring
        self.relation_encoder = relation_encoder
        self.relation_scale_factor = relation_scale_factor
        # For aggregation in phrase level
        self.agg_in_phrase_level = agg_in_phrase_level
        # Set embeddings
        self._set_embeddings_strided(indexer_name)
        # Setting
        self.use_higher_precision = True
        # For analysis
        self.timer = time_utils.Timer(
            class_name=self.__class__.__name__, func_name="search"
        )
        self.timer_stage1 = time_utils.Timer(
            class_name=self.__class__.__name__, func_name="Stage 1"
        )
        self.timer_stage2 = time_utils.Timer(
            class_name=self.__class__.__name__, func_name="Stage 2"
        )
        self.timer_stage3 = time_utils.Timer(
            class_name=self.__class__.__name__, func_name="Stage 3"
        )
        self.timer_stage4 = time_utils.Timer(
            class_name=self.__class__.__name__, func_name="Stage 4"
        )
        self.timer_d_weight = time_utils.Timer(
            class_name=self.__class__.__name__, func_name="d_weight"
        )
        self.apply_weights = True

    @property
    def is_use_d_weight(self) -> bool:
        return (
            self.d_cross_attention_layer is not None
            and self.d_weight_project_layer is not None
            and self.d_weight_layer_norm is not None
        )

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        with self.timer.measure():
            return self.search(*args, **kwargs)

    def search(
        self,
        query_tok: torch.Tensor,
        tok_weight: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        gold_doc_ids: Optional[torch.Tensor] = None,
        q_scatter_indices: Optional[torch.Tensor] = None,
        return_intermediate_pids: bool = False,
        is_debug: bool = False,
    ) -> torch.Tensor:
        # Stage 1: Get initial candidate pids
        with self.timer_stage1.measure():
            pids, centroid_scores = self.get_initial_pids(query_tok, mask)
            pids1 = pids.tolist()
        # Stage 2: Filter pids using pruned centroid scores
        with self.timer_stage2.measure():
            pids = self.filter_with_pruning_centroids(pids, centroid_scores, tok_weight)
            pids2 = pids.tolist()
        # Stage 3: Filter pids using full centroid scores
        with self.timer_stage3.measure():
            pids = self.filter_without_pruning_centroids(
                pids, centroid_scores, tok_weight
            )
            pids3 = pids.tolist()
        # Append gold_doc_ids to the last pids
        with self.timer.pause():
            if gold_doc_ids is not None:
                # Find unselected gold_doc_ids
                unselected_gold_doc_ids = [
                    doc_id for doc_id in gold_doc_ids if doc_id not in pids
                ]
                # Replace the last pids with gold_doc_ids
                pids = torch.cat(
                    [
                        pids[: len(pids) - len(unselected_gold_doc_ids)],
                        torch.tensor(
                            unselected_gold_doc_ids,
                            device=pids.device,
                            dtype=pids.dtype,
                        ),
                    ]
                )
        # Stage 4: Final ranking with decomposed embeddings
        with self.timer_stage4.measure():
            result = self.rank_pids(
                query_tok=query_tok,
                q_tok_weight=tok_weight,
                q_mask=mask,
                pids=pids,
                is_debug=is_debug,
                q_scatter_indices=q_scatter_indices,
            )
        if is_debug:
            return result[0], result[1], (pids1, pids2, pids3), result[2:]
        final_pids, scores, element_wise_scores = result[:3]
        if return_intermediate_pids:
            return final_pids, scores, element_wise_scores, (pids1, pids2, pids3)

    def _set_embeddings_strided(self, indexer_name: str) -> None:
        self.tok_embeddings_strided = CODEC_STRIDED_REGISTRY[indexer_name](
            self.index.codec,
            self.index.tok_embeddings,
            doclens=self.index.tok_lens,
            tok_ids=self.index.tok_ids,
        )
        self.tok_offsets = self.tok_embeddings_strided.codes_strided.offsets

    def get_top_centroids(
        self, query: torch.Tensor, mask: torch.Tensor, topk: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """This is the helper method for stage 1 of the PLAID scoring pipeline.

        :param query: shape (ntoks, dim)
        :type query: torch.Tensor
        :param mask: shape (ntoks)
        :type mask: torch.Tensor
        :param topk: number of centroids per token, defaults to 10
        :type topk: int
        :return: _description_
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        if self.index.codec.centroids.dtype != query.dtype:
            query = query.to(self.index.codec.centroids.dtype)
        scores = self.index.codec.centroids @ query.T
        centroids = scores.topk(topk, dim=0, sorted=False).indices.permute(1, 0)
        if mask is not None:
            mask = mask.squeeze()
            centroids = centroids[mask == False]
        centroids = centroids.flatten().contiguous()
        centroids = centroids.unique(sorted=False)
        return centroids, scores

    def get_initial_pids(
        self, query: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """This is the stage 1 of the PLAID scoring pipeline.

        :param query: shape (ntoks, dim)
        :type query: torch.Tensor
        :param mask: shape (ntoks)
        :type mask: torch.Tensor
        :return: shape (ndocs), shape (ntoks)
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        centroids, scores = self.get_top_centroids(
            query=query, mask=mask, topk=self.ncells
        )
        pids, cell_lengths = self.index.tok_ivf.lookup(centroids)
        pids = pids.unique(sorted=False)
        return pids, scores

    def filter_with_centroid_interaction(
        self,
        pids: torch.Tensor,
        centroid_scores: torch.Tensor,
        centroid_threshold: float,
        weight: torch.Tensor,
        topk: int,
    ) -> torch.Tensor:
        """
        Compute the approximated scores of each pid using the centroid scores.
            First, we compute the approximate scores for each code (i.e., token) in the document.
            Then, we reduce the code scores to get the approximated score for each pid.

        :param pids: shape (n_docs), where n_docs is the number of initial candidate documents
        :type pids: torch.Tensor
        :param centroid_scores: shape (n_centroids, ntoks), where n_centroids is the number of centroids
        :type centroid_scores: torch.Tensor
        :return: shape (ndocs), where ndocs is the number of final candidate documents
        :rtype: torch.Tensor
        """
        # Get docs whose centroid scores are above the threshold
        # centroid_threshold = centroid_scores.mean()
        is_with_pruning_centroids = centroid_threshold > 0
        valid_centroid_indices = (
            centroid_scores.max(-1).values >= centroid_threshold
            if is_with_pruning_centroids
            else None
        )

        # Apply weights
        if self.apply_weights and weight is not None:
            centroid_scores = centroid_scores * weight.transpose(0, 1).expand(
                centroid_scores.shape[0], -1
            )

        # Batch size may need to be changed in the future.
        batch_size = 100000
        all_approx_scores: List[torch.Tensor] = []
        # Perform computation in chunks to avoid OOM
        for pids_chunk in list_utils.chunks(pids, chunk_size=batch_size):
            # Get the centroid indices and the number of codes in each pid
            centroid_ids, codes_lengths = self.tok_embeddings_strided.lookup_codes(
                pids_chunk
            )

            # Prune centroids if necessary
            if is_with_pruning_centroids:
                selected_centroid_indices = valid_centroid_indices[centroid_ids]
                # Use only the valid codes
                centroid_ids = centroid_ids[selected_centroid_indices]
                # Get new length for valid codes
                pruned_codes_strided = StridedTensor(
                    selected_centroid_indices, codes_lengths, use_gpu=True
                )
                pruned_codes_padded, pruned_codes_mask = (
                    pruned_codes_strided.as_padded_tensor()
                )
                codes_lengths = (pruned_codes_padded * pruned_codes_mask).sum(dim=1)

            # Check if there are no valid indices and skip the computation
            if len(centroid_ids) == 0:
                all_approx_scores.append(
                    torch.zeros(
                        (len(pids_chunk),),
                        dtype=centroid_scores.dtype,
                        device=pids.device,
                    )
                )
                continue

            # Approximate scores of each pid using centroids scores
            approx_code_scores_with_centroid = centroid_scores[centroid_ids]

            # Reduce the code scores to get the approximated score for each pid
            approx_scores_strided = StridedTensor(
                approx_code_scores_with_centroid, codes_lengths, use_gpu=True
            )
            approx_scores_padded, approx_scores_mask = (
                approx_scores_strided.as_padded_tensor()
            )
            approx_scores, _ = reduce_element_wise_relevance_scores(
                element_wise_scores=approx_scores_padded, k_mask=~approx_scores_mask
            )

            # Append the approximated scores to the list
            all_approx_scores.append(approx_scores)
        all_approx_scores = torch.cat(all_approx_scores, dim=0)

        # Check no bug: Make sure the length is correct
        assert len(pids) == len(
            all_approx_scores
        ), f"Length mismatch: {len(pids)} != {len(all_approx_scores)}"

        # Sort pids based on the approximated scores and get the topk pids
        pids = pids[
            torch.topk(all_approx_scores, k=min(topk, len(all_approx_scores))).indices
        ]

        return pids

    def filter_with_pruning_centroids(
        self, pids: torch.Tensor, centroid_scores: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        """This is the stage 2 of the PLAID scoring pipeline.

        :param pids: shape (n_docs), where n_docs is the number of initial candidate documents
        :type pids: torch.Tensor
        :param centroid_scores: shape (n_centroids, ntoks), where n_centroids is the number of centroids
        :type centroid_scores: torch.Tensor
        :return: shape (ndocs), where ndocs is the number of final candidate documents
        :rtype: torch.Tensor
        """
        topk = self.ndocs * 4
        centroid_threshold = self.centroid_threshold
        return self.filter_with_centroid_interaction(
            pids=pids,
            centroid_scores=centroid_scores,
            centroid_threshold=centroid_threshold,
            weight=weight,
            topk=topk,
        )

    def filter_without_pruning_centroids(
        self, pids: torch.Tensor, centroid_scores: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        """This is the stage 3 of the PLAID scoring pipeline.

        :param pids: shape (n_docs), where n_docs is the number of initial candidate documents
        :type pids: torch.Tensor
        :param centroid_scores: shape (n_centroids, ntoks), where n_centroids is the number of centroids
        :type centroid_scores: torch.Tensor
        :return: shape (ndocs), where ndocs is the number of final candidate documents
        :rtype: torch.Tensor
        """
        topk = self.ndocs
        centroid_threshold = 0
        return self.filter_with_centroid_interaction(
            pids=pids,
            centroid_scores=centroid_scores,
            centroid_threshold=centroid_threshold,
            weight=weight,
            topk=topk,
        )

    def rank_pids(
        self,
        query_tok: torch.Tensor,
        q_tok_weight: torch.Tensor,
        q_mask: torch.Tensor,
        pids: torch.Tensor,
        q_scatter_indices: Optional[torch.Tensor] = None,
        is_debug: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """This is the stage 4 of the PLAID scoring pipeline.
        We compute the exact scores of the pids using the decomposed embeddings.

        :param q: shape (ntoks, dim)
        :type q: torch.Tensor
        :param q_weight: shape (ntoks)
        :type q_weight: torch.Tensor
        :param q_mask: shape (ntoks)
        :type q_mask: torch.Tensor
        :param pids: shape (ndocs)
        :type pids: torch.Tensor
        :return: shape (ndocs), Final retrieval results. Document ids ranked by the exact scores
        :rtype: List[int]
        """
        # Extract document token embeddings
        d_tok_packed, d_tok_length, d_tok_ids = self.tok_embeddings_strided.lookup_pids(
            pids
        )

        if self.is_use_d_weight:
            with self.timer_d_weight.measure():
                d_tok_packed_float = d_tok_packed.float()
                cross_encoded_d_tok_vectors, _ = self.d_cross_attention_layer(
                    d_tok_packed_float,
                    query_tok,
                    query_tok,
                )
                # Add and normalize
                cross_encoded_d_tok_vectors = (
                    cross_encoded_d_tok_vectors + d_tok_packed_float
                )
                cross_encoded_d_tok_vectors = self.d_weight_layer_norm(
                    cross_encoded_d_tok_vectors
                )
                d_tok_packed_weight = self.d_weight_project_layer(
                    cross_encoded_d_tok_vectors
                )
                d_tok_packed = (d_tok_packed * d_tok_packed_weight).to(
                    d_tok_packed.dtype
                )

        # Convert strided tensor to padded tensor
        d_tok_padded, d_tok_mask = StridedTensor(
            d_tok_packed, d_tok_length, use_gpu=True
        ).as_padded_tensor()
        d_tok_mask = ~d_tok_mask
        d_tok_padded.masked_fill_(d_tok_mask == True, 0)

        k_ids_padded, _ = StridedTensor(
            d_tok_ids, d_tok_length, use_gpu=True
        ).as_padded_tensor()
        # Make out k_ids_padded to be the same as d_tok_padded
        k_ids_padded.masked_fill_(d_tok_mask.squeeze(-1) == True, 0)

        if self.use_higher_precision:
            query_tok = query_tok.float()
            d_tok_padded = d_tok_padded.float()
        else:
            # Convert data type if necessary
            if query_tok.dtype != d_tok_packed.dtype:
                query_tok = query_tok.to(d_tok_packed.dtype)

        # Apply mask
        if q_mask is not None:
            # Reshape the mask if necessary
            if len(query_tok.shape) > len(q_mask.shape):
                q_mask = q_mask.unsqueeze(-1)
            query_tok.masked_fill_(q_mask == True, 0)

        if self.relation_encoder is None:
            # Apply weights
            if q_tok_weight is not None:
                query_tok = query_tok * q_tok_weight

            # Compute scores
            (
                max_scores_by_token,
                max_sim_by_token,
                element_wise_scores,
                max_key_tok_ids,
            ) = compute_sum_maxsim(
                q_encoded=query_tok,
                k_encoded=d_tok_padded,
                k_mask=d_tok_mask,
                return_max_scores=is_debug,
                return_element_wise_scores=True,
                k_ids=k_ids_padded,
            )
        else:
            query_tok = query_tok.unsqueeze(0).expand(
                d_tok_padded.size(0), query_tok.shape[0], query_tok.shape[1]
            )
            q_tok_weight = q_tok_weight.unsqueeze(0).expand(
                d_tok_padded.size(0), q_tok_weight.shape[0], q_tok_weight.shape[1]
            )
            if q_scatter_indices is not None:
                q_scatter_indices = q_scatter_indices.unsqueeze(0).expand(
                    d_tok_padded.size(0),
                    q_scatter_indices.shape[0],
                )
            d_tok_mask = d_tok_mask.squeeze(-1)
            (
                max_scores_by_token,
                element_wise_scores,
                max_key_tok_ids,
            ) = token_interaction_with_relation(
                q_tok=query_tok,
                q_tok_weight=q_tok_weight,
                d_tok=d_tok_padded,
                d_tok_mask=d_tok_mask,
                q_scale_factors=None,
                relation_encoder=self.relation_encoder,
                relation_scale_factor=self.relation_scale_factor,
                q_scatter_indices=q_scatter_indices,
                return_element_wise_scores=True,
                agg_in_phrase_level=self.agg_in_phrase_level,
            )
            max_sim_by_token = None

        max_scores = max_scores_by_token
        # Sort pids based on the scores
        max_scores, indices = torch.sort(max_scores, descending=True)
        # Reorder base on the scores
        pids = pids[indices]
        element_wise_scores = element_wise_scores[indices]
        if max_sim_by_token is not None:
            max_sim_by_token = max_sim_by_token[indices]
        if max_key_tok_ids is not None:
            max_key_tok_ids = max_key_tok_ids[indices]

        if is_debug:
            return (
                pids,
                max_scores,
                element_wise_scores,
                k_ids_padded[indices],
            )

        return pids, max_scores, element_wise_scores


def main():
    index_path = "/root/EAGLE/index_for_default/msmarco/indexes/msmarco.default.nbits=2"
    searcher = PLAID(index_path)
    # Create 10 dummy query vector with 128 dimension
    query = torch.randn(12, 128).cuda()
    # Create random mask for query
    q_mask = torch.randint(0, 2, (12,), dtype=torch.bool, device="cuda")
    # Create random weights for query. Scale between 0 to 1
    q_weights = torch.randint(0, 100, (12,), device="cuda").float() / 100
    pids = searcher(query=query, mask=q_mask, weight=q_weights)


if __name__ == "__main__":
    main()
    pass
