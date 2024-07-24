from typing import *

import hkkang_utils.list as list_utils
import torch

from eagle.index.codecs.residual_embeddings_strided import ResidualEmbeddingsStrided
from eagle.index.index_loader import IndexLoader
from eagle.search.algorithm import (
    compute_sum_maxsim,
    reduce_element_wise_relevance_scores,
)
from eagle.search.strided_tensor import StridedTensor


class PLAID:
    def __init__(
        self,
        index_path: str,
        ncells: int = 2,
        ndocs: int = 256,
        centroid_threshold: float = 0.45,
        skip_stage2: bool = False,
        d_cross_attention_layer: torch.nn.Module = None,
        d_weight_project_layer: torch.nn.Module = None,
        d_weight_layer_norm: torch.nn.Module = None,
    ) -> None:
        self.index = IndexLoader(index_path=index_path)
        self.ndocs = ndocs
        self.ncells = ncells
        self.skip_stage2 = skip_stage2
        self.centroid_threshold = centroid_threshold
        self.d_cross_attention_layer = d_cross_attention_layer
        self.d_weight_project_layer = d_weight_project_layer
        self.d_weight_layer_norm = d_weight_layer_norm
        self._set_embeddings_strided()

    def __call__(
        self,
        query: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        gold_doc_ids: Optional[torch.Tensor] = None,
        return_intermediate_pids: bool = False,
    ) -> torch.Tensor:
        # Stage 1: Get initial candidate pids
        pids, centroid_scores = self.get_initial_pids(query, mask)
        pids1 = pids.tolist()
        # Stage 2: Filter pids using pruned centroid scores
        pids = self.filter_with_pruning_centroids(pids, centroid_scores, weight)
        pids2 = pids.tolist()
        # Stage 3: Filter pids using full centroid scores
        pids = self.filter_without_pruning_centroids(pids, centroid_scores, weight)
        pids3 = pids.tolist()
        if gold_doc_ids is not None:
            # Find unselected gold_doc_ids
            gold_doc_ids = [doc_id for doc_id in gold_doc_ids if doc_id not in pids]
            # Replace the last pids with gold_doc_ids
            pids = torch.cat(
                [
                    pids[: len(pids) - len(gold_doc_ids)],
                    torch.tensor(gold_doc_ids, device=pids.device, dtype=pids.dtype),
                ]
            )
        # Stage 4: Final ranking with decomposed embeddings
        final_pids, scores = self.rank_pids(query, weight, mask, pids)
        if return_intermediate_pids:
            return final_pids, scores, (pids1, pids2, pids3)

    def _set_embeddings_strided(self) -> None:
        self.tok_embeddings_strided = ResidualEmbeddingsStrided(
            self.index.codec, self.index.tok_embeddings, self.index.tok_lens
        )
        self.tok_offsets = self.tok_embeddings_strided.codes_strided.offsets

        if self.index.cls_embeddings is None:
            self.cls_embeddings_strided = None
            self.cls_offsets = None
        else:
            self.cls_embeddings_strided = ResidualEmbeddingsStrided(
                self.index.codec, self.index.cls_embeddings, self.index.cls_lens
            )
            self.cls_offsets = self.cls_embeddings_strided.codes_strided.offsets

        if self.index.phrase_embeddings is None:
            self.phrase_embeddings_strided = None
            self.phrase_offsets = None
        else:
            self.phrase_embeddings_strided = ResidualEmbeddingsStrided(
                self.index.codec, self.index.phrase_embeddings, self.index.phrase_lens
            )
            self.phrase_offsets = self.phrase_embeddings_strided.codes_strided.offsets

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
        apply_weights = True
        if apply_weights and weight is not None:
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
        query: torch.Tensor,
        q_weight: torch.Tensor,
        q_mask: torch.Tensor,
        pids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        # Apply weights
        if q_weight is not None:
            query = query * q_weight
        # Apply mask
        if q_mask is not None:
            query.masked_fill_(q_mask, 0)
        # Extract document embeddings
        d_packed, d_length = self.tok_embeddings_strided.lookup_pids(pids)

        # If use document weight, compute document weights and scale the embeddings
        if self.d_cross_attention_layer is not None:
            # Compute document weights
            cross_encoded_vectors, cross_attn_weights = self.d_cross_attention_layer(
                query=d_packed.unsqueeze(0).float(),
                key=query.unsqueeze(0).float(),
                value=query.unsqueeze(0).float(),
                key_padding_mask=q_mask.unsqueeze(0).squeeze(-1),
            )
            # Add and normalize
            cross_encoded_vectors = cross_encoded_vectors.squeeze(0) + d_packed
            # Compute weights
            cross_encoded_vectors = self.d_weight_layer_norm(cross_encoded_vectors)
            d_weights = self.d_weight_project_layer(cross_encoded_vectors)
            # Scale embeddings
            d_packed = d_packed * d_weights

        if query.dtype != d_packed.dtype:
            query = query.to(d_packed.dtype)

        # Compute scores
        max_scores, _, _ = compute_sum_maxsim(
            q_encoded=query, k_encoded=d_packed, k_lengths=d_length
        )
        # Sort pids based on the scores
        max_scores, indices = torch.sort(max_scores, descending=True)
        pids = pids[indices]

        return pids, max_scores


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
