from typing import *

import hkkang_utils.list as list_utils
import torch

from colbert.indexing.codecs.residual_embeddings_strided import (
    ResidualEmbeddingsStrided,
)
from colbert.modeling.colbert import colbert_score_packed, colbert_score_reduce
from colbert.search.index_loader import IndexLoader
from colbert.search.strided_tensor import StridedTensor


class PLAID:
    def __init__(
        self,
        index_path: str,
        ncells: int = 2,
        ndocs: int = 256,
        centroid_threshold: float = 0.45,
        skip_stage2: bool = False,
    ) -> None:
        self.index = IndexLoader(index_path=index_path)
        self.ndocs = ndocs
        self.ncells = ncells
        self.skip_stage2 = skip_stage2
        self.centroid_threshold = centroid_threshold
        self._set_embeddings_strided()

    def __call__(
        self,
        query: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Stage 1: Get initial candidate pids
        pids, centroid_scores = self.get_initial_pids(query, weight, mask)
        # Stage 2: Filter pids using pruned centroid scores
        pids = self.filter_with_pruning_centroids(pids, centroid_scores)
        # Stage 3: Filter pids using full centroid scores
        pids = self.filter_without_pruning_centroids(pids, centroid_scores)
        # Stage 4: Final ranking with decomposed embeddings
        return self.rank_pids(query, weight, mask, pids)

    def _set_embeddings_strided(self) -> None:
        self.embeddings_strided = ResidualEmbeddingsStrided(
            self.index.codec, self.index.embeddings, self.index.doclens
        )
        self.offsets = self.embeddings_strided.codes_strided.offsets

    def get_top_centroids(
        self, query: torch.Tensor, weight: torch.Tensor, mask: torch.Tensor, topk: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """This is the helper method for stage 1 of the PLAID scoring pipeline.

        :param query: shape (ntoks, dim)
        :type query: torch.Tensor
        :param weight: shape (ntoks)
        :type weight: torch.Tensor
        :param mask: shape (ntoks)
        :type mask: torch.Tensor
        :param topk: number of centroids per token, defaults to 10
        :type topk: int
        :return: _description_
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        scores = self.index.codec.centroids @ query.T
        if weight is not None:
            scores = scores * weight
        centroids = scores.topk(topk, dim=0, sorted=False).indices.permute(1, 0)
        if mask is not None:
            centroids = centroids[mask == True]
        centroids = centroids.flatten().contiguous()
        centroids = centroids.unique(sorted=False)
        return centroids, scores

    def get_initial_pids(
        self, query: torch.Tensor, weight: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """This is the stage 1 of the PLAID scoring pipeline.

        :param query: shape (ntoks, dim)
        :type query: torch.Tensor
        :param weight: shape (ntoks)
        :type weight: torch.Tensor
        :param mask: shape (ntoks)
        :type mask: torch.Tensor
        :return: shape (ndocs), shape (ntoks)
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        centroids, scores = self.get_top_centroids(
            query=query, weight=weight, mask=mask, topk=self.ncells
        )
        pids, cell_lengths = self.index.ivf.lookup(centroids)
        pids = pids.unique(sorted=False)
        return pids, scores

    def filter_with_centroid_interaction(
        self,
        pids: torch.Tensor,
        centroid_scores: torch.Tensor,
        centroid_threshold: float,
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
        is_with_pruning_centroids = centroid_threshold > 0
        valid_centroid_indices = (
            centroid_scores.max(-1).values >= centroid_threshold
            if is_with_pruning_centroids
            else None
        )

        # Batch size may need to be changed in the future.
        batch_size = 100000
        all_approx_scores: List[torch.Tensor] = []
        # Perform computation in chunks to avoid OOM
        for pids_chunk in list_utils.chunks(pids, chunk_size=batch_size):
            # Get the centroid indices and the number of codes in each pid
            centroid_ids, codes_lengths = self.embeddings_strided.lookup_codes(
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
            approx_scores = colbert_score_reduce(
                approx_scores_padded, approx_scores_mask
            )

            # Append the approximated scores to the list
            all_approx_scores.append(approx_scores)
        all_approx_scores = torch.cat(all_approx_scores, dim=0)

        # Check no bug: Make sure the length is correct
        assert len(pids) == len(
            all_approx_scores
        ), f"Length mismatch: {len(pids)} != {len(all_approx_scores)}"

        # Sort pids based on the approximated scores and get the topk pids
        pids = pids[torch.topk(all_approx_scores, k=topk).indices]

        return pids

    def filter_with_pruning_centroids(
        self, pids: torch.Tensor, centroid_scores: torch.Tensor
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
            topk=topk,
        )

    def filter_without_pruning_centroids(
        self, pids: torch.Tensor, centroid_scores: torch.Tensor
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
            query = query * q_weight.unsqueeze(1)
        # Apply mask
        if q_mask is not None:
            query = query * q_mask.unsqueeze(1)
        # Extract document embeddings
        d_packed, d_length = self.embeddings_strided.lookup_pids(pids)
        # Compute scores
        scores = colbert_score_packed(
            Q=query, D_packed=d_packed, D_lengths=d_length
        ).sum(1)
        # Sort pids based on the scores
        scores, indices = torch.sort(scores, descending=True)
        pids = pids[indices]
        return pids, scores


def main():
    index_path = (
        "/root/ColBERT/index_for_default/msmarco/indexes/msmarco.default.nbits=2"
    )
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
