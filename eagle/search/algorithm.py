from typing import *

import torch


from eagle.search.strided_tensor import StridedTensor


def compute_sum_maxsim(
    q_encoded: torch.Tensor,
    k_encoded: torch.Tensor,
    q_mask: Optional[torch.Tensor] = None,
    k_mask: Optional[torch.Tensor] = None,
    k_lengths: Optional[torch.Tensor] = None,
    return_max_scores: bool = False,
    return_element_wise_scores: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Compute the sum of the maximum similarity scores between the query and the documents.

    :param q_encoded: Shape: (n_query, seq_len, dim). Vector representation of the query.
    :type q_encoded: torch.Tensor
    :param d_encoded: Shape: (n_docs, seq_len, dim). Vector representation of the documents.
    :type d_encoded: torch.Tensor
    :param d_mask: Mask for the documents, defaults to None
    :type d_mask: Optional[torch.Tensor], optional
    :param return_max_scores: Whether to return the max scores for query, defaults to False
    :type return_max_scores: bool, optional
    :param return_entire_scores: Whether to return the entire similarity score matrix, defaults to False
    :type return_entire_scores: bool, optional
    :return: Sum of the maximum similarity scores between the query and the documents.
    :rtype: torch.Tensor
    """
    max_sim_scores, element_wise_scores = compute_maxsim(
        q_encoded=q_encoded,
        k_encoded=k_encoded,
        k_mask=k_mask,
        k_lengths=k_lengths,
        return_element_wise_scores=return_element_wise_scores,
    )

    if q_mask is None:
        sum_maxsim_scores = max_sim_scores.sum(1)
    else:
        q_mask = q_mask.squeeze(-1)
        sum_maxsim_scores = torch.scatter_add(
            input=torch.zeros(
                (max_sim_scores.shape[0], 2),
                device=max_sim_scores.device,
                dtype=max_sim_scores.dtype,
            ),
            dim=1,
            index=q_mask.long(),
            src=max_sim_scores,
        )
        sum_maxsim_scores = sum_maxsim_scores[:, 0]
        # sum_maxsim_scores = torch.scatter(src=torch.zeros((max_sim_scores.shape[0], 2), device=max_sim_scores.device, dtype=max_sim_scores.dtype), sum_maxsim_scores = sum_maxsim_scores[:, 0]
        # torch.zeros((max_sim_scores.shape[0], 2), device=max_sim_scores.device, dtype=max_sim_scores.dtype).scatter_add_(dim=1, index=q_mask.long(), src=max_sim_scores)

    if not return_max_scores:
        max_sim_scores = None

    if not return_element_wise_scores:
        element_wise_scores = None

    return sum_maxsim_scores, max_sim_scores, element_wise_scores


def compute_maxsim(
    q_encoded: torch.Tensor,
    k_encoded: torch.Tensor,
    k_mask: Optional[torch.Tensor] = None,
    k_lengths: Optional[torch.Tensor] = None,
    return_element_wise_scores: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    # Compute the element-wise relevance scores
    element_wise_scores = k_encoded @ q_encoded.transpose(-2, -1)

    # Pad the relevance scores if the key vectors are packed
    if k_lengths is not None:
        assert k_mask is None, "k_mask should be None when k_lengths is provided."
        # Unpack the encoded key items
        element_wise_scores, score_mask = StridedTensor(
            element_wise_scores, k_lengths, use_gpu=True
        ).as_padded_tensor()
        k_mask = ~score_mask

    max_sim_scores = maxsim_from_element_wise_relevance_score(
        element_wise_scores=element_wise_scores, k_mask=k_mask
    )

    if not return_element_wise_scores:
        element_wise_scores = None

    return max_sim_scores, element_wise_scores


def maxsim_from_element_wise_relevance_score(
    element_wise_scores: torch.Tensor,
    k_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute the maximum similarity score from the token-wise relevance scores.

    :param element_wise_scores: Shape: (?, ?). Token relevance scores for each token in each document.
    :type element_wise_scores: torch.Tensor
    :param k_mask: Shape: (?), defaults to None
    :type k_mask: Optional[torch.Tensor], optional
    :return: Shape: (?) Maximum similarity score for each document.
    :rtype: torch.Tensor
    """
    # Apply mask to the scores
    if k_mask is not None:
        element_wise_scores.masked_fill_(k_mask, float("-inf"))

    # Find the maximum scores for each document
    return element_wise_scores.max(dim=1).values


def reduce_element_wise_relevance_scores(
    element_wise_scores: torch.Tensor,
    k_mask: Optional[torch.Tensor] = None,
    return_max_scores: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Reduce the relevance scores computed between query and key.

    :param token_relevance_scores: Shape: (n_docs, n_tokens). Token relevance scores for each token in each document.
    :type token_relevance_scores: torch.Tensor
    :param d_mask: Mask for the documents, defaults to None
    :type d_mask: Optional[torch.Tensor], optional
    :param return_max_scores: Whether to return the max scores for query, defaults to False
    :type return_max_scores: bool, optional
    :return: Reduced token relevance scores.
    :rtype: torch.Tensor
    """
    # Aggregate the maximum scores
    max_scores = maxsim_from_element_wise_relevance_score(
        element_wise_scores=element_wise_scores, k_mask=k_mask
    )

    # Sum the maximum scores
    summed_scores = max_scores.sum(1)

    if not return_max_scores:
        max_scores = None

    return summed_scores, max_scores
