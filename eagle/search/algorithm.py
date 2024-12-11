from typing import *

import torch

from eagle.model.utils import aggregate_vectors_with_indices


def compute_sum_maxsim(
    q_encoded: torch.Tensor,
    k_encoded: torch.Tensor,
    q_mask: Optional[torch.Tensor] = None,
    k_mask: Optional[torch.Tensor] = None,
    return_max_scores: bool = False,
    return_element_wise_scores: bool = False,
    k_ids: Optional[torch.Tensor] = None,
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
    max_sim_scores, element_wise_scores, max_sim_indices = compute_maxsim(
        q_encoded=q_encoded,
        k_encoded=k_encoded,
        k_mask=k_mask,
        return_element_wise_scores=return_element_wise_scores,
        is_debug=k_ids is not None,
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
            index=(q_mask == True).long(),
            src=max_sim_scores,
        )
        sum_maxsim_scores = sum_maxsim_scores[:, 0]

    if k_ids is not None:
        max_key_tok_ids = []
        for b_idx in range(k_ids.shape[0]):
            max_key_tok_ids.append(k_ids[b_idx][max_sim_indices[b_idx]])
        max_key_tok_ids = torch.stack(max_key_tok_ids)
    else:
        max_key_tok_ids = None

    if not return_max_scores:
        max_sim_scores = None

    if not return_element_wise_scores:
        element_wise_scores = None

    return sum_maxsim_scores, max_sim_scores, element_wise_scores, max_key_tok_ids


def compute_maxsim(
    q_encoded: torch.Tensor,
    k_encoded: torch.Tensor,
    k_mask: Optional[torch.Tensor] = None,
    return_element_wise_scores: bool = False,
    is_debug: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    # Compute the element-wise relevance scores
    element_wise_scores = k_encoded @ q_encoded.transpose(-2, -1)

    max_sim_scores = maxsim_from_element_wise_relevance_score(
        element_wise_scores=element_wise_scores, k_mask=k_mask
    )

    if is_debug:
        max_sim_indices = element_wise_scores.argmax(dim=1)
    else:
        max_sim_indices = None

    if not return_element_wise_scores:
        element_wise_scores = None

    return max_sim_scores, element_wise_scores, max_sim_indices


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
        element_wise_scores.masked_fill_(k_mask == True, float("-inf"))

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


def token_interaction_with_relation(
    q_tok: torch.Tensor,
    q_tok_weight: torch.Tensor,
    d_tok: torch.Tensor,
    relation_encoder: torch.nn.Module,
    relation_scale_factor: float = 1.0,
    d_tok_mask: Optional[torch.Tensor] = None,
    q_scale_factors: Optional[torch.Tensor] = None,
    q_scatter_indices: Optional[torch.Tensor] = None,
    return_element_wise_scores: bool = False,
    agg_in_phrase_level: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Literal[None]]:
    # Configurations
    q_tok_len = q_tok.shape[1]

    # Compute the similarity matrix between query tokens and document tokens
    element_wise_scores = d_tok @ q_tok.transpose(-2, -1)

    # Compute the relation embedding for each query token pairs
    first_toks_in_pairs = q_tok[:, torch.arange(0, q_tok_len - 1)]
    second_toks_in_pairs = q_tok[:, torch.arange(1, q_tok_len)]
    q_tok_pair_embs = torch.cat(
        (first_toks_in_pairs, second_toks_in_pairs),
        dim=2,
    )
    # Forward the relation embeddings to the MLP
    encoded_q_relations = None
    if relation_encoder is not None:
        encoded_q_relations: torch.Tensor = relation_encoder(q_tok_pair_embs)

    # Find the maximum similarity with relation in considered sequentially
    max_values_batch: List[torch.Tensor] = []
    max_indices_batch: List[torch.Tensor] = []
    element_wise_scores_with_relation_batch: List[torch.Tensor] = []
    for q_tok_idx in range(q_tok_len):
        # Find the maximum similarity with relation in considered
        selected_element_wise_scores = element_wise_scores[:, :, q_tok_idx]
        if q_tok_idx == 0:
            # Find the maximum value for the first token
            max_value, max_idx = selected_element_wise_scores.max(dim=1)
            final_selected_wise_scores = selected_element_wise_scores
        else:
            # Get the query relation embedding for the current token
            if relation_encoder is None:
                selected_q_relations = torch.ones(
                    (q_tok.shape[0], q_tok.shape[-1]),
                    device=q_tok.device,
                    dtype=q_tok.dtype,
                )
            else:
                selected_q_relations = encoded_q_relations[:, q_tok_idx - 1]

            if relation_encoder is None:
                encoded_d_relations = torch.ones_like(d_tok)
            else:
                # Create the document token relation embeddings
                prev_d_idx_batch = max_indices_batch[q_tok_idx - 1]
                # selected the document embeddings for the previous token
                prev_selected_d_toks = d_tok[
                    torch.arange(d_tok.shape[0]), prev_d_idx_batch
                ]
                # Repeat the previous document embeddings for the current token
                repeated_prev_selected_d_toks = prev_selected_d_toks.unsqueeze(
                    1
                ).expand(-1, d_tok.shape[1], -1)

                # create the document token pair embeddings
                d_tok_pair_embs = torch.cat(
                    [repeated_prev_selected_d_toks, d_tok],
                    dim=2,
                )
                # Forward the relation embeddings to the MLP
                encoded_d_relations: torch.Tensor = relation_encoder(d_tok_pair_embs)

            # Compute the similarity scores between the document token relation embeddings and the query token relation embedding
            element_wise_relation_scores = (
                selected_q_relations.unsqueeze(1)
                @ encoded_d_relations.transpose(-2, -1)
            ).squeeze(1)

            if d_tok_mask is None:
                element_wise_relation_scores = (
                    relation_scale_factor * element_wise_relation_scores
                )
            else:
                element_wise_relation_scores = (
                    relation_scale_factor * element_wise_relation_scores
                ).masked_fill(d_tok_mask == True, float(0))

            if q_scatter_indices is not None:
                # is_same_phrase shape: [400]
                is_same_phrase = (
                    q_scatter_indices[:, q_tok_idx - 1]
                    == q_scatter_indices[:, q_tok_idx]
                )
                # element_wise_relation_scores shape: [400, 184]
                element_wise_relation_scores = element_wise_relation_scores.masked_fill(
                    (~is_same_phrase)
                    .unsqueeze(1)
                    .expand(-1, element_wise_relation_scores.size(1)),
                    float(0),
                )

            # Add the similarity scores with the relational similarity scores
            final_selected_wise_scores = (
                selected_element_wise_scores + element_wise_relation_scores
            )
            # Find the maximum value for the current token
            max_value, max_idx = final_selected_wise_scores.max(dim=1)

        # Apply the query weight if it exists
        if q_tok_weight is not None:
            max_value = max_value * q_tok_weight[:, q_tok_idx].squeeze(-1)

        # Save the maximum value and index
        max_values_batch.append(max_value)
        max_indices_batch.append(max_idx)
        if return_element_wise_scores:
            element_wise_scores_with_relation_batch.append(final_selected_wise_scores)

    # Stack the maximum value and index

    max_values = torch.stack(max_values_batch).transpose(0, 1)
    # max_indices = torch.stack(max_indices_batch).transpose(0, 1)
    if return_element_wise_scores:
        element_wise_scores_with_relation_batch = torch.stack(
            element_wise_scores_with_relation_batch
        ).transpose(0, 1)

    if agg_in_phrase_level:
        assert (
            q_scatter_indices is not None
        ), "q_scatter_indices is required for phrase-level retrieval"
        max_values = aggregate_vectors_with_indices(
            src_tensor=max_values,
            scatter_indices=q_scatter_indices,
            reduce="mean",
        )

    # Compute the final scores
    sim_scores = max_values.sum(dim=1)
    if q_scale_factors is not None:
        sim_scores = sim_scores * q_scale_factors

    return sim_scores, element_wise_scores_with_relation_batch, None
