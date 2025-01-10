import logging
from typing import *

import torch
from beir.retrieval.evaluation import EvaluateRetrieval
from transformers import EvalPrediction


def aggregate_intermediate_metrics(
    probs: List[List[torch.Tensor]], total_data_num: int
) -> float:
    aggregated_probs = 0.0
    aggregated_num = 0
    # Loop over each inference step
    for probs_from_each_step in probs:
        # Loop over each item in the rank
        for probs_from_each_rank in probs_from_each_step:
            # Loop over each rank
            for probs_from_each_item in probs_from_each_rank:
                # Check if the number of data is valid
                if aggregated_num < total_data_num:
                    aggregated_num += 1
                    aggregated_probs += probs_from_each_item.item()
    # Check all items are evaluated
    assert (
        aggregated_num == total_data_num
    ), f"Invalid aggregated_num vs total_data_num: {aggregated_num} vs {total_data_num}"

    return aggregated_probs / total_data_num


def aggregate_final_metrics(
    metrics: List[List[Dict[str, torch.Tensor]]], total_data_num: int
) -> Dict[str, float]:
    aggregated_metrics = {}
    aggregated_num = 0
    # Loop over each inference step
    for metric_from_each_step in metrics:
        # Loop over each item in the rank
        for metric in metric_from_each_step:
            # Loop over each rank
            for rank_idx in range(len(list(metric.values())[0])):
                # Add all key and values to the aggregated_metrics
                if aggregated_num < total_data_num:
                    aggregated_num += 1
                    for key, value in metric.items():
                        if key not in aggregated_metrics:
                            aggregated_metrics[key] = 0.0
                        aggregated_metrics[key] += value[rank_idx].item()
    # Check all items are evaluated
    assert (
        aggregated_num == total_data_num
    ), f"Invalid aggregated_num vs total_data_num: {aggregated_num} vs {total_data_num}"

    # Average the metrics
    aggregated_metrics = {
        key: value / total_data_num for key, value in aggregated_metrics.items()
    }
    return aggregated_metrics


def result_to_beir_format(
    logits_batch: torch.Tensor, pids_batch: Optional[torch.Tensor] = None
) -> Dict[str, Dict[str, float]]:
    """Convert logits to BEIR format (i.e., dictionary with predicted score for each qid-pid)."""
    assert len(logits_batch.shape) == 2, f"Invalid logits shape: {logits_batch.shape}"

    if pids_batch is not None:
        # Check if the pids are valid
        assert (
            logits_batch.shape == pids_batch.shape
        ), f"Invalid pids shape: {pids_batch.shape}"

    logits_batch = logits_batch.tolist()

    dic = {}
    if pids_batch is None:
        for qid, logits in enumerate(logits_batch):
            dic[str(qid)] = {str(pid): float(logit) for pid, logit in enumerate(logits)}
    else:
        pids_batch = pids_batch.tolist()
        for qid, (logits, pids) in enumerate(zip(logits_batch, pids_batch)):
            dic[str(qid)] = {str(pid): float(logit) for logit, pid in zip(logits, pids)}
    return dic


def label_to_beir_format(
    labels_batch: torch.Tensor, pids_batch: Optional[torch.Tensor] = None
) -> Dict[str, Dict[str, int]]:
    """Convert labels to BEIR format (i.e., dictionary with target score for each qid-pid)."""
    assert len(labels_batch.shape) == 2, f"Invalid labels shape: {labels_batch.shape}"

    if pids_batch is not None:
        # Check if the pids are valid
        assert (
            labels_batch.shape == pids_batch.shape
        ), f"Invalid pids shape: {pids_batch.shape}"

    labels_batch = labels_batch.tolist()

    dic = {}
    if pids_batch is None:
        for qid, labels in enumerate(labels_batch):
            dic[str(qid)] = {str(pid): int(label) for pid, label in enumerate(labels)}
    else:
        pids_batch = pids_batch.tolist()
        for qid, (labels, pids) in enumerate(zip(labels_batch, pids_batch)):
            dic[str(qid)] = {str(pid): int(label) for label, pid in zip(labels, pids)}
    return dic


def compute_metrics(eval_pred: EvalPrediction, prefix: str = None) -> Dict[str, int]:
    k_values = [1, 3, 5, 10, 50, 100]
    # Extract logits and labels
    if eval_pred.inputs is None:
        logits, labels = eval_pred
        pids = None
    else:
        logits, labels, pids = eval_pred

    # Convert the format
    pred = result_to_beir_format(logits, pids)
    label = label_to_beir_format(labels, pids)

    # Evaluate with BEIR
    logging.getLogger().disabled = True
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
        qrels=label, results=pred, k_values=k_values, ignore_identical_ids=False
    )
    mrr = EvaluateRetrieval.evaluate_custom(
        qrels=label, results=pred, k_values=k_values, metric="mrr"
    )
    acc = EvaluateRetrieval.evaluate_custom(
        qrels=label, results=pred, k_values=k_values, metric="acc"
    )
    logging.getLogger().disabled = False

    # Add custom recall rate
    custom_recall = get_custom_metrics(logits, labels)
    success_rate = get_success_rate(logits, labels)

    # Combine metrics into one dictionary
    combined_metrics = (
        ndcg | _map | recall | precision | mrr | acc | custom_recall | success_rate
    )
    if prefix is not None:
        combined_metrics = {
            f"{prefix}_{key}": value for key, value in combined_metrics.items()
        }
    return combined_metrics


def get_recall_rates(
    ranked_pids: List[int], gold_pids: List[int], metric: str = "all"
) -> Dict[str, float]:
    """Get recall rates for top-k passages.
    is_absolute: If True, mark as correct when every gold pids are retrieved."""
    # Configure metric
    if metric == "all":
        is_correct = all
    elif metric == "any":
        is_correct = any
    else:
        raise ValueError(f"Invalid metric: {metric}")

    recall_rates = {}
    for k in [5, 10, 50, 100, 1000]:
        # Pass if k is larger than the number of retrieved passages
        if k > len(ranked_pids):
            continue
        # Calculate recall
        recall = is_correct([pid in ranked_pids[:k] for pid in gold_pids])
        recall_rates[f"@{k}"] = recall
    return recall_rates


def get_success_rate(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """Success@k means the percentage of queries that have at least one correct passage in the top-k retrieved passages."""
    # Logits: (batch_size, num_docs). It contains the scores of the retrieved passages.
    # Labels: (batch_size, num_docs). It contains the labels of the retrieved passages (0 or 1).

    # Get indices of passages sorted by scores (descending)
    sorted_indices = torch.argsort(logits, dim=1, descending=True)

    # Reorder labels according to sorted indices
    sorted_labels = torch.gather(labels, 1, sorted_indices)

    # Calculate cumulative maximum to check if there's any correct passage up to position k
    cummax_labels = torch.cummax(sorted_labels, dim=1)[0]

    # Calculate success@k for different k values
    k_values = [1, 3, 5, 10, 50, 100]
    results = {}

    for k in k_values:
        if k <= sorted_labels.size(
            1
        ):  # Only calculate if k is less than number of documents
            # Check if there's at least one correct passage in top-k
            success_at_k = (cummax_labels[:, k - 1] > 0).float().mean().item()
            results[f"success@{k}"] = success_at_k
        else:
            results[f"success@{k}"] = None

    return results


def get_custom_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    ranked_pids_list = []
    gold_pids_list = []
    ranked_pids_list = torch.argsort(logits, dim=1, descending=True).tolist()
    true_indices = (labels == True).nonzero().tolist()
    gold_pids_list = [[] for _ in range(len(labels))]
    for b_idx, label_idx in true_indices:
        gold_pids_list[b_idx].append(label_idx)
    assert len(gold_pids_list) == len(labels)
    return custome_recall_rate(ranked_pids_list, gold_pids_list)


def custome_recall_rate(
    ranked_pids_list: List[List[int]], gold_pids_list: List[List[int]]
) -> Dict:
    recalls = dict()
    for ranked_pids, gold_pids in zip(ranked_pids_list, gold_pids_list):
        recall = get_recall_rates(ranked_pids=ranked_pids, gold_pids=gold_pids)
        for key, value in recall.items():
            if f"custom{key}" not in recalls:
                recalls[f"custom{key}"] = []
            recalls[f"custom{key}"].append(value)
    # Average the recall
    final_recalls = dict()
    for key, value in recalls.items():
        final_recalls[key] = sum(value) / len(value)
    return final_recalls


def move_first_retrieved_item_to_end(
    pids_in_batch: List[torch.Tensor],
    scores_in_batch: List[torch.Tensor],
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Move the first retrieved item to the end of the list.
    This is for correctly format the input for the evaluation script for BEIR-Arguana.
    - pids_in_batch: shape (batch_size, num_docs)
    - scores_in_batch: shape (batch_size, num_docs)
    """
    new_pids_in_batch: List[torch.Tensor] = []
    new_scores_in_batch: List[torch.Tensor] = []
    for pids, scores in zip(pids_in_batch, scores_in_batch, strict=True):
        new_pids = torch.cat([pids[1:], pids[0:1]], dim=0)
        new_scores = torch.cat(
            [
                scores[1:],
                torch.tensor([0], device=scores.device, dtype=scores.dtype),
            ],
            dim=0,
        )
        new_pids_in_batch.append(new_pids)
        new_scores_in_batch.append(new_scores)
    return new_pids_in_batch, new_scores_in_batch
