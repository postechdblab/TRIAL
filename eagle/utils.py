import logging
import random
from typing import *

import diskcache as dc
import numpy as np
import torch
from beir.retrieval.evaluation import EvaluateRetrieval
from omegaconf import DictConfig, open_dict
from transformers import EvalPrediction

from eagle.metrics.utils import get_recall_rates

# Create a cache object
cache = dc.Cache("/tmp/diskcache")


def set_random_seed(seed: int = 12345) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Decorator to cache function results
def disk_cache():
    def decorator(func):
        def wrapper(*args, **kwargs):
            key = (func.__name__, args, tuple(kwargs.items()))
            if key in cache:
                return cache[key]
            result = func(*args, **kwargs)
            cache[key] = result
            return result

        return wrapper

    return decorator


def handle_old_ckpt(cfg, key) -> Any:
    if key not in cfg:
        return None
    return cfg[key]


def overwrite_config(src: DictConfig, dst: DictConfig) -> DictConfig:
    """Overwrite the src config with the dst config."""
    with open_dict(src):
        for key, value in dst.items():
            if key in src:
                if isinstance(value, DictConfig):
                    src[key] = overwrite_config(src[key], value)
                else:
                    src[key] = value
    return src


def add_config(cfg: DictConfig, key: str, value: Any) -> DictConfig:
    with open_dict(cfg):
        cfg[key] = value
    return cfg


def add_global_configs(
    cfg: DictConfig, global_dic: DictConfig = None, exclude_keys: List[str] = None
) -> DictConfig:
    if global_dic is None:
        assert hasattr(cfg, "_global"), "Global configs are not found in the config"
        global_dic = cfg._global

    with open_dict(cfg):
        for sub_cfg_name, sub_cfg in cfg.items():
            if sub_cfg_name != "global" and isinstance(sub_cfg, DictConfig):
                if exclude_keys and sub_cfg_name in exclude_keys:
                    continue
                # Append key to the sub_cfg
                for key, value in global_dic.items():
                    if key not in sub_cfg:
                        sub_cfg[key] = value
                # Recursively add global configs
                add_global_configs(sub_cfg, global_dic=global_dic)
    return cfg


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

    # Combine metrics into one dictionary
    combined_metrics = ndcg | _map | recall | precision | mrr | acc | custom_recall
    if prefix is not None:
        combined_metrics = {
            f"{prefix}_{key}": value for key, value in combined_metrics.items()
        }
    return combined_metrics


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


def remove_key_with_none_value(dic: Dict) -> Dict:
    return {key: value for key, value in dic.items() if value is not None}
