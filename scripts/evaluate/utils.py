import csv
import logging
import os
import sys
from typing import *

import hkkang_utils.file as file_utils

from model import RetrievalResult

logger = logging.getLogger(f"EvalUtils")

csv.field_size_limit(sys.maxsize)


def load_data(
    dataset_dir: str, dataset_name: str, filter_type: str = None, sample_num: int = None
) -> List[Tuple]:
    """
    List[Tuple[str, str, List[str], List[str]]
    - Tuple: (qid, query, pids, p_titles, scores)
    """
    data = load_beir_data(
        dataset_dir=dataset_dir, dataset_name=dataset_name, return_unique=True
    )

    # Sample data
    if sample_num:
        data = data[:sample_num]
        logger.info(f"Sampled data: {len(data)}")

    if filter_type:
        # Filter data that contains the filter_type in the query text
        logger.info(f"Filtering {len(data)} data with {filter_type}...")
        data = [item for item in data if filter_type in item[1]]
        logger.info(f"Filtered data: {len(data)}")

    return data


def load_dataset(
    dataset_dir: str,
    dataset_name: str,
    return_unique: bool = False,
) -> List[Tuple[str, str, List[str], List[str], List[int]]]:
    """Load BEIR data.

    :param dataset_dir: Directory of the dataset
    :type dataset_dir: str
    :param dataset_name: Name of the dataset
    :type dataset_name: str
    :param return_unique: Whether to combine the answers for the same query text, defaults to False
    :type return_unique: bool, optional
    :return: List of tuples (qid, query, pids, p_titles, scores)
    :rtype: List[Tuple[str, str, List[str], List[str], List[int]]]
    """
    # Load query_path
    query_path = os.path.join(dataset_dir, f"{dataset_name}/queries.jsonl")
    data_path = os.path.join(dataset_dir, f"{dataset_name}/dev.jsonl")

    # Load data
    queries = file_utils.read_jsonl_file(query_path)
    query_dict = {str(item["_id"]): item["text"] for item in queries}
    data = file_utils.read_jsonl_file(data_path)

    # Format data
    final_data = []
    skipping_items = []
    for item in data:
        qid = item["id"]
        query = query_dict[str(qid)]
        # Modify pid
        final_data.append([qid, query, item["answers"], [None], [None]])
    if skipping_items:
        logger.info(f"Skipped {len(skipping_items)} items")

    # Combine the answers for the same query text
    if return_unique:
        unique_data = []
        unique_qids = []
        for item in final_data:
            qid, query, pids, p_titles, scores = item
            if qid not in unique_qids:
                unique_data.append(item)
                unique_qids.append(qid)
            else:
                idx = unique_qids.index(qid)
                unique_data[idx][2].extend(pids)
                unique_data[idx][3].extend(p_titles)
                unique_data[idx][4].extend(scores)
        final_data = unique_data

    return final_data


def data_to_beir_format(data: List[Tuple]) -> Dict[str, Dict[str, int]]:
    output = {}
    for qid, query, pids, p_titles, scores in data:
        if qid not in output:
            output[qid] = {}
        for idx, pid in enumerate(pids):
            output[qid][pid] = scores[idx]
    return output


def results_to_beir_format(
    qids: List[str], results: List[List[RetrievalResult]]
) -> Dict[str, Dict[str, float]]:
    assert len(qids) == len(
        results
    ), f"Different number of queries and topk passages: {len(qids)} vs {len(results)}"
    output = {}
    for qid, topk_passages in zip(qids, results):
        if qid not in output:
            output[qid] = {}
        for retrieval_result in topk_passages:
            pid = retrieval_result.doc.id
            if isinstance(pid, int):
                pid = str(pid)
            output[qid][pid] = float(retrieval_result.score)
    return output


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
