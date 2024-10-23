import copy
import functools
import math
import os
import pickle
from typing import *

import hkkang_utils.file as file_utils
import hkkang_utils.list as list_utils
import lz4.frame as lz4f
import numpy as np
import torch
import tqdm

from eagle.utils import disk_cache


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
        pass
        # logger.info(f"Skipped {len(skipping_items)} items")

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


def is_token_included(src: set[int], target: List[int]) -> List[bool]:
    src_set = torch.tensor(list(src))
    isin = torch.isin(torch.tensor(target), src_set)
    return [False, False] + isin.tolist()[2:-1] + [False]


def convert_range_to_scatter(range_indices: List[Tuple[int, int]]) -> List[int]:
    """Convert range indices to scatter indices.
    :param range_indices: List of range indices
    :type range_indices: List[Tuple[int, int]]
    :return: List of scatter indices
    :rtype: List[int]
    """
    scatter_indices: List[int] = []
    last_seen_idx = 0
    target_idx = 0
    # Add indices
    for start, end in range_indices:
        # If range indices are not continuous, add the missing indices
        if start > last_seen_idx:
            for idx in range(last_seen_idx, start):
                scatter_indices.append(target_idx)
                target_idx += 1

        # Add the indices
        new_indices = [target_idx] * (end - start)
        scatter_indices.extend(new_indices)

        # Update state
        target_idx += 1
        last_seen_idx = end
    assert (
        len(scatter_indices) == range_indices[-1][-1]
    ), f"Length mismatch: {len(scatter_indices)} vs {range_indices[-1][-1]}"
    return scatter_indices


def add_padding_for_ranges(
    ranges: List[Tuple[int, int]], max_len: int
) -> List[Tuple[int, int]]:
    # Get the last index
    last_idx = ranges[-1][-1] if ranges else 0
    assert (
        last_idx <= max_len
    ), f"Last index is larger than max_len: {last_idx} vs {max_len}"
    # Add padding
    if last_idx < max_len:
        ranges.extend([(i, i + 1) for i in range(last_idx, max_len)])
    return ranges


def validate_ranges(
    ranges: List[Tuple[int, int]], max_len: int = None
) -> List[Tuple[int, int]]:
    if max_len is not None:
        assert (
            ranges[-1][-1] <= max_len
        ), f"Last index is larger than max_len: {ranges[-1][-1]} vs {max_len}"
    for i in range(0, len(ranges) - 1):
        assert (
            ranges[i][-1] <= ranges[i + 1][0]
        ), f"Ranges are not continuous: {ranges[i]} vs {ranges[i+1]}"
    return True


def fix_ranges(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    new_ranges = []
    last_end = 0
    for start, end in ranges:
        if start < last_end:
            # Append to the last range
            new_ranges[-1] = (new_ranges[-1][0], end)
        else:
            new_ranges.append((start, end))
        last_end = end
    return new_ranges


def fill_ranges(
    ranges: List[Tuple[int, int]], max_len: int
) -> List[List[Tuple[int, int]]]:
    if len(ranges) == 0:
        new_ranges = [(i, i + 1) for i in range(max_len)]
    else:
        assert (
            ranges[-1][-1] <= max_len
        ), f"Last index is larger than max_len: {ranges[-1][-1]} vs {max_len}"
        new_ranges = [(i, i + 1) for i in range(0, ranges[0][0])]
        for start, end in ranges:
            # Append
            if len(new_ranges) == 0:
                new_ranges.extend([(i, i + 1) for i in range(0, ranges[0][0])])
            else:
                last_end = new_ranges[-1][-1]
                if last_end != start:
                    new_ranges.extend([(i, i + 1) for i in range(last_end, start)])
            # Append
            new_ranges.append((start, end))
        # End range
        new_ranges.extend([(i, i + 1) for i in range(ranges[-1][-1], max_len)])
    return new_ranges


def combine_word_phrase_ranges(
    word_ranges: List[Tuple[int, int]], phrase_ranges: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    phrase_ranges = [item for item in phrase_ranges if item[1] - item[0] > 1]
    word_ranges = copy.deepcopy(word_ranges)
    combined: List[Tuple[int, int]] = []
    used_word_indices = set()
    for phrase_i in range(len(phrase_ranges)):
        phrase_start, phrase_end = phrase_ranges[phrase_i]

        # While there is a word that satisfies the condition
        for word_i in range(len(word_ranges)):
            word_start, word_end = word_ranges[word_i]

            # Pass if the word is already used
            if word_i in used_word_indices:
                continue

            # Add the word if it is before the phrase
            if word_end <= phrase_start:
                combined.append((word_start, word_end))
                used_word_indices.add(word_i)
                continue
            elif word_start >= phrase_start and word_end <= phrase_end:
                used_word_indices.add(word_i)
                continue

            if (
                word_start >= phrase_start
                and word_start < phrase_end
                and word_end > phrase_end
            ):
                phrase_end = word_end
                used_word_indices.add(word_i)
                break
            elif word_start <= phrase_start:
                if word_end <= phrase_end:
                    # Expand phrase (in front)
                    phrase_start = word_start
                elif word_end > phrase_start:
                    # Expand phrase (in back)
                    phrase_end = word_end
                used_word_indices.add(word_i)
                break
        combined.append((phrase_start, phrase_end))
    # Validate ranges
    combined = fix_ranges(combined)
    # Append words that are later than the last phrase
    last_phrase_end = combined[-1][-1] if combined else 0
    for word_start, word_end in word_ranges:
        if word_start >= last_phrase_end:
            combined.append((word_start, word_end))
    # Validate ranges
    validate_ranges(combined)
    return combined


def extract_word_range_with_multi_tokens(
    toks: List[str], split_prefix: str = "##"
) -> List[Tuple[int, int]]:
    word_indices: List[Tuple[int, int]] = []

    stack = []
    for i, tok in enumerate(toks):
        # Update stack if it is not a splitted token
        if not tok.startswith(split_prefix):
            if stack:
                word_indices.append((stack[0], i))
                stack = []
        stack.append(i)
    # Flush stack if not empty
    if stack:
        word_indices.append((stack[0], len(toks)))
    # Get the word that are splitted into multiple tokens
    return [indices for indices in word_indices if indices[1] - indices[0] > 1]


def save_compressed(file_path: str, data: Any) -> None:
    with lz4f.open(file_path, mode="wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    return None


def read_compressed(file_path: str) -> Any:
    with lz4f.open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def split_list(a_list: List, chunk_size: int) -> List[List]:
    return [a_list[i : i + chunk_size] for i in range(0, len(a_list), chunk_size)]


def read_corpus(path: str) -> Dict[str, str]:
    corpus: List[Dict] = file_utils.read_json_file(path, auto_detect_extension=True)
    corpus: Dict[str, str] = {str(doc["_id"]): doc["text"] for doc in corpus}
    return corpus


def read_queries(path: str) -> Dict[str, str]:
    queries: List[Dict] = file_utils.read_json_file(path, auto_detect_extension=True)
    queries: Dict[str, str] = {query["_id"]: query["text"] for query in queries}
    return queries


def read_qrels_qids(path: str) -> List[str]:
    qrels = file_utils.read_csv_file(path, delimiter="\t", first_row_as_header=True)
    return [item["query-id"] for item in qrels]


def get_mask(input_ids: torch.Tensor, skip_ids: List[int]) -> torch.Tensor:
    """
    Mask all the tokens in the skiplist.
    Set to 0 if the token needs to be computed.
    Set to 1 if the token needs to be masked.
    """
    # Convert skip_ids list to a tensor for comparison
    skip_ids_tensor = torch.tensor(skip_ids, device=input_ids.device)

    # Create mask: 1 if the token is in skip_ids, else 0
    mask = torch.isin(input_ids, skip_ids_tensor).float()
    return mask


def get_att_mask(input_ids: torch.Tensor, skip_ids: List[int]) -> torch.Tensor:
    """
    Mask all the tokens in the skiplist.
    Set to 1 if the token needs to be computed.
    Set to 0 if the token needs to be masked.
    """
    return (1 - get_mask(input_ids, skip_ids)).float()


@functools.lru_cache(maxsize=None)
def get_labels(bsize: int, neg_num: int) -> np.ndarray:
    return np.repeat(
        np.array([True] + [False] * neg_num, dtype=bool).reshape(1, -1), bsize, axis=0
    )


@disk_cache()
def get_indices_to_avoid_repeated_qids_in_minibatch(
    qids: List[int], batch_size: int
) -> List[int]:
    """Returns a list of indices to avoid repeated qids in the minibatch."""

    def get_item(dic: Dict, get_unique_qid: bool = False) -> int:
        """Return the qid that appears the most in the dictionary."""
        max_key = 1 if get_unique_qid else max(dic.keys())
        qid = dic[max_key].pop(0)
        if len(dic[max_key]) == 0:
            dic.pop(max_key)
        if max_key > 1:
            # Append the qid back to the dictionary with a count of max_key-1
            if max_key - 1 in dic:
                dic[max_key - 1].append(qid)
            else:
                dic[max_key - 1] = [qid]
        return qid

    # List of indices of the qids
    qid_indices = {}
    for i, qid in enumerate(qids):
        if qid not in qid_indices:
            qid_indices[qid] = [i]
        else:
            qid_indices[qid].append(i)
    # Create dictionary that counts the number of times each qid appears
    dic: Dict[int, int] = {}
    for qid in qids:
        dic[qid] = dic.get(qid, 0) + 1

    # Inverted index of the dictionary
    new_dic: Dict[int, int] = {}
    for key, value in dic.items():
        if value not in new_dic:
            new_dic[value] = [key]
        else:
            new_dic[value].append(key)

    indices = []
    for i in tqdm.tqdm(
        range(0, math.ceil(len(qids) // batch_size)), desc="Shuffling train indices"
    ):
        # Add the index of the qid to the list of indices
        get_unique_qid = False
        tmp = []
        for _ in range(i * batch_size, min((i + 1) * batch_size, len(qids))):
            qid = get_item(new_dic, get_unique_qid)
            indices.append(qid_indices[qid].pop())
            get_unique_qid = True
            if qid in tmp:
                stop = 1
            tmp.append(qid)
    # Append the remaining indices
    for qid, item_indices in qid_indices.items():
        if dic:
            indices.extend(item_indices)
    return indices


def combine_splitted_tok_ids(
    tok_ids_list: List[List[int]],
) -> Tuple[List[int], List[int]]:
    """Combine splitted list of tok ids (i.e., sentences) into one list of tok ids (i.e., one text)"""
    # This needs to be changed when the adding of the special tokens as prefix changes
    BEGIN_SPECIAL_TOK_NUM = 2
    sent_start_indices = []
    # Remove special tokens in front, except the first sentence
    new_tok_ids_list: List[List[int]] = [
        tok_ids if idx == 0 else tok_ids[BEGIN_SPECIAL_TOK_NUM:]
        for idx, tok_ids in enumerate(tok_ids_list)
    ]
    # Combine sentences
    combined_tok_ids = []
    for tok_ids in new_tok_ids_list:
        # Add the start indices
        sent_start_indices.append(len(combined_tok_ids))
        # Add tokens for the current sentence
        combined_tok_ids.extend(tok_ids)

    return combined_tok_ids, sent_start_indices


def combined_phrase_ranges_into_one_sentence(
    phrase_ranges_list: List[List[Tuple[int, int]]]
) -> None:
    SPECIAL_TOK_NUM = 2
    # Figure out the number to add for each sentences
    number_to_adds = []
    for sent_idx, p_ranges in enumerate(phrase_ranges_list):
        if sent_idx == 0:
            number_to_adds.append(0)
            continue
        # Get the last number to have the cumulative number
        base_number = number_to_adds[-1]
        # Add the max value of the previous sentence
        previous_token_cnt = phrase_ranges_list[sent_idx - 1][-1][-1]
        # Remove the first two special tokens if the previous sentence is not the first sentence
        if sent_idx > 0:
            previous_token_cnt = previous_token_cnt - SPECIAL_TOK_NUM
        number_to_adds.append(base_number + previous_token_cnt)

    # Modify the token ranges
    for sent_idx, (number_to_add, p_ranges) in enumerate(
        zip(number_to_adds, phrase_ranges_list)
    ):
        # Create new p_ranges

        # Remove the ranges for the first two tokens
        if sent_idx != 0:
            p_ranges = p_ranges[SPECIAL_TOK_NUM:]
        # Modify the token idx
        p_ranges = [(s + number_to_add, e + number_to_add) for s, e in p_ranges]

        # update the data
        phrase_ranges_list[sent_idx] = p_ranges

    # Flatten list of list to list
    phrase_ranges_list = list_utils.do_flatten_list(phrase_ranges_list)

    return phrase_ranges_list
