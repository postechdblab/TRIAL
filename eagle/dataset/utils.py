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
from torch.nn.utils.rnn import pad_sequence

from eagle.tokenizer import NewTokenizer
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


def add_query_ranges_and_mask(
    input_dict: Dict,
    word_ranges: List[Tuple[int, int]],
    phrase_ranges: List[Tuple[int, int]],
    skip_ids: List[int],
    use_coarse_emb: bool,
    return_ranges: bool = False,
) -> Dict:
    # Create token mask
    q_tok_mask = get_mask(
        input_ids=torch.tensor(
            input_dict["q_tok_ids"], dtype=torch.int64, device="cpu"
        ),
        skip_ids=skip_ids,
    )
    input_dict["q_tok_mask"] = q_tok_mask

    # Create new q_tok_mask
    q_phrase_mask = None
    q_scatter_indices = None
    if use_coarse_emb:
        # Create ranges
        q_ranges = fill_ranges(
            combine_word_phrase_ranges(word_ranges, phrase_ranges),
            max_len=len(input_dict["q_tok_ids"]),
        )

        # Create q_phrase_mask
        q_phrase_mask = torch.stack(
            [q_tok_mask[start:end].sum(dim=0) > 0 for start, end in q_ranges], dim=0
        ).float()

        # Create scatter indices
        q_scatter_indices = convert_range_for_scatter(q_ranges)
    # Add extract data to the input_dict
    input_dict["q_phrase_mask"] = q_phrase_mask
    input_dict["q_scatter_indices"] = q_scatter_indices

    if return_ranges:
        return input_dict, q_ranges
    return input_dict


def add_doc_ranges_and_mask(
    input_dict: Dict,
    word_ranges: List[List[Tuple[int, int]]],
    phrase_ranges: List[List[Tuple[int, int]]],
    skip_ids: List[int],
    use_coarse_emb: bool,
    return_ranges: bool = False,
) -> Dict:
    # Create token mask
    doc_ids = pad_sequence(
        [
            torch.tensor(item, dtype=torch.int64, device="cpu")
            for item in input_dict["doc_tok_ids"]
        ],
        batch_first=True,
    )
    doc_tok_mask = get_mask(input_ids=doc_ids, skip_ids=skip_ids)
    input_dict["doc_tok_mask"] = doc_tok_mask
    # Create doc_phrase_mask
    doc_phrase_mask = None
    doc_scatter_indices = None
    if use_coarse_emb:
        # Create ranges
        doc_ranges = []
        assert len(word_ranges) == len(
            phrase_ranges
        ), f"Word and phrase ranges are not consistent: {len(word_ranges)} vs {len(phrase_ranges)}"
        for i, (d_word_range, d_phrase_range) in enumerate(
            zip(word_ranges, phrase_ranges)
        ):
            max_len = len(input_dict["doc_tok_ids"][i])
            d_word_range = [
                (start, end) for start, end in d_word_range if end <= max_len
            ]
            d_phrase_range = [
                (start, end) for start, end in d_phrase_range if end <= max_len
            ]
            doc_ranges.append(
                fill_ranges(
                    combine_word_phrase_ranges(d_word_range, d_phrase_range),
                    max_len=max_len,
                )
            )
        doc_scatter_indices = [convert_range_for_scatter(item) for item in doc_ranges]
        new_doc_tok_mask = []
        for i, items in enumerate(doc_ranges):
            new_doc_tok_mask.append(
                torch.stack(
                    [doc_tok_mask[i, start:end].sum(dim=0) > 0 for start, end in items],
                    dim=0,
                )
            )
        doc_phrase_mask = pad_sequence(new_doc_tok_mask, batch_first=True).float()
    # Append extract data to the input_dict
    input_dict["doc_scatter_indices"] = doc_scatter_indices
    input_dict["doc_phrase_mask"] = doc_phrase_mask

    if return_ranges:
        return input_dict, doc_ranges
    return input_dict


def collate_ranges(ranges: List[torch.Tensor]) -> torch.Tensor:
    # Find the maximum length
    max_idx = max([item[-1] for item in ranges])
    return pad_sequence(ranges, batch_first=True, padding_value=max_idx)


def convert_range_for_scatter(range_indices: List[Tuple[int, int]]) -> List[List[int]]:
    converted = []
    # Add indices
    for target_idx, (start, end) in enumerate(range_indices):
        new_indices = [target_idx] * (end - start)
        converted.extend(new_indices)
    return converted


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


def preprocess(example: Dict, unbatch: bool = False, *args, **kwargs) -> Dict:
    if type(example["q_texts"]) == list:
        result = preprocess_batch(example, *args, **kwargs)
    else:
        result = preprocess_nobatch(example, *args, **kwargs)
    if unbatch:
        result["q_ids"] = result["q_ids"][0]
        result["q_tok_ids"] = result["q_tok_ids"][0]
        result["q_tok_att_mask"] = result["q_tok_att_mask"][0]
        if result["doc_tok_ids"] is not None:
            result["doc_tok_ids"] = result["doc_tok_ids"][0]
            result["doc_tok_att_mask"] = result["doc_tok_att_mask"][0]
    return result


def preprocess_nobatch(
    example: Dict,
    q_tokenizer: NewTokenizer,
    d_tokenizer: NewTokenizer,
    is_eval: bool = False,
    is_compress: bool = True,
) -> Dict:
    example["q_texts"] = [example["q_texts"]]
    example["pos_doc_text_list"] = [example["pos_doc_text_list"]]
    example["neg_doc_texts_list"] = [example["neg_doc_texts_list"]]
    return preprocess_batch(
        example_batch=example,
        q_tokenizer=q_tokenizer,
        d_tokenizer=d_tokenizer,
        is_eval=is_eval,
        is_compress=is_compress,
    )


def preprocess_batch(
    example_batch: Dict,
    q_tokenizer: NewTokenizer,
    d_tokenizer: NewTokenizer,
    is_eval: bool = False,
    is_compress: bool = True,
) -> Dict:
    """
    The input example_batch is a dictionary with the following keys
    - q_texts: List[str]
    - pos_doc_texts_list: List[str]
    - neg_doc_texts_list: List[List[str]]
    """
    # Aggregate doc texts for each query
    bsize = len(example_batch["q_texts"])
    neg_num = None
    pos_num = None
    doc_texts = []
    assert len(example_batch["q_texts"]) == len(
        example_batch["pos_doc_texts_list"]
    ), f"Query and doc size is not consistent: {len(example_batch['q_texts'])} vs {len(example_batch['pos_doc_text_list'])}"
    for pos_doc_text, neg_doc_text_list in zip(
        example_batch["pos_doc_texts_list"], example_batch["neg_doc_texts_list"]
    ):
        if neg_num is None:
            neg_num = len(neg_doc_text_list)
        assert neg_num == len(
            neg_doc_text_list
        ), f"Neg num is not consistent: {neg_num} vs {len(neg_doc_text_list)}"
        if pos_num is None:
            pos_num = len(pos_doc_text)
        if pos_doc_text:
            doc_texts.extend(pos_doc_text)
        for neg_doc in neg_doc_text_list:
            if neg_doc:
                doc_texts.append(neg_doc)

    # Tokenize text
    q_ids = example_batch["q_ids"]
    q_tokens = q_tokenizer(example_batch["q_texts"])
    doc_tokens = d_tokenizer(doc_texts) if doc_texts else None

    # Check if the tokens are expressible in uint16
    is_compress = 2**15 > len(q_tokenizer) and 2**15 > len(
            d_tokenizer
        )

    if is_compress:
        q_token_ids = [np.uint16(item) for item in q_tokens["input_ids"]]
        q_token_att_mask = [
            np.array(item, dtype=bool) for item in q_tokens["attention_mask"]
        ]

        if doc_tokens:
            doc_tok_ids = split_list(
                [np.uint16(item) for item in doc_tokens["input_ids"]], neg_num + pos_num
            )
            doc_tok_att_mask = split_list(
                [np.array(item, dtype=bool) for item in doc_tokens["attention_mask"]],
                neg_num + pos_num,
            )
        else:
            doc_tok_ids = None
            doc_tok_att_mask = None
    else:
        # Split doc tokens
        if doc_tokens:
            doc_tok_ids = split_list(doc_tokens["input_ids"], neg_num + 1)
            doc_tok_att_mask = split_list(doc_tokens["attention_mask"], neg_num + 1)
        else:
            doc_tok_ids = None
            doc_tok_att_mask = None
        q_token_ids = q_tokens["input_ids"]
        q_token_att_mask = q_tokens["attention_mask"]

    result = {
            "q_id": q_ids,
            "q_tok_ids": q_token_ids,
            "q_tok_att_mask": q_token_att_mask,
            "doc_tok_ids": doc_tok_ids,
            "doc_tok_att_mask": doc_tok_att_mask,
            "pos_doc_ids": example_batch["pos_doc_ids_list"],
            "neg_doc_ids": example_batch["neg_doc_ids_list"]
        }

    if is_eval:
        result["labels"] = get_labels(bsize=bsize, neg_num=neg_num)
    return result


def get_mask(input_ids: torch.Tensor, skip_ids: List[int]) -> List[int]:
    """
    Mask all the tokens in the skiplist.
    Set to 1 if the token needs to be computed.
    Set to 0 if the token needs to be masked.
    """
    # Create mask
    mask = torch.ones_like(input_ids, requires_grad=False)
    for skip_id in skip_ids:
        # 1 if the token is not in the skip_ids
        mask *= input_ids != skip_id
    return mask.float()


def collate_fn(input_dics: List[Dict]) -> Dict:
    """Collate list of dictionaries into a single dictionary."""

    def get_dtype(key: str) -> torch.dtype:
        if "mask" in key or key == "labels":
            return torch.bool
        elif "id" in key:
            return torch.int32
        return torch.long

    new_dict = {}
    # Assume all dictionaries have the same keys
    keys = list(input_dics[0].keys())
    # Collate for each key
    for key in keys:
        if input_dics[0][key] is None:
            new_dict[key] = None
            continue
        if key == "q_scatter_indices":
            padded_values = collate_ranges(
                [
                    torch.tensor(dic[key], dtype=get_dtype(key), device="cpu")
                    for dic in input_dics
                ]
            )
        elif key == "doc_scatter_indices":
            padded_values = list_utils.do_flatten_list(
                [input_dic[key] for input_dic in input_dics]
            )
            padded_values = collate_ranges(
                [
                    torch.tensor(item, dtype=get_dtype(key), device="cpu")
                    for item in padded_values
                ]
            )
        elif key in ["q_tok_ids", "q_tok_att_mask", "labels"]:
            values = [
                torch.tensor(dic[key], dtype=get_dtype(key), device="cpu")
                for dic in input_dics
            ]
            padded_values = pad_sequence(values, batch_first=True)
        elif key in ["doc_tok_ids", "doc_tok_att_mask"]:
            values = []
            for input_dic in input_dics:
                for item in input_dic[key]:
                    values.append(
                        torch.tensor(item, dtype=get_dtype(key), device="cpu")
                    )
            padded_values = pad_sequence(values, batch_first=True)
            padded_values = padded_values.reshape(
                len(input_dics), -1, padded_values.shape[1]
            )
        elif key in ["q_tok_mask", "q_phrase_mask"]:
            values = [dic[key].clone().detach().unsqueeze(-1) for dic in input_dics]
            padded_values = (pad_sequence(values, batch_first=True) == 0)
        elif key in ["doc_tok_mask", "doc_phrase_mask"]:
            values = list_utils.do_flatten_list(
                [torch.unbind(dic[key].clone().detach()) for dic in input_dics]
            )
            padded_values = (pad_sequence(values, batch_first=True).unsqueeze(-1) == 0)
        elif key == "fine_grained_label":
            values = []
            for dic in input_dics:
                for item in dic[key]:
                    values.append(torch.tensor(item, dtype=torch.bool, device="cpu"))
            padded_values = pad_sequence(values, batch_first=True).float()
            summed_padded_values = padded_values.sum(dim=1, keepdim=True)
            mask_padded_values = (summed_padded_values > 0).squeeze()
            padded_values = (
                padded_values[mask_padded_values]
                / summed_padded_values[mask_padded_values]
            )
            new_dict["fine_grained_label_mask"] = mask_padded_values
        elif key in ["q_id", "pos_doc_idxs"]:
            padded_values = [dic[key] for dic in input_dics]
        elif key in ["pos_doc_ids", "neg_doc_ids"]:
            continue
        else:
            raise ValueError(f"Unsupported key: {key}")
        new_dict[key] = padded_values

    return new_dict


@functools.lru_cache(maxsize=None)
def get_labels(bsize: int, neg_num: int) -> np.ndarray:
    return np.repeat(
        np.array([True] + [False] * neg_num, dtype=bool).reshape(1, -1), bsize, axis=0
    )


# TODO: Create disk-based cache here
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
