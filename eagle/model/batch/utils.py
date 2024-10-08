import copy
from typing import *

import hkkang_utils.list as list_utils
import torch
from torch.nn.utils.rnn import pad_sequence

from eagle.dataset.utils import convert_range_to_scatter, fill_ranges, get_mask


def add_query_ranges_and_mask(
    input_dict: Dict,
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
    q_phrase_scatter_indices = None
    if use_coarse_emb:
        # Create ranges
        q_ranges = fill_ranges(
            phrase_ranges,
            max_len=len(input_dict["q_tok_ids"]),
        )

        # Create q_phrase_mask
        q_phrase_mask = torch.stack(
            [q_tok_mask[start:end].sum(dim=0) > 0 for start, end in q_ranges], dim=0
        ).float()

        # Create scatter indices
        q_phrase_scatter_indices = convert_range_to_scatter(q_ranges)
    # Add extract data to the input_dict
    input_dict["q_phrase_mask"] = q_phrase_mask
    input_dict["q_phrase_scatter_indices"] = q_phrase_scatter_indices

    if return_ranges:
        return input_dict, q_ranges
    return input_dict


def add_doc_ranges_and_mask(
    input_dict: Dict,
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
    doc_phrase_scatter_indices = None
    if use_coarse_emb:
        # Create ranges
        doc_ranges = []
        for i, d_phrase_range in enumerate(phrase_ranges):
            max_len = len(input_dict["doc_tok_ids"][i])
            d_phrase_range = [
                (start, end) for start, end in d_phrase_range if end <= max_len
            ]
            doc_ranges.append(fill_ranges(d_phrase_range, max_len=max_len))
        doc_phrase_scatter_indices = [
            convert_range_to_scatter(item) for item in doc_ranges
        ]
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
    input_dict["doc_phrase_scatter_indices"] = doc_phrase_scatter_indices
    input_dict["doc_phrase_mask"] = doc_phrase_mask

    if return_ranges:
        return input_dict, doc_ranges
    return input_dict


def collate_ranges(ranges: List[torch.Tensor]) -> torch.Tensor:
    # Find the maximum length
    max_idx = max([item[-1] for item in ranges])
    return pad_sequence(ranges, batch_first=True, padding_value=max_idx)


def combined_phrase_ranges_into_one_sentence(
    phrase_ranges_list: List[List[Tuple[int, int]]]
) -> List[Tuple[int, int]]:
    phrase_ranges_list = copy.deepcopy(phrase_ranges_list)
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
