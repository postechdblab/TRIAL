from typing import *

import torch
from torch.nn.utils.rnn import pad_sequence

from eagle.dataset.utils import convert_range_to_scatter, fill_ranges, get_mask


def cut_off_phrase_ranges_by_max_len(
    phrase_ranges: List[Tuple[int, int]], max_len: int
) -> List[Tuple[int, int]]:
    new_phrase_ranges = []
    for i, (start, end) in enumerate(phrase_ranges):
        if start >= max_len:
            break
        if end > max_len:
            new_phrase_ranges.append((start, max_len))
            break
        new_phrase_ranges.append((start, end))
    return new_phrase_ranges


def zero_pad_1d_tensor(data: List[torch.Tensor]) -> torch.Tensor:
    return torch.nn.utils.rnn.pad_sequence(data, batch_first=True)


def zero_pad_2d_tensor(data: List[List[torch.Tensor]]) -> List[torch.Tensor]:
    return [torch.nn.utils.rnn.pad_sequence(item, batch_first=True) for item in data]


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
