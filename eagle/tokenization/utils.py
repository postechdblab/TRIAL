from typing import *


def combine_splitted_tok_ids(
    tok_ids_list: List[List[int]], begin_special_tok_num: int = 2
) -> Tuple[List[int], List[int]]:
    """Combine splitted list of tok ids (i.e., sentences) into one list of tok ids (i.e., one text)"""
    # This needs to be changed when the adding of the special tokens as prefix changes
    sent_start_indices = []
    # Remove special tokens in front, except the first sentence
    new_tok_ids_list: List[List[int]] = [
        tok_ids if idx == 0 else tok_ids[begin_special_tok_num:]
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
