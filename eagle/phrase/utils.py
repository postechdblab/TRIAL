import copy
import os
import string
from typing import *

import hkkang_utils.list as list_utils
import torch

from eagle.tokenization.tokenizer import Tokenizer

SPLIT_DIR_NAME = "splitted"


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


def fix_bad_index_ranges(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Fix the bad ranges in the given ranges.
    :param ranges: list of ranges
    :type ranges: List[Tuple[int, int]]
    :return: list of ranges with the bad ranges fixed
    :rtype: List
    """
    new_ranges = []
    for start, end in ranges:
        if start >= end:
            continue
        new_ranges.append((start, end))
    return new_ranges


def fill_in_missing_phrase_ranges(
    ranges: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """Fill in the missing ranges in the given ranges.
    :param ranges: list of ranges
    :type ranges: List[Tuple[int, int]]
    :return: list of ranges with the missing ranges filled in
    :rtype: List
    """
    last_seen = 0
    new_ranges = []
    for start, end in ranges:
        # Add the missing ranges
        if start > last_seen:
            for i in range(last_seen, start):
                new_ranges.append((i, i + 1))
        # Add the current range
        new_ranges.append((start, end))
        # Update state
        last_seen = end
    return new_ranges


def remove_file_name_from_path(path: str) -> str:
    return os.path.join("/", *[item for item in path.split("/")[:-1] if item])


def get_partial_data_name(
    dir_path: str, file_name: str, total_proc_num: int, i: int
) -> str:
    if total_proc_num > 1:
        path = os.path.join(dir_path, f"{file_name}.{i}_{total_proc_num}")
    else:
        path = os.path.join(dir_path, file_name)
    return path


def get_output_file_name(
    prefix: str, total_process_num: int, process_idx: int = None
) -> str:
    return (
        f"phrase_indices.{prefix}.pkl.{process_idx}_{total_process_num}"
        if total_process_num > 1
        else f"phrase_indices.{prefix}.pkl"
    )


def get_tokenized_path(tokenizer: Tokenizer, dir_path: str, filename: str) -> str:
    return os.path.join(dir_path, f"{filename}.{tokenizer.model_name}-tok.cache")


def _split_into_batches2(items: List[Any], bsize: int) -> List[List[Any]]:
    batches = []
    for offset in range(0, len(items), bsize):
        batches.append(items[offset : offset + bsize])

    return batches


def get_range_of_phrases_in_token_level(
    token_range_in_char: List[Tuple[int, int]],
    noun_phrase_range_in_char: List[Tuple[int, int]],
    offset: int = 0,
    padding: int = 0,
    max_token_len: int = 0,
    is_partial: bool = False,
) -> List[Tuple[int, int]]:
    def append_phrase_indices(
        all_indices: List, start_tok_idx: int, end_tok_idx: int
    ) -> bool:
        """Append the phrase indices in token-level to the list"""
        real_start = start_tok_idx + offset
        real_end = end_tok_idx + offset
        # Validate input: start and end are found
        assert (
            real_start != None and real_end != None
        ), f"real_start:{real_start} real_end:{real_end}"
        # Validate input: start is less than end
        if real_start >= real_end:
            raise ValueError(
                f"real_start={real_start} >= real_end={real_end}. \ntoken_range_in_char:{token_range_in_char}\nnoun_phrase_range_in_char:{noun_phrase_range_in_char}"
            )
        # Validate input: check if the phrase is out of the max token length
        if max_token_len:
            max_tmp = max_token_len - padding
            if real_end > max_tmp:
                return False
        # Check if the phrase should be concatenated with the previous phrase
        if len(all_indices) > 0 and real_start < all_indices[-1][1]:
            all_indices[-1] = (all_indices[-1][0], real_end)
            return True
        else:
            # Add the new phrase
            all_indices.append((real_start, real_end))
        return True

    token_idx = 0
    all_phrase_indices_in_tok = (
        [] if is_partial else [(i, i + 1) for i in range(offset)]
    )
    for phrase_idx, (p_start_in_char, p_end_in_char) in enumerate(
        noun_phrase_range_in_char
    ):
        # Skip the phrase if it is out of max token range
        if (
            max_token_len and offset + token_idx >= max_token_len - padding
        ) or token_idx >= len(token_range_in_char):
            break

        # If the phrase does not cover all the tokens in the given text
        # Skip to the token_idx that is greater than the start of the phrase
        while (
            token_idx < len(token_range_in_char)
            and token_range_in_char[token_idx][1] <= p_start_in_char
        ):
            token_idx += 1

        if token_idx == len(token_range_in_char):
            # TODO: Need to shorten the phrase indices
            break

        # Get the current token start and end indices in character level
        t_start_in_char, t_end_in_char = token_range_in_char[token_idx]

        # Find the phrase start in token-level
        # Find the token start that is equal or less than the phrase start
        if t_start_in_char <= p_start_in_char:
            phrase_start_in_tok = token_idx
        else:
            # If not, connect with the previous token
            # Loop backward to find the start of the phrase
            phrase_start_in_tok = 0
            for i in range(token_idx, -1, -1):
                t_start, t_end = token_range_in_char[i]
                if t_start <= p_start_in_char:
                    phrase_start_in_tok = i
                    break
        # Find the phrase end in token-level
        if t_end_in_char >= p_end_in_char:
            phrase_end_in_tok = token_idx + 1
        else:
            # If not, connect with the next token
            # Loop forward to find the end of the phrase
            phrase_end_in_tok = len(token_range_in_char)
            for i in range(token_idx, len(token_range_in_char)):
                t_start, t_end = token_range_in_char[i]
                if t_end >= p_end_in_char:
                    phrase_end_in_tok = i + 1
                    break
        # Append to the phrase indices
        append_phrase_indices(
            all_phrase_indices_in_tok, phrase_start_in_tok, phrase_end_in_tok
        )

    if padding and not is_partial:
        last_idx = all_phrase_indices_in_tok[-1][1]
        all_phrase_indices_in_tok.extend(
            [(i + last_idx, i + last_idx + 1) for i in range(padding)]
        )
    return all_phrase_indices_in_tok


def get_range_of_tokens_in_char_level(
    tokens: List[str], sentence: str
) -> List[Tuple[int, int]]:
    """Get the start and end character indices for the tokens in the sentence.

    :param sentence: target sentence
    :type sentence: str
    :param tokens: list of tokens in the sentence
    :type tokens: List[str]
    :return: list of tuples of start and end character indices for the tokens
    :rtype: List[Tuple[int, int]]
    """
    char_idx = 0
    token_idx = 0
    found_indices = []
    while token_idx < len(tokens) and char_idx < len(sentence):
        # Check if the token is a word
        token = tokens[token_idx]
        # Pass empty space in the sentence
        while char_idx < len(sentence) and sentence[char_idx] in string.whitespace:
            char_idx += 1
        if token == sentence[char_idx : char_idx + len(token)]:
            found_indices.append((char_idx, char_idx + len(token)))
            char_idx += len(token)
        elif (
            token.startswith("##")
            and token[2:] == sentence[char_idx : char_idx + len(token) - 2]
        ):
            found_indices.append((char_idx, char_idx + len(token) - 2))
            char_idx += len(token) - 2
        else:
            # Assume that the whole word until space or punctuation is the token
            if token == "[unk]":
                stop_idx = char_idx + 1
                special_char_found = False
                for idx in range(char_idx + 1, len(sentence)):
                    if (
                        sentence[idx] in string.whitespace
                        or sentence[idx] in string.punctuation
                    ):
                        stop_idx = idx
                        special_char_found = True
                        break

                # If the token is the last token in the sentence, then stop_idx is the end of the sentence
                if not special_char_found:
                    stop_idx = len(sentence)
                    assert (
                        token_idx == len(tokens) - 1
                    ), f"token_idx={token_idx} != len(tokens)-1={len(tokens)-1} for {sentence}"

                found_indices.append((char_idx, stop_idx))
                char_idx = stop_idx
            else:
                raise ValueError(
                    f"Case2: token_idx:{token_idx} char_idx:{char_idx} Cannot find {token} in {sentence}"
                )
        token_idx += 1
    return found_indices


def get_phrase_indices_by_toks(
    input_toks: List[List[str]],
    input_texts: List[str],
    parsed_texts: List[Text],
    has_special_tokens: bool = True,
    max_token_len: int = 0,
    all_noun_only: bool = False,
    noun_only: bool = False,
    prop_noun_only: bool = False,
    named_entity_only: bool = False,
) -> List[List[Tuple]]:
    # Remove the special tokens (i.e., [CLS], [unused0], [unused1] and [SEP])
    if has_special_tokens:
        # Remove the special tokens
        input_toks = [toks[2:-1] for toks in input_toks]
        offset = 2
        padding = 1
    else:
        offset = 0
        padding = 0

    # Find the character indices for the tokens
    char_indices: List[List[Tuple[int, int]]] = []
    for b_size in range(len(input_texts)):
        # try:
        tmp_indices = get_range_of_tokens_in_char_level(
            [tok.lower() for tok in input_toks[b_size]], input_texts[b_size].lower()
        )
        # except:
        #     print(f"Passing {b_size}-th item. Mismatch with tokenized results.")
        #     tmp_indices = []
        char_indices.append(tmp_indices)
    # validates = [validate(input_toks[b_size][2:-1], char_indices[b_size], input_texts[b_size]) for b_size in range(len(input_texts))]
    # assert all(validates), f"False idx={[(idx, input_texts[idx]) for idx, item in enumerate(validates) if not item]}"

    # Get the noun phrase indices in character ids
    all_phrase_indices: List[List[Tuple[int, int]]] = []
    for item in parsed_texts:
        if noun_only:
            indices = item.noun_phrase_indices
        elif prop_noun_only:
            indices = item.prop_noun_phrase_indices
        elif named_entity_only:
            indices = item.named_entity_indices
        elif all_noun_only:
            indices = item.all_noun_phrase_indices
        else:
            indices = item.phrase_indices
        all_phrase_indices.append(indices)

    # Get phrase start indices in token ids
    fail_cnt = 0
    phrase_indices: List[List[Tuple[int, int]]] = []
    for i, (toks, phrases) in enumerate(zip(char_indices, all_phrase_indices)):
        if len(toks) == 0:
            tmp_ranges = []
            fail_cnt += 1
        else:
            # try:
            tmp_ranges = get_range_of_phrases_in_token_level(
                toks,
                phrases,
                offset=offset,
                padding=padding,
                max_token_len=max_token_len,
                is_partial=all_noun_only
                or noun_only
                or prop_noun_only
                or named_entity_only,
            )
        # except:
        #     print(f"Passing {i}-th item. Range mismatch.")
        #     tmp_ranges = []
        #     fail_cnt += 1
        phrase_indices.append(tmp_ranges)
    if fail_cnt > 0:
        print(
            f"Failed to get phrase indices for {fail_cnt} items. out of {len(char_indices)} items."
        )
    return phrase_indices


def get_phrase_indices(
    tok_ids: torch.Tensor,
    tokenizer,
    input_texts: List[str],
    parsed_texts: List[Text],
    bsize: int,
    tok_masks: torch.Tensor = None,
    has_special_tokens: bool = True,
    max_token_len: int = 0,
    all_noun_only: bool = False,
    noun_only: bool = False,
    prop_noun_only: bool = False,
    named_entity_only: bool = False,
) -> Tuple[List, List]:
    """
    Q_ids Shape: (bsize, max_query_length)
    Q_mask Shape: (bsize, max_query_length)
    D_ids Shape: (bsize * nway, max_doc_length)
    D_mask Shape: (bsize * nway, max_doc_length)
    """

    if type(tok_ids) == torch.Tensor:
        if tok_masks is None:
            tok_ids_ = [tok_ids[b_size] for b_size in range(len(tok_ids))]
        else:
            tok_ids_ = [
                tok_ids[b_size][tok_masks[b_size] == 1]
                for b_size in range(len(tok_ids))
            ]
    else:
        tok_ids_ = tok_ids
    input_toks: List[List[str]] = list(map(tokenizer.convert_ids_to_tokens, tok_ids_))

    # Get phrase start indices in token ids
    phrase_indices = get_phrase_indices_by_toks(
        input_toks,
        input_texts,
        parsed_texts,
        has_special_tokens=has_special_tokens,
        max_token_len=max_token_len,
        all_noun_only=all_noun_only,
        noun_only=noun_only,
        prop_noun_only=prop_noun_only,
        named_entity_only=named_entity_only,
    )

    phrase_indices_batches = _split_into_batches2(phrase_indices, bsize)
    return phrase_indices_batches


def get_phrase_start_indices_batch(
    toks_batch: List[List[int]], noun_indices_batch: List[List[List[int]]]
) -> List[List[int]]:
    assert len(toks_batch) == len(
        noun_indices_batch
    ), f"len(toks_batch)={len(toks_batch)} != len(noun_indices_batch)={len(noun_indices_batch)}"
    # Sort the noun indices by their idx id
    noun_indices_batch = [
        sorted(noun_indices, key=lambda x: x[0]) for noun_indices in noun_indices_batch
    ]
    phrase_start_indices_batch: List[List[int]] = []
    for b_idx, (toks, noun_indices) in enumerate(zip(toks_batch, noun_indices_batch)):
        phrase_start_indices: List[int] = []
        i = 0
        noun_cnt = 0
        while i < len(toks):
            if noun_cnt < len(noun_indices) and i == noun_indices[noun_cnt][0]:
                # Append the start index of the phrase
                phrase_start_indices.append(i)
                # Increase the index by the length of the phrase
                i += len(noun_indices[noun_cnt])
                # Move to the next noun
                noun_cnt += 1
            else:
                # If not splitted token, append as the start index of the phrase (single word phrase)
                if not toks[i].startswith("##"):
                    phrase_start_indices.append(i)
                i += 1
        phrase_start_indices_batch.append(phrase_start_indices)
    return phrase_start_indices_batch


def validate_ranges(ranges: List[Tuple[int, int]], tok_ids_len: int) -> bool:
    # Check if the ranges are valid
    for start, end in ranges:
        if start >= end:
            return False
    # Check if the ranges cover all the tokens
    if end != tok_ids_len:
        return False
    return True


def fix_ranges(ranges: List[Tuple[int, int]], tok_ids_len: int) -> bool:
    # Check if the ranges cover all the tokens
    end = ranges[-1][1]
    if end != tok_ids_len:
        # Append tuples until the tok_ids_len
        for i in range(end, tok_ids_len):
            ranges.append((i, i + 1))
    return ranges
