from typing import *

import torch
import tqdm
import logging

from eagle.phrase.constituency import ConstituencyParser, Phrase
from eagle.phrase.utils import (
    get_range_of_phrases_in_token_level,
    get_range_of_tokens_in_char_level,
)
from eagle.tokenization import Tokenizer

logger = logging.getLogger("PhraseExtractor")


class PhraseExtractor:
    def __init__(
        self, tokenizer: Tokenizer, consider_special_tokens: Optional[bool] = True
    ) -> None:
        self.tokenizer = tokenizer
        self.constituency_parser = ConstituencyParser()
        self.consider_special_tokens = consider_special_tokens

    def __call__(
        self,
        texts: List[str],
        max_tok_len: Optional[int] = None,
        tok_ids_list: Optional[List[List[int]]] = None,
        to_char_indices: bool = False,
        to_token_indices: bool = False,
        show_progress: bool = False,
    ) -> List[List[Tuple[int]]]:
        # Parse the text with spacy
        phrases: List[List[Phrase]] = self.constituency_parser(
            texts, show_progress=show_progress
        )
        # Convert indices
        if to_char_indices:
            results = self.convert_phrase_to_char_indices(
                texts=texts,
                phrases=phrases,
                max_tok_len=max_tok_len,
                show_progress=show_progress,
            )
        elif to_token_indices:
            results = self.convert_phrase_to_token_indices(
                texts=texts,
                phrases=phrases,
                tok_ids_list=tok_ids_list,
                max_tok_len=max_tok_len,
                show_progress=show_progress,
            )
        else:
            results = phrases

        return results

    @property
    def offset(self) -> int:
        return 2 if self.consider_special_tokens else 0

    @property
    def padding(self) -> int:
        return 1 if self.consider_special_tokens else 0

    def convert_phrase_to_char_indices(
        self,
        texts: List[str],
        phrases: List[List[Phrase]],
        max_tok_len: Optional[int] = None,
        show_progress: bool = False,
    ) -> None:

        raise NotImplementedError("TODO: Implement this method")

    def convert_phrase_to_token_indices(
        self,
        texts: List[str],
        phrases: List[List[Phrase]],
        tok_ids_list: List[List[int]],
        max_tok_len: Optional[int] = None,
        show_progress: bool = False,
    ) -> None:
        # Tokenize
        if tok_ids_list is None:
            tokenized_result = self.tokenizer(texts, show_progress=show_progress)
            tok_ids_list = tokenized_result["input_ids"]
        if self.consider_special_tokens:
            tok_ids_list = [ids[2:-1] for ids in tok_ids_list]
        input_tokens = [
            self.tokenizer.tokenizer.convert_ids_to_tokens(ids) for ids in tok_ids_list
        ]

        # Find the character indices for the tokens
        all_token_in_char_indices: List[List[Tuple[int, int]]] = []
        for b_size in tqdm.tqdm(range(len(tok_ids_list)), disable=not show_progress):
            # try:
            tmp_indices = get_range_of_tokens_in_char_level(
                [tok.lower() for tok in input_tokens[b_size]],
                texts[b_size].lower(),
            )
            all_token_in_char_indices.append(tmp_indices)

        # Extract phrase indices
        all_phrase_indices_in_char: List[List[Tuple[int, int]]] = []
        for b_idx in range(len(texts)):
            # Get phrase indices
            all_phrase_indices_in_char.append([p.idx_range for p in phrases[b_idx]])

        # Convert the char-level indices into token-level indices
        all_phrase_indices_in_tok = []
        for b_idx in range(len(texts)):
            phrase_indices_in_tok = get_range_of_phrases_in_token_level(
                all_token_in_char_indices[b_idx],
                all_phrase_indices_in_char[b_idx],
                offset=self.offset,
                padding=self.padding,
                max_token_len=max_tok_len,
            )
            phrase_indices_in_tok = self.fix_ranges(
                phrase_indices_in_tok,
                len(tok_ids_list[b_idx]) + self.offset + self.padding,
            )
            assert self.validate_ranges(
                phrase_indices_in_tok,
                tok_ids_len=len(tok_ids_list[b_idx]) + self.offset + self.padding,
            ), f"Invalid ranges: {phrase_indices_in_tok}, {len(tok_ids_list[b_idx])+self.offset+self.padding}"
            all_phrase_indices_in_tok.append(phrase_indices_in_tok)

        # Handle phrases that exceed the max_len
        if max_tok_len is None:
            filtered_phrases_list = all_phrase_indices_in_tok
        else:
            filtered_phrases_list: List[List[Tuple[int, int]]] = []
            for phrases in all_phrase_indices_in_tok:
                filtered_phrases: List[Tuple[int, int]] = []
                for p_start, p_end in phrases:
                    if p_end <= max_tok_len:
                        filtered_phrases.append([p_start, p_end])
                filtered_phrases_list.append(filtered_phrases)
        return filtered_phrases_list

    def validate_ranges(self, ranges: List[Tuple[int, int]], tok_ids_len: int) -> bool:
        # Check if the ranges are valid
        for start, end in ranges:
            if start >= end:
                return False
        # Check if the ranges cover all the tokens
        if end != tok_ids_len:
            return False
        return True

    def fix_ranges(self, ranges: List[Tuple[int, int]], tok_ids_len: int) -> bool:
        # Check if the ranges cover all the tokens
        end = ranges[-1][1]
        if end != tok_ids_len:
            # Append tuples until the tok_ids_len
            for i in range(end, tok_ids_len - 1):
                ranges.append((i, i + 1))
        return ranges
