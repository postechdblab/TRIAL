from typing import *

import torch
import tqdm

from eagle.phrase.constituency import ConstituencyParser, Phrase
from eagle.phrase.utils import (
    get_range_of_phrases_in_token_level,
    get_range_of_tokens_in_char_level,
)
from eagle.tokenizer import Tokenizer

HAS_SPECIAL_TOKENS = True


class PhraseExtractor2:
    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer
        self.constituency_parser = ConstituencyParser()

    def __call__(
        self,
        texts: List[str],
        max_tok_len: Optional[int] = None,
        tok_ids: Optional[torch.Tensor] = None,
        show_progress: bool = False,
    ) -> List[List[Tuple[int]]]:
        # Parse the text with spacy
        all_phrases: List[List[Phrase]] = self.constituency_parser(
            texts, show_progress=show_progress
        )

        # Tokenize
        if tok_ids is None:
            tokenized_result = self.tokenizer(texts, show_progress=show_progress)
            tok_ids = tokenized_result["input_ids"]
        if HAS_SPECIAL_TOKENS:
            tok_ids = [ids[2:-1] for ids in tok_ids]
        input_tokens = [
            self.tokenizer.tokenizer.convert_ids_to_tokens(ids) for ids in tok_ids
        ]

        # Find the character indices for the tokens
        all_char_indices: List[List[Tuple[int, int]]] = []
        for b_size in tqdm.tqdm(range(len(tok_ids)), disable=not show_progress):
            # try:
            tmp_indices = get_range_of_tokens_in_char_level(
                [tok.lower() for tok in input_tokens[b_size]],
                texts[b_size].lower(),
            )
            all_char_indices.append(tmp_indices)

        # Extract phrase indices
        all_phrase_indices_in_char: List[List[Tuple[int, int]]] = []
        for b_idx in range(len(texts)):
            # Get phrase indices
            all_phrase_indices_in_char.append([p.idx_range for p in all_phrases[b_idx]])

        # Convert the char-level indices into token-level indices
        all_phrase_indices_in_tok = []
        for b_idx in range(len(texts)):
            phrase_indices_in_tok = get_range_of_phrases_in_token_level(
                all_char_indices[b_idx],
                all_phrase_indices_in_char[b_idx],
                offset=2 if HAS_SPECIAL_TOKENS else 0,
                padding=1 if HAS_SPECIAL_TOKENS else 0,
                max_token_len=max_tok_len,
            )
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
