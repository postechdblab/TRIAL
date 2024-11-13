import logging
from typing import *

import hkkang_utils.data as data_utils
import hkkang_utils.pattern as pattern_utils
import spacy
import tqdm
from transformers import T5TokenizerFast

from eagle.phrase.utils import (
    fix_ranges,
    get_range_of_phrases_in_token_level,
    get_range_of_tokens_in_char_level,
    validate_ranges,
)
from eagle.tokenization.tokenizer import Tokenizer

MAX_TOKEN_LENGTH = 512
MIN_SPACY_TOKEN_LENGTH_TO_CHECK = 100

logger = logging.getLogger("Constituency")

nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")

# Initialize the BERT tokenizer
t5_tokenizer = T5TokenizerFast.from_pretrained(
    "t5-base",
    model_max_length=MAX_TOKEN_LENGTH,
    legacy=False,
)


@spacy.language.Language.component("truncate_exceeding_tokens")
# Create a custom component to filter out long tokens
def truncate_exceeding_tokens(doc) -> None:
    # Check if the tokenized length exceeds the MAX_TOKEN_LENGTH
    if len(doc) > MIN_SPACY_TOKEN_LENGTH_TO_CHECK:

        # Tokenize the text using BERT tokenizer
        t5_tokens = t5_tokenizer.tokenize(doc.text)

        # Truncate the text using BERT token length
        truncated_text = t5_tokenizer.convert_tokens_to_string(
            t5_tokens[: MAX_TOKEN_LENGTH - 3]
        )

        # Create a new Doc object with the truncated text
        truncated_doc = nlp(truncated_text)
        return truncated_doc

    # Return the original doc if within limits
    return doc


@data_utils.dataclass
class POSToken:
    text: str
    pos: str
    start: int

    @property
    def range(self) -> Tuple[int, int]:
        return (self.start, self.start + len(self.text))


class POSParser(metaclass=pattern_utils.SingletonMetaWithArgs):
    def __init__(self, gpu_id: int = 0, tokenizer: Tokenizer = None) -> None:
        if gpu_id != -1:
            spacy.require_gpu(gpu_id=gpu_id)
        self.load_spacy_safely(model_name="en_core_web_lg")
        self.model.add_pipe("truncate_exceeding_tokens")
        self.consider_special_tokens = True
        self.tokenizer = tokenizer

    def __call__(
        self,
        text_or_texts: Union[List[str], str],
        tok_ids_or_tok_ids_list: Optional[Union[List[List[int]], List[int]]] = None,
        show_progress: bool = False,
        batch_size: int = 1000,
        max_tok_len: Optional[int] = None,
    ) -> List[Any]:
        if type(text_or_texts) == str:
            return self._extract(
                text=text_or_texts,
                tok_ids=tok_ids_or_tok_ids_list,
                show_progress=show_progress,
                max_tok_len=max_tok_len,
            )
        elif type(text_or_texts) == list:
            return self._extract_batch(
                texts=text_or_texts,
                tok_ids_list=tok_ids_or_tok_ids_list,
                show_progress=show_progress,
                batch_size=batch_size,
                max_tok_len=max_tok_len,
            )
        else:
            raise ValueError(f"Invalid type for text_or_texts: {type(text_or_texts)}")

    @property
    def offset(self) -> int:
        return 2 if self.consider_special_tokens else 0

    @property
    def padding(self) -> int:
        return 1 if self.consider_special_tokens else 0

    def _extract(
        self,
        text: str,
        tok_ids: Optional[List[int]] = None,
        show_progress: bool = False,
        max_tok_len: Optional[int] = None,
    ) -> List[Any]:
        # Parse the text
        return self._extract_batch(
            texts=[text],
            tok_ids_list=[tok_ids] if tok_ids is not None else None,
            show_progress=show_progress,
            batch_size=1,
            max_tok_len=max_tok_len,
        )

    def _extract_batch(
        self,
        texts: List[str],
        tok_ids_list: Optional[List[List[int]]] = None,
        show_progress: bool = False,
        batch_size: int = 10000,
        max_tok_len: Optional[int] = None,
    ) -> List[Tuple[str, str, Tuple[int, int]]]:
        # Parse the texts in batches
        parsed_results: List[List[POSToken]] = []
        for i, doc in enumerate(
            tqdm.tqdm(
                self.model.pipe(texts, batch_size=batch_size),
                disable=not show_progress,
                total=len(texts),
            )
        ):
            parsed_results.append(
                [POSToken(str(tok), tok.pos_, tok.idx) for tok in doc]
            )
            del doc
        # Get the token indices
        range_results = self.convert_word_to_token_indices(
            texts=texts,
            pos_tokens=parsed_results,
            tok_ids_list=tok_ids_list,
            max_tok_len=max_tok_len,
            show_progress=show_progress,
        )
        # Format the results
        final_results: List[Tuple[str, str, Tuple[int, int]]] = []
        for i in range(len(range_results)):
            ranges = range_results[i][2:-1]
            pos_tokens = parsed_results[i]
            pos_tokens = pos_tokens[: len(ranges)]
            pos_tags = [item.pos for item in pos_tokens]
            pos_token_strs = [item.text for item in pos_tokens]
            final_results.append((pos_token_strs, pos_tags, ranges))

        return final_results

    def convert_word_to_token_indices(
        self,
        texts: List[str],
        pos_tokens: List[List[POSToken]],
        tok_ids_list: List[List[int]],
        max_tok_len: Optional[int] = None,
        show_progress: bool = False,
    ) -> None:
        # Tokenize
        if tok_ids_list is None:
            tok_ids_list = self.tokenizer(texts)["input_ids"]
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
        all_word_indices_in_char: List[List[Tuple[int, int]]] = []
        for b_idx in range(len(texts)):
            # Get phrase indices
            all_word_indices_in_char.append([p.range for p in pos_tokens[b_idx]])

        # Convert the char-level indices into token-level indices
        all_word_indices_in_tok = []
        for b_idx in range(len(texts)):
            phrase_indices_in_tok = get_range_of_phrases_in_token_level(
                all_token_in_char_indices[b_idx],
                all_word_indices_in_char[b_idx],
                offset=self.offset,
                padding=self.padding,
                max_token_len=max_tok_len,
            )
            phrase_indices_in_tok = fix_ranges(
                phrase_indices_in_tok,
                len(tok_ids_list[b_idx]) + self.offset + self.padding,
            )
            assert validate_ranges(
                phrase_indices_in_tok,
                tok_ids_len=len(tok_ids_list[b_idx]) + self.offset + self.padding,
            ), f"Invalid ranges: {phrase_indices_in_tok}, {len(tok_ids_list[b_idx])+self.offset+self.padding}"
            all_word_indices_in_tok.append(phrase_indices_in_tok)

        # Handle phrases that exceed the max_len
        if max_tok_len is None:
            filtered_phrases_list = all_word_indices_in_tok
        else:
            filtered_phrases_list: List[List[Tuple[int, int]]] = []
            for phrases in all_word_indices_in_tok:
                filtered_phrases: List[Tuple[int, int]] = []
                for p_start, p_end in phrases:
                    if p_end <= max_tok_len:
                        filtered_phrases.append([p_start, p_end])
                filtered_phrases_list.append(filtered_phrases)
        return filtered_phrases_list

    def load_spacy_safely(self, model_name: str) -> spacy.language.Language:
        try:
            model = spacy.load(model_name)
        except:
            import subprocess

            subprocess.run(["python", "-m", "spacy", "download", model_name])
            model = spacy.load(model_name)
        self.model = model
