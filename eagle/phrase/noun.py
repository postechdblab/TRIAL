import functools
import logging
import string
from typing import *

import hkkang_utils.data as data_utils
import hkkang_utils.list as list_utils
import hkkang_utils.pattern as pattern_utils
import spacy
import tqdm

logger = logging.getLogger("IdentifyNoun")


@data_utils.dataclass
class Token:
    text: str
    pos: str  # Part of speech
    start_idx: int

    @property
    def end_idx(self) -> int:
        return self.start_idx + len(self.text)


@data_utils.dataclass
class Text:
    text: str
    tokens: List[Token]
    named_entities: List[Token]

    @functools.cached_property
    def named_entity_indices(self) -> List[Tuple[int, int]]:
        return self.get_named_entity_phrase_indices()

    @functools.cached_property
    def phrase_indices(self) -> List[Tuple[int, int]]:
        return self.get_noun_phrase_indices(return_all_phrases=True)

    @functools.cached_property
    def noun_phrase_indices(self) -> List[Tuple[int, int]]:
        return self.get_noun_phrase_indices(return_all_phrases=False, only_noun=True)

    @functools.cached_property
    def prop_noun_phrase_indices(self) -> List[Tuple[int, int]]:
        return self.get_noun_phrase_indices(return_all_phrases=False, only_prop=True)

    @functools.cached_property
    def all_noun_phrase_indices(self) -> List[Tuple[int, int]]:
        return self.get_noun_phrase_indices(return_all_phrases=False)

    @functools.cached_property
    def stop_words_indices(self) -> List[Tuple[int, int]]:
        return self.get_stop_word_indices()

    @functools.cached_property
    def informative_tokens(self) -> List[Tuple[Token]]:
        tmp = []
        for token in self.tokens:
            if token.pos in ["NOUN", "PROPN", "VERB", "ADJ"]:
                tmp.append(token)
        return tmp

    def get_noun_phrase_indices(
        self,
        return_all_phrases: bool = False,
        only_noun: bool = False,
        only_prop: bool = False,
    ) -> List[Tuple[int, int]]:
        """Using Tag to identify noun chunks."""
        if not self.text and not self.tokens:
            logger.warning("It seems that the text and tokens are not initialized!")
            return []

        char_used_bitmap = [0] * len(self.text)
        phrases: List[List[int]] = (
            []
        )  # List of List of start and end indices of noun chunks
        # First use NER to identify proper noun chunks
        for named_entity in self.named_entities:
            char_start_idx = named_entity.start_idx
            char_end_idx = named_entity.end_idx
            # Mark the characters as used
            for i in range(char_start_idx, char_end_idx):
                char_used_bitmap[i] = 1
            phrases.append([char_start_idx, char_end_idx])

        # For non overlapping words, identify noun chunks by POS tags
        if only_prop:
            target_pos = ["PROPN"]
        elif only_noun:
            target_pos = ["NOUN"]
        else:
            target_pos = ["PROPN", "NOUN"]
        tok_i = 0
        while tok_i < len(self.tokens):
            item = self.tokens[tok_i]
            # Skip space
            if item.pos == "SPACE":
                tok_i += 1
                continue
            char_start_idx = item.start_idx
            char_end_idx = item.end_idx
            if item.pos in target_pos or (tok_i != 0 and item.text[0].isupper()):
                next_tok_i = tok_i + 1
                # Add consecutive noun words
                while next_tok_i < len(self.tokens):
                    next_item = self.tokens[next_tok_i]
                    # If the next word is also a noun, or it is a proper noun, or it is a capitalized word
                    if next_item.pos == item.pos or next_item.text[0].isupper():
                        char_end_idx = next_item.end_idx
                        next_tok_i += 1
                    else:
                        break
                # Check if any of part the phrase is already used
                if all(
                    [
                        char_used_bitmap[i] == 0
                        for i in range(char_start_idx, char_end_idx)
                    ]
                ):
                    for i in range(char_start_idx, char_end_idx):
                        char_used_bitmap[i] = 1
                    phrases.append([char_start_idx, char_end_idx])
                    # Update the current token index
                    tok_i = next_tok_i
                # Check if only the current item can be used as a phrase
                elif all(
                    [
                        char_used_bitmap[i] == 0
                        for i in range(item.start_idx, item.end_idx)
                    ]
                ):
                    for i in range(item.start_idx, item.end_idx):
                        char_used_bitmap[i] = 1
                    phrases.append([item.start_idx, item.end_idx])
                    # Update the current token index
                    tok_i += 1
                else:
                    # Update the current token index
                    tok_i += 1
            else:
                if return_all_phrases:
                    if all(
                        [
                            char_used_bitmap[i] == 0
                            for i in range(char_start_idx, char_end_idx)
                        ]
                    ):
                        for i in range(char_start_idx, char_end_idx):
                            char_used_bitmap[i] = 1
                        phrases.append([item.start_idx, item.end_idx])
                # Update the current token index
                tok_i += 1
        # Sort by start index
        phrases = sorted(phrases, key=lambda x: x[0])

        # Combine the phrases if no white space in between (unless they are punctuations)
        if phrases:
            final_phrases = [phrases[0]]
        else:
            final_phrases = []
        for phrase in phrases[1:]:
            # Check: 1) There is no space in between 2) The phrase is not a punctuation
            if (
                final_phrases[-1][1] == phrase[0]
                and self.text[phrase[0] : phrase[1]] not in string.punctuation
                and self.text[final_phrases[-1][0] : final_phrases[-1][1]]
                not in string.punctuation
            ):
                final_phrases[-1][1] = phrase[1]
            else:
                final_phrases.append(phrase)
        return final_phrases

    def get_stop_word_indices(self) -> List[Tuple[int, int]]:
        stop_words = spacy.lang.en.stop_words.STOP_WORDS
        indices = []
        for i, tok in enumerate(self.tokens):
            if tok.text.lower() in stop_words:
                indices.append((tok.start_idx, tok.end_idx))
        return indices

    def get_named_entity_phrase_indices(self) -> List[Tuple[int, int]]:
        """Using NER to identify named entities."""
        if not self.text and not self.tokens:
            logger.warning("It seems that the text and tokens are not initialized!")
            return []
        phrases: List[Tuple[int]] = []
        for named_entity in self.named_entities:
            phrases.append([named_entity.start_idx, named_entity.end_idx])
        return phrases


class SpacyModel(metaclass=pattern_utils.SingletonMetaWithArgs):
    def __init__(self, gpu_id: int = 0) -> None:
        if gpu_id != -1:
            spacy.require_gpu(gpu_id=gpu_id)
        model_name = "en_core_web_trf"
        try:
            self.model = spacy.load(model_name)
        except:
            import subprocess

            subprocess.run(["python", "-m", "spacy", "download", model_name])
            self.model = spacy.load(model_name)
        self.ids = list(spacy.parts_of_speech.IDS.keys())

    def get_pos_tag(
        self, texts: List[str], batch_size: int = 300, show_progress: bool = False
    ) -> List[Text]:
        pos_tags = []
        for item in tqdm.tqdm(
            self.model.pipe(texts, batch_size=batch_size), disable=not show_progress
        ):
            pos_tags.append(
                Text(
                    text=item.text,
                    tokens=[
                        Token(text=token.text, pos=token.pos_, start_idx=token.idx)
                        for token in item
                    ],
                    named_entities=[],
                )
            )
        return pos_tags

    def simple_forward(
        self, texts: List[str], batch_size: int = 300, show_progress: bool = False
    ) -> List[Text]:
        all_texts: List[Text] = []
        for item in tqdm.tqdm(
            self.model.pipe(texts, batch_size=batch_size), disable=not show_progress
        ):
            e_text: List[Token] = []
            for entity in item.ents:
                e_text.append(
                    Token(
                        text=entity.text, pos=entity.label_, start_idx=entity.start_char
                    )
                )
            all_texts.append(Text(text=item.text, tokens=e_text, named_entities=e_text))
        return all_texts

    def __call__(
        self,
        texts: List[str],
        batch_size: int = 100,
        show_progress: bool = False,
    ) -> List[Text]:
        # Create pipeline
        pipeline = functools.partial(
            self.model.pipe,
            disable=["parser", "lemmatizer", "textcat"],
            batch_size=batch_size,
        )
        # Forward
        results: List[List[List[int]]] = []
        for item in tqdm.tqdm(
            pipeline(texts),
            desc="Extracting noun indices",
            total=len(texts),
            disable=not show_progress,
        ):
            ## Convert to my data structure

            # Identify named entities
            indices_for_named_entity = list_utils.do_flatten_list(
                [list(range(i.start, i.end)) for i in item.ents]
            )
            named_entities: List[Token] = []

            tokens = []
            do_skip = False
            # Fix the tokenization issue with my own heuristic
            for tok_idx in range(len(item)):
                # Skip this token if it is already included in the previous token
                if do_skip:
                    do_skip = False
                    continue

                # Heuristic
                # 1. Handle "id"
                # 2. Handle "dont"
                token = item[tok_idx]
                if (
                    (
                        token.text == "i"
                        and len(item) > tok_idx + 1
                        and item[tok_idx + 1].text == "d"
                    )
                    or (
                        token.text == "do"
                        and len(item) > tok_idx + 1
                        and item[tok_idx + 1].text == "nt"
                    )
                    or (
                        len(item) > tok_idx + 1
                        and item[tok_idx + 1].text == "d"
                        and item[tok_idx].pos + 1 == item[tok_idx + 1].pos
                    )
                ):
                    next_token = item[tok_idx + 1].text
                    # Combine the two tokens and skip the next token
                    do_skip = True
                    my_token = Token(
                        text=token.text + next_token,
                        pos=token.pos_,
                        start_idx=token.idx,
                    )
                    tokens.append(my_token)
                    # Append to named entity list
                    if (
                        tok_idx in indices_for_named_entity
                        or tok_idx + 1 in indices_for_named_entity
                    ):
                        named_entities.append(my_token)
                else:
                    my_token = Token(
                        text=token.text, pos=token.pos_, start_idx=token.idx
                    )
                    tokens.append(my_token)
                    # Append to named entity list
                    if tok_idx in indices_for_named_entity:
                        named_entities.append(my_token)

            result = Text(text=item.text, tokens=tokens, named_entities=named_entities)

            # Aggregate
            results.append(result)
        return results


def extract_nouns_indices_batch(
    texts: List[str], batch_size: int = 100, show_progress: bool = False
) -> List[List[List[int]]]:
    # Initialize model
    model = SpacyModel()
    # Create pipeline
    results = model(texts, batch_size=batch_size, show_progress=show_progress)
    final_results = [item.phrase_indices for item in results]
    return final_results


def extract_noun_indices(text: str, show_progress: bool = False) -> List[List[int]]:
    return extract_nouns_indices_batch([text], show_progress=show_progress)[0]


def extract_nouns(text: str, show_progress: bool = False) -> List[str]:
    return extract_nouns_batch([text])[0]


def extract_nouns_batch(
    texts: List[str], show_progress: bool = False
) -> List[List[str]]:
    # Extract noun indices
    noun_indices_batch = extract_nouns_indices_batch(texts, show_progress=show_progress)
    assert len(noun_indices_batch) == len(
        texts
    ), f"({len(noun_indices_batch)} != {len(texts)})"
    # Extract noun words
    noun_words_batch = []
    for noun_indices, sentence in zip(noun_indices_batch, texts):
        noun_words = [sentence[i:j] for i, j in noun_indices]
        noun_words_batch.append(noun_words)
    return noun_words_batch


def get_noun_phrase_indices(
    text: str, tokens: List[Token], named_entities: List[Token]
) -> List[Tuple[int, int]]:
    """Using Tag to identify noun chunks."""
    char_used_bitmap = [0] * len(text)
    nouns: List[List[int]] = []  # List of List of start and end indices of noun chunks
    # First use NER to identify proper noun chunks
    for named_entity in named_entities:
        char_start_idx = named_entity.start_idx
        char_end_idx = named_entity.end_idx
        # Mark the characters as used
        for i in range(char_start_idx, char_end_idx):
            char_used_bitmap[i] = 1
        nouns.append([char_start_idx, char_end_idx])

    # For non overlapping words, identify noun chunks by POS tags
    target_pos = ["PROPN", "NOUN"]
    tok_i = 0
    while tok_i < len(tokens):
        item = tokens[tok_i]
        if item.pos in target_pos:
            tok_i += 1
            # Add consecutive noun words
            char_start_idx = item.start_idx
            char_end_idx = item.end_idx
            while tok_i < len(tokens):
                next_item = tokens[tok_i]
                # If the next word is also a noun, or it is a proper noun, or it is a capitalized word
                if next_item.pos in target_pos or next_item.text[0].isupper():
                    char_end_idx = next_item.end_idx
                    tok_i += 1
                else:
                    break
            # Check if any of part the phrase is already used
            if all(
                [char_used_bitmap[i] == 0 for i in range(char_start_idx, char_end_idx)]
            ):
                for i in range(char_start_idx, char_end_idx):
                    char_used_bitmap[i] = 1
                nouns.append([char_start_idx, char_end_idx])
        else:
            tok_i += 1
    # Sort by start index
    nouns = sorted(nouns, key=lambda x: x[0])
    return nouns


def main():
    while True:
        sentence = input("Enter sentence: ").strip()
        if sentence == "q":
            print("Bye!")
            break
        noun_words = extract_nouns(text=sentence)
        print(noun_words)


if __name__ == "__main__":
    # main()
    # Read in queries
    text = """Are Local H and For Against both from the United States?"""
    # text = """what is pcnt"""
    noun_words = extract_nouns(text=text)
    print(noun_words)
