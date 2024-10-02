from typing import *

from eagle.tokenization import Tokenizer


def tokenize_and_cache_corpus(
    tokenizer: Tokenizer, corpus: Dict[str, Union[str, List[str]]]
) -> Dict[str, List[int]]:
    """Tokenize a list of texts using the provided tokenizer."""
    if type(list(corpus.values())[0]) == str:
        tokenized_items = tokenizer.tokenize_batch(list(corpus.values()))
        items = {}
        for idx, key in enumerate(corpus.keys()):
            items[key] = tokenized_items["input_ids"][idx]
    elif type(list(corpus.values())[0]) == list:
        # Flatten the list of lists
        tokenized_items = tokenizer.tokenize_batch(
            [f" {tokenizer.tokenizer.special_tokens_map["sep_token"]} ".join(sublist) for sublist in list(corpus.values())]
        )
        # Back to list of lists
        items = {}
        for idx, key in enumerate(corpus.keys()):
            items[key] = tokenized_items["input_ids"][idx]
    else:
        raise ValueError("Corpus values must be either a string or a list of strings.")

    return items
