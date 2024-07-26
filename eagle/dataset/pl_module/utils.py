from typing import *

from eagle.tokenizer import Tokenizer


def tokenize_and_cache_corpus(
    tokenizer: Tokenizer, corpus: Dict[str, str]
) -> Dict[str, List[int]]:
    """Tokenize a list of texts using the provided tokenizer."""
    tokenized_items = tokenizer.tokenize_batch(corpus.values())
    items = {}
    for idx, key in enumerate(corpus.keys()):
        items[key] = tokenized_items["input_ids"][idx]
    return items
