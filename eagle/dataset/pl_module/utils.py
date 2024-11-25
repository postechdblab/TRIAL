from typing import *

from eagle.tokenization.tokenizer import Tokenizer


def tokenize_batch_sentences(
    tokenizer: Tokenizer, corpus: Dict[int, Union[str, List[List[str]]]]
) -> Dict[str, List[List[int]]]:
    """Tokenize a list of texts using the provided tokenizer."""
    if type(list(corpus.values())[0]) == int:
        tokenized_items = tokenizer.tokenize_batch(list(corpus.values()))
        items = {}
        for idx, key in enumerate(corpus.keys()):
            items[key] = tokenized_items["input_ids"][idx]
    elif type(list(corpus.values())[0]) == list:
        nums: List[int] = 0
        all_sentences: List[str] = []
        for sublist in list(corpus.values()):
            nums += len(sublist)
            all_sentences.extend(sublist)
        # Tokenize all sentences
        tokenized_items = tokenizer.tokenize_batch(all_sentences)

        # Back to list of lists
        items = {}
        cnt = 0
        for idx, key in enumerate(corpus.keys()):
            num = nums[idx]
            items[key] = tokenized_items["input_ids"][cnt : cnt + num]
            cnt += num
    else:
        raise ValueError("Corpus values must be either a string or a list of strings.")

    return items
