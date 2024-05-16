from typing import *

import hydra

from colbert.modeling.tokenization.utils import get_phrase_indices
from eagle.tokenizer import NewTokenizer
from colbert.noun_extraction.identify_noun import SpacyModel, Text


class PhraseExtractor:
    def __init__(self, tokenizer: NewTokenizer) -> None:
        self.tokenizer = tokenizer
        self.spacy_model = SpacyModel()

    def __call__(
        self,
        texts: List[str],
        max_len: int,
        tokenized_result: Dict[str, List[int]] = None,
    ) -> List[List[Tuple[int]]]:
        # Parse the text with spacy
        parsed_texts: List[Text] = self.spacy_model(texts, max_token_num=max_len)
        # Tokenize
        if tokenized_result is None:
            tokenized_result = self.tokenizer(texts)
        input_ids = tokenized_result["input_ids"]
        attention_mask = tokenized_result["attention_mask"]
        # Extract phrase by token indices
        q_phrases = get_phrase_indices(
            input_ids,
            attention_mask,
            self.tokenizer.tokenizer,
            texts,
            parsed_texts,
            bsize=len(texts),
            all_noun_only=True,
        )[0]
        return q_phrases


@hydra.main(version_base=None, config_path="/root/ColBERT/config", config_name="config")
def main(cfg):
    tokenizer = NewTokenizer(cfg.q_tokenizer)
    extractor = PhraseExtractor(tokenizer=tokenizer)
    extractor(["Hello, my name is John Doe.", "I am a software engineer."], max_len=512)


if __name__ == "__main__":
    main()
