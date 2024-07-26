from typing import *

import hydra

from eagle.phrase.noun import SpacyModel, Text
from eagle.phrase.utils import get_phrase_indices
from eagle.tokenizer import Tokenizer, QTokenizer


class PhraseExtractor:
    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer
        self.spacy_model = SpacyModel()

    def __call__(
        self,
        texts: List[str],
        max_tok_len: Optional[int] = None,
        tokenized_result: Dict[str, List[int]] = None,
    ) -> List[List[Tuple[int]]]:
        # Parse the text with spacy
        parsed_texts: List[Text] = self.spacy_model.simple_forward(texts)

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
            named_entity_only=True,
        )[0]

        # Handle phrases that exceed the max_len
        if max_tok_len is None:
            filtered_phrases_list = q_phrases
        else:
            filtered_phrases_list: List[List[Tuple[int, int]]] = []
            for phrases in q_phrases:
                filtered_phrases: List[Tuple[int, int]] = []
                for p_start, p_end in phrases:
                    if p_end <= max_tok_len:
                        filtered_phrases.append([p_start, p_end])
                filtered_phrases_list.append(filtered_phrases)
        return filtered_phrases_list


@hydra.main(version_base=None, config_path="/root/EAGLE/config", config_name="config")
def main(cfg):
    tokenizer = QTokenizer(cfg.q_tokenizer)
    extractor = PhraseExtractor(tokenizer=tokenizer)
    extractor(["Hello, my name is John Doe.", "I am a software engineer."], max_len=512)


if __name__ == "__main__":
    main()
