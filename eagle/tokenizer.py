import functools
import logging
import string
from typing import *

import torch
from omegaconf import DictConfig
from transformers import AutoTokenizer

logger = logging.getLogger("NewTokenizesr")


class NewTokenizer:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.name)
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [self.cfg.special_tok]}
        )
        self.new_special_tok_id = self.tokenizer.convert_tokens_to_ids(
            self.cfg.special_tok
        )

    def __call__(self, text_or_texts: Union[str, List[str]], **kwargs) -> torch.Tensor:
        if type(text_or_texts) == str:
            return self.tokenize(text_or_texts, **kwargs)
        elif type(text_or_texts) == list:
            return self.tokenize_batch(text_or_texts, **kwargs)
        raise ValueError(f"Unsupported type: {type(text_or_texts)}")

    def __len__(self) -> int:
        return len(self.tokenizer)

    @functools.cached_property
    def special_toks_ids(self) -> List[int]:
        tokens = [
            self.tokenizer.pad_token_id,
            self.tokenizer.mask_token_id,
            self.new_special_tok_id,
        ]
        tokens = [token for token in tokens if token is not None]
        return list(set(tokens))

    @functools.cached_property
    def punctuations(self) -> List[int]:
        tokens = [
            self.tokenizer.encode(symbol, add_special_tokens=False)[0]
            for symbol in string.punctuation
        ]
        tokens = [token for token in tokens if token is not None]
        return list(set(tokens))

    def _preprocess_text(self, text: str) -> str:
        return self.cfg.special_tok + " " + text

    def tokenize(self, text: str, **kwargs) -> torch.Tensor:
        return self.tokenize_batch(texts=[text], **kwargs)

    def tokenize_batch(
        self, texts: List[str], padding=False, return_tensors: str = None
    ) -> torch.Tensor:
        texts: List[str] = list(map(self._preprocess_text, texts))
        return self.tokenizer(
            texts,
            padding=padding,
            truncation=True,
            max_length=self.cfg.max_len,
            return_tensors=return_tensors,
        )


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
