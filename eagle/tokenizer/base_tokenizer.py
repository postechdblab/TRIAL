import copy
import functools
import logging
import string
from typing import *

import hkkang_utils.concurrent as concurrent_utils
import hkkang_utils.list as list_utils
import torch
import tqdm
from omegaconf import DictConfig
from transformers import AutoTokenizer

from eagle.utils import handle_old_ckpt

logger = logging.getLogger("NewTokenizesr")

DUMMY_TOK = "[dummy]"


def pickleable_func(tokenizer, *args, **kwargs):
    return tokenizer(*args, **kwargs)


class BaseTokenizer:
    def __init__(self, cfg: DictConfig, model_name: str) -> None:
        self.cfg = cfg
        self.name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        new_tok_offset = handle_old_ckpt(self.cfg, "new_tok_offset")
        self.tokenizer.add_special_tokens(
            {
                "additional_special_tokens": self._get_dummy_tokens(
                    0 if new_tok_offset is None else new_tok_offset
                )
                + [self.cfg.special_tok]
            }
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

    def _get_dummy_tokens(self, n: int) -> List[str]:
        toks = []
        for i in range(n):
            toks.append(DUMMY_TOK.replace("]", f"{i}]"))
        return toks

    @property
    def model_name(self) -> str:
        name = self.name.split("/")[-1]
        sub_names = name.split("-")
        if len(sub_names) > 2:
            final_name = "-".join(sub_names[:2])
        else:
            final_name = sub_names[0]
        return final_name

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
        batch_size = 10000

        if len(texts) > batch_size:
            if True:
                chunks = list_utils.divide_into_chunks(texts, 32)
                multiprocessor = concurrent_utils.MultiProcessor(num_workers=32)
                logger.info("Tokenizing texts in parallel")
                for i in range(32):
                    multiprocessor.run(
                        pickleable_func,
                        copy.deepcopy(self.tokenizer),
                        chunks[i],
                        padding=padding,
                        truncation=True,
                        max_length=self.cfg.max_len,
                        return_tensors=return_tensors,
                    )
                multiprocessor.join()
                results = multiprocessor.results
                input_ids = []
                attention_mask = []
                for result in results:
                    input_ids.extend(result["input_ids"])
                    attention_mask.extend(result["attention_mask"])
                tokenized_texts = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }
                logger.info("Done!")
            elif False:
                logger.info(f"Tokenizing {len(texts)} texts in batches of {batch_size}")
                input_ids = []
                attention_mask = []
                for i in tqdm.tqdm(range(0, len(texts), batch_size)):
                    tmp = self.tokenizer(
                        texts[i : i + batch_size],
                        padding=padding,
                        truncation=True,
                        max_length=self.cfg.max_len,
                        return_tensors=return_tensors,
                    )
                    input_ids.extend(tmp["input_ids"])
                    attention_mask.extend(tmp["attention_mask"])
                tokenized_texts = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }
        else:
            tokenized_texts = self.tokenizer(
                texts,
                padding=padding,
                truncation=True,
                max_length=self.cfg.max_len,
                return_tensors=return_tensors,
            )
        return tokenized_texts
