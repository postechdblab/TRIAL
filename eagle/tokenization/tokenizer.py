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
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

from eagle.utils import handle_old_ckpt

logger = logging.getLogger("NewTokenizesr")

DUMMY_TOK = "[dummy]"


def pickleable_func(tokenizer, *args, **kwargs):
    return tokenizer(*args, **kwargs)


class Tokenizer:
    def __init__(self, cfg: DictConfig, model_name: str) -> None:
        self.cfg = cfg
        self.name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
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

    @property
    def skip_tok_ids(self) -> List[int]:
        if self.cfg.skip_new_special_token:
            special_toks_ids = self.special_toks_ids[0:1] + self.special_toks_ids[2:]
        else:
            special_toks_ids = self.special_toks_ids
        return special_toks_ids

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
        return self.tokenize_batch(texts=[text], **kwargs)[0]

    def tokenize_batch(
        self,
        texts: List[str],
        padding=False,
        return_tensors: str = None,
        truncation=True,
    ) -> torch.Tensor:
        texts: List[str] = list(map(self._preprocess_text, texts))
        batch_size = 100000

        # TODO: Need this?
        if len(texts) > batch_size and False:
            chunks = list_utils.divide_into_chunks(texts, 32)
            multiprocessor = concurrent_utils.MultiProcessor(num_workers=32)
            logger.info("Tokenizing texts in parallel")
            for i in range(32):
                multiprocessor.run(
                    pickleable_func,
                    copy.deepcopy(self.tokenizer),
                    chunks[i],
                    padding=padding,
                    truncation=truncation,
                    max_length=self.cfg.max_len if truncation else None,
                    return_tensors=return_tensors,
                )
            multiprocessor.join()
            results = multiprocessor.results
            input_ids = []
            attention_mask = []
            for result in results:
                input_ids.extend(result["input_ids"])
                attention_mask.extend(result["attention_mask"])

            # Padd sequences
            if padding:
                input_ids = pad_sequence(input_ids, batch_first=True)
                attention_mask = pad_sequence(attention_mask, batch_first=True)

            tokenized_texts = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
        else:
            tokenized_texts = self.tokenizer(
                texts,
                padding=padding,
                truncation=truncation,
                max_length=self.cfg.max_len if truncation else None,
                return_tensors=return_tensors,
            )
        return tokenized_texts

    def cutoff_by_max_len(
        self, tok_ids: Union[List, torch.Tensor], maintain_special_tokens: bool = True
    ) -> Union[List, torch.Tensor]:
        # Check if the length is less than the max length
        if len(tok_ids) <= self.cfg.max_len:
            return tok_ids
        # Get the type of the input
        is_list = type(tok_ids) == list
        # Convert to tensor if it is a list
        if not is_list:
            device, dtype = tok_ids.device, tok_ids.dtype
            tok_ids = tok_ids.tolist()
        # Find the cutoff index
        cutoff_idx = self.cfg.max_len
        if maintain_special_tokens and tok_ids[-1] == self.tokenizer.sep_token_id:
            cutoff_idx -= 1
        # Cut off by the max length
        tok_ids = tok_ids[:cutoff_idx]
        if maintain_special_tokens:
            tok_ids = tok_ids + [self.tokenizer.sep_token_id]
        # Convert back to tensor if it was a tensor
        if not is_list:
            tok_ids = torch.tensor(tok_ids, dtype=dtype, device=device)
        return tok_ids

    def pad_1d_tensor_by_max_len(self, data: torch.Tensor) -> torch.Tensor:
        if len(data) >= self.cfg.max_len:
            return data
        return torch.tensor(
            data.tolist()
            + [self.tokenizer.pad_token_id] * (self.cfg.max_len - len(data)),
            dtype=data.dtype,
            device=data.device,
        )

    def pad_2d_tensor_by_max_len(self, data: torch.Tensor) -> torch.Tensor:
        assert len(data.shape) == 2, f"Unsupported shape: {data.shape}"
        if data.shape[1] >= self.cfg.max_len:
            return data
        # Create a tensor with the maximum length
        tmp = torch.zeros(
            (data.shape[0], self.cfg.max_len),
            dtype=data.dtype,
            device=data.device,
        )
        # Fill with pad token
        if self.tokenizer.pad_token_id != 0:
            tmp.fill_(self.tokenizer.pad_token_id)
        # Copy the original data
        tmp[:, : data.shape[1]] = data
        return tmp

    def pad_tensor_by_max_len(self, data: torch.Tensor) -> torch.Tensor:
        if len(data.shape) == 1:
            return self.pad_1d_tensor_by_max_len(data)
        elif len(data.shape) == 2:
            return self.pad_2d_tensor_by_max_len(data)
        raise ValueError(f"Unsupported shape: {data.shape}")

    def pad_sequence_by_max_len(
        self, data: Union[List, torch.Tensor]
    ) -> Union[List, torch.Tensor]:
        if isinstance(data, torch.Tensor):
            return self.pad_tensor_by_max_len(data)
        raise ValueError(f"Unsupported type: {type(data)}")

    def decode(self, *args, **kwargs) -> Any:
        return self.tokenizer.decode(*args, **kwargs)
