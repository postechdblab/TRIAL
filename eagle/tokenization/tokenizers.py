import abc
import logging
from typing import *

from omegaconf import DictConfig
from eagle.tokenization.tokenizer import Tokenizer

logger = logging.getLogger("Tokenizers")


class Tokenizers(abc.ABC):
    def __init__(self, q_cfg: DictConfig, d_cfg: DictConfig, model_name: str) -> None:
        self.q_tokenizer = Tokenizer(q_cfg, model_name)
        self.d_tokenizer = Tokenizer(d_cfg, model_name)
        # Check if initializers are valid
        assert len(self.q_tokenizer) != len(
            self.d_tokenizer
        ), f"Tokenizers have the same sizes: {len(self.q_tokenizer)} vs {len(self.d_tokenizer)}"

    @property
    def skip_tok_ids(self) -> List[int]:
        return list(set(self.q_tokenizer.skip_tok_ids + self.d_tokenizer.skip_tok_ids))

    @property
    def vocab_num(self) -> int:
        """Return the number of unique tokens in the entire tokenizers set."""
        return max(len(self.q_tokenizer), len(self.d_tokenizer))

    @property
    def model_name(self) -> str:
        assert (
            self.q_tokenizer.model_name == self.d_tokenizer.model_name
        ), f"Model names are different: {self.q_tokenizer.model_name} vs {self.d_tokenizer.model_name}"
        return self.q_tokenizer.model_name
