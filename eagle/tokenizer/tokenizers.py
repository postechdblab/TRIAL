import logging
from typing import *

from omegaconf import DictConfig
from eagle.tokenizer import QTokenizer, DTokenizer

logger = logging.getLogger("Tokenizers")


class Tokenizers:
    def __init__(self, q_cfg: DictConfig, d_cfg: DictConfig, model_name: str) -> None:
        self.q_tokenizer = QTokenizer(q_cfg, model_name)
        self.d_tokenizer = DTokenizer(d_cfg, model_name)
        # Check if initializers are valid
        assert len(self.q_tokenizer) != len(
            self.d_tokenizer
        ), f"Tokenizers have the same sizes: {len(self.q_tokenizer)} vs {len(self.d_tokenizer)}"

    @property
    def vocab_num(self) -> int:
        """Return the number of unique tokens in the entire tokenizers set."""
        return max(len(self.q_tokenizer), len(self.d_tokenizer))
