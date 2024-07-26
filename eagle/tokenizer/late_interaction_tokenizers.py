import logging
from typing import *

from eagle.tokenizer import Tokenizers

logger = logging.getLogger("Tokenizers")


class LateInteractionTokenizer(Tokenizers):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
