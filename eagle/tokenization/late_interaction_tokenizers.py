import logging
from typing import *

from eagle.tokenization.tokenizers import Tokenizers

logger = logging.getLogger("Tokenizers")


class LateInteractionTokenizer(Tokenizers):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
