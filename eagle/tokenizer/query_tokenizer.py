from typing import *

from eagle.tokenizer import BaseTokenizer


class QTokenizer(BaseTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    @property
    def skip_tok_ids(self) -> List[int]:
        return self.special_toks_ids