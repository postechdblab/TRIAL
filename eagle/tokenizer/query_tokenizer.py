from typing import *

from eagle.tokenizer import BaseTokenizer


class QTokenizer(BaseTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def skip_tok_ids(self) -> List[int]:
        if self.cfg.skip_new_special_token:
            special_toks_ids = self.special_toks_ids[0:1] + self.special_toks_ids[2:]
        else:
            special_toks_ids = self.special_toks_ids
        return special_toks_ids
