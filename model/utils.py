import functools
from typing import *

import hkkang_utils.data as data_utils


@data_utils.dataclass
class Document:
    id: str
    title: str
    text: str
    tokens: List[str] = data_utils.field(default_factory=list)
