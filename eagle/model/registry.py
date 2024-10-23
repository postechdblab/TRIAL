from typing import Dict, Type

from eagle.model.base_model import BaseModel
from eagle.model.colbert import ColBERT
from eagle.model.dpr import DPR
from eagle.model.eagle import EAGLE

MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
    "eagle": EAGLE,
    "colbert": ColBERT,
    "cross_encoder": DPR,
}
