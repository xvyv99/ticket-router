from typing import Dict, List, Type

from ticket_router_base.predictor import Predictor

from .lr import LRPredictor, LRTrainer
from .xgb import XGBPredictor, XGBTrainer

MODEL_LST: List[Type[Predictor]] = [LRPredictor, XGBPredictor]

MODEL_REGISTRY: Dict[str, Type[Predictor]] = {
    model_cls.name: model_cls for model_cls in MODEL_LST
}


# TODO: add more models and update registry globally, e.g. in a separate file or via entry points
def get_model(name: str) -> Type[Predictor]:
    """Get a model by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name]


__all__ = [
    "LRPredictor",
    "LRTrainer",
    "XGBPredictor",
    "XGBTrainer",
    "MODEL_REGISTRY",
    "get_model",
]
