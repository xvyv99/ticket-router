"""Dataset registry and base classes."""

from typing import Type, Dict, List

from ticket_router_base.data.base import (
    BaseDataset,
    ClassificationTask,
    GenerationTask,
)
from .multilingual_customer_support import (
    MultilingualCustomerSupportDataset,
)
from .cfpb_complaints import CFPBComplaintsDataset
from .french_gov_oss import FrenchGovOSSDataset

DATASET_LST: List[Type[BaseDataset]] = [
    MultilingualCustomerSupportDataset,
    CFPBComplaintsDataset,
    FrenchGovOSSDataset,
]

DATASET_REGISTRY: Dict[str, Type[BaseDataset]] = {cls.name: cls for cls in DATASET_LST}


def get_dataset(name: str) -> BaseDataset:
    """Instantiate a dataset by name."""
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[name]()


__all__ = [
    "BaseDataset",
    "ClassificationTask",
    "GenerationTask",
    "MultilingualCustomerSupportDataset",
    "CFPBComplaintsDataset",
    "FrenchGovOSSDataset",
    "DATASET_REGISTRY",
    "get_dataset",
]
