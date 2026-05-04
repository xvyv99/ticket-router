"""Dataset registry and base classes."""

from typing import Type, Dict, List

from ticket_router.base.data.base import (
    BaseDataset,
)
from .multilingual_customer_support import (
    MultilingualCustomerSupportDataset,
)
from .cfpb_complaints import CFPBComplaintsDataset
from .french_gov_oss import FrenchGovOSSDataset

DATASET_LST_: List[Type[BaseDataset]] = [
    MultilingualCustomerSupportDataset,
    CFPBComplaintsDataset,
    FrenchGovOSSDataset,
]  # User should not use this directly; use DATASET_REGISTRY or get_dataset() instead

DATASET_REGISTRY: Dict[str, Type[BaseDataset]] = {cls.name: cls for cls in DATASET_LST_}


def get_dataset(name: str) -> Type[BaseDataset]:
    """Get a dataset class by name."""
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[name]


__all__ = [
    "MultilingualCustomerSupportDataset",
    "CFPBComplaintsDataset",
    "FrenchGovOSSDataset",
    "DATASET_REGISTRY",
    "get_dataset",
]
