"""Dataset registry and base classes."""

from ticket_router_base.datasets.base import (
    BaseDataset,
    ClassificationTask,
    GenerationTask,
)
from ticket_router_base.datasets.multilingual_customer_support import (
    MultilingualCustomerSupportDataset,
)
from ticket_router_base.datasets.cfpb_complaints import CFPBComplaintsDataset
from ticket_router_base.datasets.french_gov_oss import FrenchGovOSSDataset

DATASET_REGISTRY: dict[str, type[BaseDataset]] = {
    MultilingualCustomerSupportDataset.name: MultilingualCustomerSupportDataset,
    CFPBComplaintsDataset.name: CFPBComplaintsDataset,
    FrenchGovOSSDataset.name: FrenchGovOSSDataset,
}


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
