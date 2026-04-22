# ticket_router.data package
from .base import (
    BaseDataset,
    CSVDataset,
    ClassificationTask,
    GenerationTask,
    OrdinalTask,
)
from .datasets import (
    MultilingualCustomerSupportDataset,
    CFPBComplaintsDataset,
    FrenchGovOSSDataset,
    get_dataset,
    DATASET_REGISTRY,
)
from .loader import load_dataset, load_test_set, load_train_set

__all__ = [
    # base classes
    "BaseDataset",
    "CSVDataset",
    "ClassificationTask",
    "GenerationTask",
    "OrdinalTask",
    # datasets
    "MultilingualCustomerSupportDataset",
    "CFPBComplaintsDataset",
    "FrenchGovOSSDataset",
    "get_dataset",
    "DATASET_REGISTRY",
    # loader functions
    "load_dataset",
    "load_test_set",
    "load_train_set",
]
