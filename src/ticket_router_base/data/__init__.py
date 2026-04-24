# ticket_router.data package
from .base import (
    BaseDataset,
    DFDataset,
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
from .utils import load_dataset

__all__ = [
    # base classes
    "BaseDataset",
    "DFDataset",
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
    # data utils
]
