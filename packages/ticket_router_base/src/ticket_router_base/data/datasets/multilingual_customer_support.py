"""Multilingual customer support tickets dataset."""

from ticket_router_base.config import DATASET_DIR
from ticket_router_base.data import (
    CSVDataset,
    ClassificationTask,
    GenerationTask,
    OrdinalTask,
)

DEFAULT_DATASET_PATH = (
    DATASET_DIR
    / "multilingual-customer-support-tickets"
    / "dataset-tickets-multi-lang3-4k.csv"
)


class MultilingualCustomerSupportDataset(CSVDataset):
    """Original multilingual customer support ticket dataset (4k/20k/28k)."""

    DEFAULT_DATASET_PATH = DEFAULT_DATASET_PATH

    name = "multilingual-customer-support"

    title_column = "subject"
    body_column = "body"
    language_column = "language"

    classification_tasks = [
        ClassificationTask(
            "queue",
            "queue",
            [
                "Technical Support",
                "Product Support",
                "Customer Service",
                "IT Support",
                "Billing and Payments",
                "Returns and Exchanges",
                "Sales and Pre-Sales",
                "Service Outages and Maintenance",
                "General Inquiry",
                "Human Resources",
            ],
        ),
    ]
    ordinal_tasks = [
        OrdinalTask("priority", "priority", ["low", "medium", "high"]),
    ]
    generation_task = GenerationTask("preliminary_answer", "answer")
    discrete_feature_columns = ["type", "business_type", "tag_1", "tag_2"]
