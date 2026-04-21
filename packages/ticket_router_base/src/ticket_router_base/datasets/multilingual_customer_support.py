"""Multilingual customer support tickets dataset."""

from ticket_router_base.config import PROJECT_ROOT
from ticket_router_base.datasets.base import (
    BaseDataset,
    ClassificationTask,
    GenerationTask,
)


class MultilingualCustomerSupportDataset(BaseDataset):
    """Original multilingual customer support ticket dataset (4k/20k/28k)."""

    name = "multilingual-customer-support"
    csv_path = (
        PROJECT_ROOT
        / "dataset"
        / "multilingual-customer-support-tickets"
        / "dataset-tickets-multi-lang3-4k.csv"
    )
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
        ClassificationTask("priority", "priority", ["high", "medium", "low"]),
    ]
    generation_task = GenerationTask("preliminary_answer", "answer")
    discrete_feature_columns = ["type", "business_type", "tag_1", "tag_2"]
