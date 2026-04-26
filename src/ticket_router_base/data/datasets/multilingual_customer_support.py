"""Multilingual customer support tickets dataset."""

from ticket_router_base.config import DATASET_DIR
from ticket_router_base.data import (
    DFDataset,
    ClassificationTask,
    GenerationTask,
    OrdinalTask,
)
from ticket_router_base.types import GroundRecord, Language

DEFAULT_DATASET_PATH = (
    DATASET_DIR
    / "multilingual-customer-support-tickets"
    / "dataset-tickets-multi-lang3-4k.csv"
)


class MultilingualCustomerSupportDataset(DFDataset):
    """Original multilingual customer support ticket dataset (4k/20k/28k)."""

    DEFAULT_DATASET_PATH = DEFAULT_DATASET_PATH

    TEST_RATIO = 0.25
    VALID_RATIO = 0.18

    name = "multilingual-customer-support"

    title_column = "subject"
    body_column = "body"
    language_column = "language"

    str2lang = {
        "en": Language.ENGLISH,
        "es": Language.SPANISH,
        "fr": Language.FRENCH,
        "de": Language.GERMAN,
        "pt": Language.PORTUGUESE,
    }

    classification_tasks = [
        ClassificationTask(
            name="queue",
            target_column="queue",
            labels=[
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
        OrdinalTask(
            name="priority", target_column="priority", labels=["low", "medium", "high"]
        ),
    ]
    generation_task = GenerationTask(name="preliminary_answer", target_column="answer")
    discrete_feature_columns = ["type", "business_type", "tag_1", "tag_2"]

    stratified_columns = ["language", "queue"]
    sensitive_columns = ["language"]

    def _demo_record(self) -> GroundRecord:
        """Return a minimal demo record for prompt examples."""
        labels = {task.name: task.labels[0] for task in self.classification_tasks}
        labels.update({task.name: task.labels[0] for task in self.ordinal_tasks})
        return GroundRecord(
            labels=labels,
            discrete_features={},
            generation_target="Thank you for your request. We will get back to you shortly.",
            sensitive_attributes={
                "language": "en",
            },
        )
