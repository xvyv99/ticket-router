"""Multilingual customer support tickets dataset."""

from ticket_router_base.config import DATASET_DIR
from ticket_router_base.data import (
    DFDataset,
    ClassificationTask,
    GenerationTask,
    OrdinalTask,
)
from ticket_router_base.data.prompt_descriptor import PromptDescriptor
from ticket_router_base.types import GroundRecord, Language

DEFAULT_DATASET_PATH = (
    DATASET_DIR
    / "multilingual-customer-support-tickets"
    / "dataset-tickets-multi-lang3-4k.csv"
)


PROMPT_DESC = PromptDescriptor(
    system_role=(
        "You are a multilingual customer support routing assistant.\n"
        "Your job is to:\n"
        "1. Read the customer's request carefully\n"
        "2. Classify it into the most appropriate queue/category\n"
        "3. Assess urgency level (priority)\n"
        "4. Draft a brief, polite preliminary reply in the SAME language as the customer"
    ),
    label_descriptions={
        "queue": {
            "Technical Support": "Software bugs, feature malfunctions, system errors, crashes, technical glitches",
            "Product Support": "Usage questions, how-to guides, product feature inquiries, documentation help",
            "Customer Service": "General complaints, policy questions, account issues, feedback, satisfaction concerns",
            "IT Support": "Internal infrastructure, network problems, hardware failures, VPN, email outages",
            "Billing and Payments": "Invoices, refunds, payment failures, subscription issues, charge disputes",
            "Returns and Exchanges": "Refund requests, product returns, swap orders, defective or damaged items",
            "Sales and Pre-Sales": "Pricing inquiries, demos, purchase questions, upgrades, contract negotiations",
            "Service Outages and Maintenance": "Downtime alerts, scheduled maintenance, system-wide unavailability",
            "General Inquiry": "Miscellaneous requests that do not fit any specific category above",
            "Human Resources": "Employee-related issues, payroll, benefits, hiring, internal HR policies",
        },
        "priority": {
            "low": "Non-urgent, informational, no immediate business impact. Example: feature suggestions, general questions.",
            "medium": "Moderate impact, workaround exists, respond within standard SLA. Example: minor bugs, non-critical issues.",
            "high": "Critical business impact, no workaround, requires immediate attention. Example: system down, data loss, security breach.",
        },
    },
    fairness_notes=(
        "The customer may write in any supported language.\n"
        "The semantic content of the request -- not the language -- determines the queue and priority.\n"
        "Please ensure your classification is consistent across languages.\n"
        "Requests with similar meaning should receive the same queue and priority regardless of language.\n"
        "Pay special attention to small queues (e.g., Human Resources, General Inquiry) -- do not default to large queues."
    ),
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

    prompt_descriptor = PROMPT_DESC

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
