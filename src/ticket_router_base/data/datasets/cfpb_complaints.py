"""CFPB consumer complaints dataset."""

from logging import getLogger

from ticket_router_base.config import DATASET_DIR
from ticket_router_base.data import (
    DFDataset,
    ClassificationTask,
    GenerationTask,
)

DEFAULT_DATASET_PATH = DATASET_DIR / "complaints.parquet"

logger = getLogger(__name__)


class CFPBComplaintsDataset(DFDataset):
    """Consumer Financial Protection Bureau complaints dataset (~25M rows)."""

    DEFAULT_DATASET_PATH = DEFAULT_DATASET_PATH

    name = "cfpb-complaints"

    title_column = None
    body_column = "Consumer complaint narrative"
    language_column = None
    id_column = "Complaint ID"

    # label lists are shortened for brevity; full lists should be inferred from data
    classification_tasks = [
        ClassificationTask("issue", "Issue", []),
        ClassificationTask(
            "sub_issue",
            "Sub-issue",
            [],  # populated dynamically in load() from data
        ),
    ]
    generation_task = GenerationTask("company_response", "Company response to consumer")
    discrete_feature_columns = [
        "State",
        "ZIP code",
        "Tags",
        "Submitted via",
        "Company",
    ]
