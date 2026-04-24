"""French government open-source software support tickets dataset."""

from ticket_router_base.config import DATASET_DIR
from ticket_router_base.data import (
    DFDataset,
    ClassificationTask,
    GenerationTask,
    OrdinalTask,
)

DEFAULT_DATASET_PATH = (
    DATASET_DIR / "tickets-de-support-logiciels-libres-interministeriel.csv"
)


class FrenchGovOSSDataset(DFDataset):
    """French government inter-ministerial OSS support tickets (518 rows)."""

    ENCODING = "utf-8-sig"  # dataset contains a UTF-8 BOM, so we use utf-8-sig to handle it correctly

    DEFAULT_DATASET_PATH = DEFAULT_DATASET_PATH

    name = "french-gov-oss"

    title_column = None
    body_column = "SUJET"
    language_column = None
    id_column = "ID TICKET"

    classification_tasks = [
        ClassificationTask(
            "type_ticket",
            "TYPE TICKET",
            ["Anomalie", "Demande d'information"],
        ),
    ]
    ordinal_tasks = [
        OrdinalTask(
            "priorite",
            "PRIORITE",
            ["Non bloquant", "Bloquant"],
        ),
        OrdinalTask(
            "criticite",
            "CRITICITE",
            ["Non-critique", "Critique"],
        ),
    ]
    generation_task = GenerationTask("etat_reversement", "ETAT REVERSEMENT")
    discrete_feature_columns = ["LOGICIEL"]
