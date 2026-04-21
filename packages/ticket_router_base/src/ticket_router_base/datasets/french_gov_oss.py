"""French government open-source software support tickets dataset."""

from ticket_router_base.config import PROJECT_ROOT
from ticket_router_base.datasets.base import (
    BaseDataset,
    ClassificationTask,
    GenerationTask,
)


class FrenchGovOSSDataset(BaseDataset):
    """French government inter-ministerial OSS support tickets (518 rows)."""

    name = "french-gov-oss"
    csv_path = (
        PROJECT_ROOT
        / "dataset"
        / "tickets-de-support-logiciels-libres-interministeriel.csv"
    )
    delimiter = ";"
    encoding = "utf-8-sig"
    title_column = None
    body_column = "SUJET"
    language_column = None
    id_column = "ID TICKET"

    classification_tasks = [
        ClassificationTask(
            "priorite",
            "PRIORITE",
            ["Bloquant", "Non bloquant"],
        ),
        ClassificationTask(
            "type_ticket",
            "TYPE TICKET",
            ["Anomalie", "Demande d'information"],
        ),
        ClassificationTask(
            "statut",
            "STATUT",
            [
                "1 - Rappel effectué",
                "2 - Suspendue/Complément demandé",
                "3 - Complément transmis",
                "4 - Contournement livré",
                "5 - Analysé",
                "7 - Information/solution fournie",
                "9 - Fermé",
                "10 - Abandonné",
                "11 - Rejeté",
            ],
        ),
        ClassificationTask(
            "criticite",
            "CRITICITE",
            ["Critique", "Non-critique"],
        ),
    ]
    generation_task = GenerationTask("etat_reversement", "ETAT REVERSEMENT")
    discrete_feature_columns = ["LOGICIEL"]
