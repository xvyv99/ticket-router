from io import TextIOWrapper
from pathlib import Path
from typing import Dict, Any, List
import json
from dataclasses import asdict

import pandas as pd
from datasets import Dataset

from .types import PredSave, Prediction, Language, Record, RecordDF, Task, df_to_records
from .types import QUEUE2ID, PRIORITY2ID, ID2QUEUE, ID2PRIORITY


class JSONLLogger:
    path: Path
    file: TextIOWrapper

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", encoding="utf-8")

    def write(self, record: Dict[str, Any]) -> None:
        self.file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.file.flush()

    def close(self) -> None:
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def write_pred(
    preds: List[Prediction], records: List[Record] | RecordDF, save_path: Path
) -> None:
    assert len(preds) == len(records), (
        "Number of predictions must match number of records"
    )

    if isinstance(records, pd.DataFrame):
        records = df_to_records(records)

    with JSONLLogger(save_path) as logger:
        for p, r in zip(preds, records):
            save_rec = PredSave(
                request_id=p.request_id,
                language=Language(r.language),
                predicted=p,
                ground_truth=r,
            )
            logger.write(asdict(save_rec))


def task2labels(task: Task) -> Dict[int, str]:
    match task:
        case Task.QUEUE:
            return ID2QUEUE
        case Task.PRIORITY:
            return ID2PRIORITY
        case _:
            raise ValueError(f"Unsupported task: {task}")
