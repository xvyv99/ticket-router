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


def create_datasets(records: RecordDF | List[Record]) -> Dataset:
    records_lst = []

    if isinstance(records, pd.DataFrame):
        records = df_to_records(records)

    for r in records:
        text = f"{r.subject}\n{r.body}"
        tags = [r.tag_1, r.tag_2] if r.tag_1 and r.tag_2 else []
        # TODO: handle tags better, currently just taking the first 2 tags, but some records have more than 2 tags
        records_lst.append(
            {
                "text": text,
                "queue": QUEUE2ID.get(str(r.queue)),
                "priority": PRIORITY2ID.get(str(r.priority)),
                "tags": tags,
            }
        )
    return Dataset.from_list(records_lst)


def task2labels(task: Task) -> Dict[int, str]:
    match task:
        case Task.QUEUE:
            return ID2QUEUE
        case Task.PRIORITY:
            return ID2PRIORITY
        case _:
            raise ValueError(f"Unsupported task: {task}")
