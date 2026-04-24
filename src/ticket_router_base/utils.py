"""Shared utilities for record conversion, text combination, and prediction I/O."""

from io import TextIOWrapper
from pathlib import Path
from typing import Dict, Any, Generator, List, Literal
import json

from .types import PredSave, Prediction, Record


class JSONLLogger:
    path: Path
    file: TextIOWrapper

    mode: Literal["w", "r"]

    def __init__(self, path: Path, mode: Literal["w", "r"] = "w") -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.mode = mode
        self.file = self.path.open(mode, encoding="utf-8")

    def write(self, record: Dict[str, Any]) -> None:
        assert self.mode == "w", "File not opened in write mode"

        self.file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.file.flush()

    def read(self) -> Generator[str, None, None]:
        assert self.mode == "r", "File not opened in read mode"

        while True:
            line = self.file.readline()
            if not line:
                break
            yield line

    def close(self) -> None:
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def write_pred(preds: List[Prediction], records: List[Record], save_path: Path) -> None:
    assert len(preds) == len(records), (
        "Number of predictions must match number of records"
    )

    with JSONLLogger(save_path) as logger:
        for p, r in zip(preds, records):
            save_rec = PredSave(
                predicted=p,
                ground_truth=r,
            )
            logger.write(save_rec.model_dump())


def combine_text(title: str, body: str) -> str:
    return f"{title}\n{body}" if title else body


def combine_texts(records: List[Record]) -> List[str]:
    return [combine_text(r.title or "", r.body) for r in records]
