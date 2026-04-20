import pandas as pd

from pandera.typing import DataFrame

from ticket_router_base.types import ITCusomterSupportSchema, RecordSchema
from ticket_router_base.config import DATASET_4K_PATH, OUTPUT_DIR


def load_4k() -> DataFrame[ITCusomterSupportSchema]:
    df = pd.read_csv(DATASET_4K_PATH, encoding="utf-8")
    return ITCusomterSupportSchema.validate(df)


def load_test_set() -> DataFrame[RecordSchema]:
    test_path = OUTPUT_DIR / "test_set.jsonl"
    df = pd.read_json(test_path, orient="records", lines=True, encoding="utf-8")
    return RecordSchema.validate(df)


def load_train_set() -> DataFrame[RecordSchema]:
    train_path = OUTPUT_DIR / "train_set.jsonl"
    df = pd.read_json(train_path, orient="records", lines=True, encoding="utf-8")
    return RecordSchema.validate(df)
