"""mBERT fine-tuning training and inference for arbitrary classification tasks."""

from logging import getLogger
from pathlib import Path
from typing import Dict, List

from datasets import Dataset
from transformers import (
    RemBertForSequenceClassification,
    RemBertTokenizer,
    Trainer,
    TrainingArguments,
)

from ticket_router_base.config import SEED
from ticket_router_base.data import BaseDataset
from ticket_router_base.predictor import (
    Trainer as TrainerProtocol,
    register_model,
)

from .hf import HFPredictor
from ticket_router_base.types import Record

from ticket_router_supervised.config import SAVE_DIR
from ticket_router_supervised.utils import create_datasets

logger = getLogger(__name__)

DEFAULT_TRAIN_ARGS = TrainingArguments(
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    max_grad_norm=1.0,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    seed=SEED,
    logging_steps=30,
    report_to=["none"],
)


def tokenize_func(examples, tokenizer, max_length: int = 256):
    return tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=max_length
    )


def train_mbert(
    train_ds: Dataset,
    val_ds: Dataset,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    target_col: str,
    save_path: Path,
    model_name: str = "google/rembert",
    epochs: int = 3,
) -> Trainer:
    tokenizer = RemBertTokenizer.from_pretrained(model_name)
    model = RemBertForSequenceClassification.from_pretrained(
        model_name, num_labels=len(label2id)
    )
    model.config.id2label = id2label
    model.config.label2id = label2id

    train_tok = train_ds.map(lambda x: tokenize_func(x, tokenizer), batched=True)
    val_tok = val_ds.map(lambda x: tokenize_func(x, tokenizer), batched=True)
    train_tok = train_tok.rename_column(target_col, "labels")
    val_tok = val_tok.rename_column(target_col, "labels")
    train_tok.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    val_tok.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    args = DEFAULT_TRAIN_ARGS
    args.output_dir = str(save_path.parent / target_col)
    args.num_train_epochs = epochs
    trainer = Trainer(
        model=model, args=args, train_dataset=train_tok, eval_dataset=val_tok
    )
    trainer.train()
    trainer.save_model(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    return trainer


@register_model
class MBERTPredictor(HFPredictor):
    name = "mbert"
    DEFAULT_SAVE_DIR = SAVE_DIR
    INFER_BATCH_SIZE = 64


class MBERTTrainer(TrainerProtocol):
    dataset: BaseDataset

    def __init__(self, dataset: BaseDataset):
        self.dataset = dataset

    def train(
        self,
        records: List[Record],
        val_records: List[Record] | None = None,
        epochs: int = 3,
    ) -> HFPredictor:
        if val_records is None:
            raise ValueError("MBERTTrainer requires val_records for early stopping")
        train_ds = create_datasets(records, self.dataset)
        val_ds = create_datasets(val_records, self.dataset)

        for task in self.dataset.all_tasks:
            logger.info(f"Starting remBERT training for {task.name}...")
            save_path = MBERTPredictor._get_model_path(task, self.dataset)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            train_mbert(
                train_ds,
                val_ds,
                id2label=self.dataset.get_id2label(task.name),
                label2id=self.dataset.get_label2id(task.name),
                target_col=task.name,
                save_path=save_path,
                epochs=epochs,
            )
            logger.info(f"{task.name} model training complete!")

        return MBERTPredictor.load_model(self.dataset)
