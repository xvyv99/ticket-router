"""Abstract base for HuggingFace transformer-based predictors and trainers."""

from __future__ import annotations

from abc import ABC
from logging import getLogger
from pathlib import Path
from typing import ClassVar, Dict, List, Tuple, Type

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer as _HFTrainer,
    TrainingArguments,
)

from ticket_router_base.config import MODEL_DIR as BASE_MODEL_DIR
from ticket_router_base.data import BaseDataset
from ticket_router_base.predictor import Predictor, Trainer as TrainerProtocol
from ticket_router_base.types import (
    ErrorFlag,
    Prediction,
    Record,
)
from ticket_router_base.utils import combine_texts

from ticket_router.supervised.config import TORCH_DEVICE
from ticket_router.supervised.utils import create_datasets

logger = getLogger(__name__)


def tokenize_func(examples, tokenizer, max_length: int = 256):
    return tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=max_length
    )


class HFPredictor(Predictor, ABC, skip_check=True):
    """Base predictor for HuggingFace transformer sequence classification models.

    `_get_model_path` and `_predict_task` both have sensible defaults using
    AutoTokenizer / AutoModelForSequenceClassification. Subclasses only need
    to override for non-standard layouts or custom inference logic.
    """

    INFER_BATCH_SIZE: ClassVar[int]
    MAX_LENGTH: ClassVar[int] = 256

    _model_paths: Dict[str, Path]
    dataset: BaseDataset

    def __init__(self, model_paths: Dict[str, Path], dataset: BaseDataset):
        self._model_paths = model_paths
        self.dataset = dataset

    @classmethod
    def load_model(cls, dataset: BaseDataset) -> "HFPredictor":
        """Load fine-tuned task models from disk and return a predictor instance."""
        model_paths: Dict[str, Path] = {}
        for task in dataset.all_tasks:
            path = cls._get_model_path(task, dataset)
            if not path.exists():
                raise FileNotFoundError(f"Model for {task.name} not found at {path}")
            model_paths[task.name] = path
        return cls(model_paths=model_paths, dataset=dataset)

    @classmethod
    def _get_model_path(cls, task, dataset: BaseDataset) -> Path:
        """Default save path: ``MODEL_DIR/<model_name>/<dataset_name>/<task>_best``."""
        return BASE_MODEL_DIR / cls.name / dataset.name / f"{task.name}_best"

    def _predict_task(
        self,
        model_path: Path,
        records: List[Record],
        id2label: Dict[int, str],
    ) -> List[Tuple[str, float]]:
        """Default batch inference using Auto classes.

        Subclasses may override for custom logic (e.g. different batch sizes
        or quantization-aware loading).
        """
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        model.to(TORCH_DEVICE)

        texts = combine_texts(records)
        results: List[Tuple[str, float]] = []

        with torch.inference_mode():
            for i in tqdm(
                range(0, len(texts), self.INFER_BATCH_SIZE),
                desc="Batched inference",
            ):
                batch_texts = texts[i : i + self.INFER_BATCH_SIZE]
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=self.MAX_LENGTH,
                )
                inputs = {k: v.to(TORCH_DEVICE) for k, v in inputs.items()}
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)

                for j in range(len(batch_texts)):
                    pred_id = int(torch.argmax(probs[j]))
                    confidence = float(probs[j][pred_id])
                    pred_label = id2label.get(pred_id, str(pred_id))
                    results.append((pred_label, confidence))

        return results

    def predict(self, records: List[Record], run_id: int = 0) -> List[Prediction]:
        """Run inference for all tasks and assemble predictions."""
        task_results: Dict[str, List[Tuple[str, float]]] = {}
        for task_name, model_path in self._model_paths.items():
            id2label = self.dataset.get_id2label(task_name)
            task_results[task_name] = self._predict_task(model_path, records, id2label)

        predictions = []
        for i, rec in enumerate(records):
            labels: Dict[str, str] = {}
            confidences: Dict[str, float] = {}
            for task_name in self.dataset.task_names:
                result = task_results[task_name]
                labels[task_name] = result[i][0]
                confidences[task_name] = result[i][1]

            pred = Prediction(
                request_id=rec.request_id,
                sensitive_attributes=rec.sensitive_attributes,
                labels=labels,
                discrete_features=rec.discrete_features,
                generation_target=None,
                confidences=confidences,
                raw_output=None,
                error=ErrorFlag.SUCCESS,
            )
            predictions.append(pred)

        return predictions


class HFTrainer(TrainerProtocol, ABC):
    """Base trainer for HuggingFace transformer sequence classification models.

    Subclasses configure ``predictor_cls``, ``model_name`` and
    ``DEFAULT_TRAIN_ARGS``. All training orchestration is handled here.
    """

    predictor_cls: ClassVar[Type[HFPredictor]]
    model_name: ClassVar[str]
    DEFAULT_TRAIN_ARGS: ClassVar[TrainingArguments]

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
            raise ValueError("HFTrainer requires val_records for early stopping")
        train_ds = create_datasets(records, self.dataset)
        val_ds = create_datasets(val_records, self.dataset)

        for task in self.dataset.all_tasks:
            logger.info(
                f"Starting {self.predictor_cls.name} training for {task.name}..."
            )
            save_path = self.predictor_cls._get_model_path(task, self.dataset)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            self._train_task(train_ds, val_ds, task, save_path, epochs)
            logger.info(f"{task.name} model training complete!")

        return self.predictor_cls.load_model(self.dataset)

    def _train_task(
        self,
        train_ds: Dataset,
        val_ds: Dataset,
        task,
        save_path: Path,
        epochs: int,
    ) -> _HFTrainer:
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=len(task.labels)
        )
        model.config.id2label = self.dataset.get_id2label(task.name)
        model.config.label2id = self.dataset.get_label2id(task.name)

        train_tok = train_ds.map(
            lambda x: tokenize_func(x, tokenizer, self.predictor_cls.MAX_LENGTH),
            batched=True,
        )
        val_tok = val_ds.map(
            lambda x: tokenize_func(x, tokenizer, self.predictor_cls.MAX_LENGTH),
            batched=True,
        )

        train_tok = train_tok.rename_column(task.name, "labels")
        val_tok = val_tok.rename_column(task.name, "labels")
        train_tok.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        val_tok.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

        args = self.DEFAULT_TRAIN_ARGS
        args.output_dir = str(save_path.parent / task.name)
        args.num_train_epochs = epochs
        trainer = _HFTrainer(
            model=model, args=args, train_dataset=train_tok, eval_dataset=val_tok
        )
        trainer.train()
        trainer.save_model(str(save_path))
        tokenizer.save_pretrained(str(save_path))
        return trainer
