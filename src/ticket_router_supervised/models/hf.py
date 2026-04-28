"""Abstract base for HuggingFace transformer-based predictors."""

from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import ClassVar, Dict, List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ticket_router_base.config import MODEL_DIR as BASE_MODEL_DIR
from ticket_router_base.data import BaseDataset
from ticket_router_base.predictor import Predictor
from ticket_router_base.types import (
    ErrorFlag,
    Prediction,
    Record,
)
from ticket_router_base.utils import combine_texts

from ticket_router_supervised.config import TORCH_DEVICE


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
