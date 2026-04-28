"""Abstract base for HuggingFace transformer-based predictors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple

from ticket_router_base.data import BaseDataset
from ticket_router_base.predictor import Predictor
from ticket_router_base.types import (
    ErrorFlag,
    Prediction,
    Record,
)


class HFPredictor(Predictor, ABC):
    """Base predictor for HuggingFace transformer sequence classification models.

    Subclasses must implement:
      - `_get_model_path(task)` — staticmethod returning the saved model Path.
      - `_predict_task(model_path, records, id2label)` — instance method for batch inference.
    """

    _model_paths: Dict[str, Path]
    dataset: BaseDataset

    def __init__(self, model_paths: Dict[str, Path], dataset: BaseDataset):
        self._model_paths = model_paths
        self.dataset = dataset

    @classmethod
    def load_model(cls, dataset: BaseDataset) -> HFPredictor:
        """Load fine-tuned task models from disk and return a predictor instance."""
        model_paths: Dict[str, Path] = {}
        for task in dataset.all_tasks:
            path = cls._get_model_path(task)
            if not path.exists():
                raise FileNotFoundError(f"Model for {task.name} not found at {path}")
            model_paths[task.name] = path
        return cls(model_paths=model_paths, dataset=dataset)

    @staticmethod
    @abstractmethod
    def _get_model_path(task) -> Path:
        """Return the saved model directory for a given task.

        Implemented by subclasses; typically returns
        ``MODEL_DIR / f"{task.name}_best"``.
        """
        ...

    @abstractmethod
    def _predict_task(
        self,
        model_path: Path,
        records: List[Record],
        id2label: Dict[int, str],
    ) -> List[Tuple[str, float]]:
        """Run batch inference for a single task.

        Implemented by subclasses; e.g. calls ``predict_xlm_roberta(...)``.
        """
        ...

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
