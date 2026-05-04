"""Interpretability evaluation for HFPredictor using transformers-interpret.

Computes token-level attributions for transformer-based sequence classification
models and aggregates them by predicted class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers_interpret import SequenceClassificationExplainer

from ticket_router.base.data import BaseDataset
from ticket_router.base.types import Record
from ticket_router.base.utils import combine_texts
from ticket_router.supervised.config import TORCH_DEVICE

logger = getLogger(__name__)


@dataclass(frozen=True)
class TokenAttribution:
    """Single token attribution score."""

    token: str
    score: float


@dataclass
class SampleAttribution:
    """Attribution results for a single sample on a single task."""

    request_id: str
    task_name: str
    text: str
    predicted_label: str
    true_label: str
    confidence: float
    # Tokens that push the prediction toward the predicted class (positive scores)
    top_positive: List[TokenAttribution] = field(default_factory=list)
    # Tokens that push the prediction away from the predicted class (negative scores)
    top_negative: List[TokenAttribution] = field(default_factory=list)


@dataclass
class ClassAttributionSummary:
    """Aggregated attribution summary for a single predicted class."""

    class_label: str
    sample_count: int
    # Token -> list of scores across all samples predicted as this class
    token_scores: Dict[str, List[float]] = field(default_factory=dict)

    def top_tokens(self, k: int = 10) -> List[Tuple[str, float]]:
        """Return top-k tokens by mean absolute attribution score."""
        scored = [
            (token, sum(scores) / len(scores))
            for token, scores in self.token_scores.items()
            if len(scores) > 0
        ]
        scored.sort(key=lambda x: abs(x[1]), reverse=True)
        return scored[:k]


@dataclass
class TaskAttributionReport:
    """Attribution report for a single task."""

    task_name: str
    sample_attributions: List[SampleAttribution] = field(default_factory=list)
    class_summaries: Dict[str, ClassAttributionSummary] = field(default_factory=dict)


class HFInterpretabilityEvaluator:
    """Evaluate token-level attributions for HFPredictor models.

    Usage:
        from ticket_router_eval.interpretability import HFInterpretabilityEvaluator
        from ticket_router.supervised import MBERTPredictor
        from ticket_router.base.data import get_dataset

        dataset = get_dataset("multilingual-customer-support")()
        predictor = MBERTPredictor.load_model(dataset)
        evaluator = HFInterpretabilityEvaluator(predictor, dataset)

        # Evaluate on first 10 test records
        report = evaluator.evaluate(test_records[:10], top_k=10)
    """

    def __init__(
        self,
        predictor,
        dataset: BaseDataset,
        device: str = "cpu",
        n_steps: int = 20,
        internal_batch_size: int = 10,
        predict_batch_size: int = 1,
    ):
        """Args:
        predictor: HFPredictor subclass instance (e.g. MBERTPredictor).
        dataset: Dataset descriptor for task definitions.
        device: Device for prediction pass ("cpu", "cuda", "auto").
            Default "cpu" because GPU is often occupied by vLLM.
            Attribution is always done on CPU to avoid OOM.
        n_steps: Number of steps for LIG attribution. Default 20 (faster than 50).
        internal_batch_size: Batch size for LIG internal forward passes. Default 10.
        predict_batch_size: Batch size for the prediction pass. Default 1 to
            minimize GPU memory when vLLM models are resident.
        """
        self.predictor = predictor
        self.dataset = dataset
        self.device = device
        self.n_steps = n_steps
        self.internal_batch_size = internal_batch_size
        self.predict_batch_size = predict_batch_size

        # Map task_name -> (model_path, id2label)
        self._task_info: Dict[str, Tuple[Path, Dict[int, str]]] = {}
        for task_name, model_path in predictor._model_paths.items():
            id2label = dataset.get_id2label(task_name)
            self._task_info[task_name] = (model_path, id2label)

    def evaluate(
        self,
        records: List[Record],
        top_k: int = 10,
        max_samples: int | None = None,
        task_names: List[str] | None = None,
    ) -> Dict[str, TaskAttributionReport]:
        """Compute attributions for selected tasks and aggregate by class.

        Args:
            records: Test records to evaluate.
            top_k: Number of top positive/negative tokens to keep per sample.
            max_samples: If set, limit evaluation to first N samples per task.
            task_names: If set, only evaluate these task names.

        Returns:
            Dict mapping task_name -> TaskAttributionReport.
        """
        if max_samples is not None:
            records = records[:max_samples]

        reports: Dict[str, TaskAttributionReport] = {}
        for task_name, (model_path, id2label) in self._task_info.items():
            if task_names is not None and task_name not in task_names:
                continue
            logger.info(f"Running interpretability for task: {task_name}")
            report = self._evaluate_task(
                task_name, model_path, id2label, records, top_k
            )
            reports[task_name] = report

        return reports

    def _evaluate_task(
        self,
        task_name: str,
        model_path: Path,
        id2label: Dict[int, str],
        records: List[Record],
        top_k: int,
    ) -> TaskAttributionReport:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        texts = combine_texts(records)

        # --- Prediction pass ---
        use_gpu = self.device == "cuda" or (
            self.device == "auto" and torch.cuda.is_available()
        )

        if use_gpu:
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            model.eval()
            model.to(TORCH_DEVICE)
            predict_device = TORCH_DEVICE
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            model.eval()
            predict_device = torch.device("cpu")
            model = model.to(predict_device)

        pred_ids: List[int] = []
        confidences: List[float] = []

        for i in range(0, len(texts), self.predict_batch_size):
            batch_texts = texts[i : i + self.predict_batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.predictor.MAX_LENGTH,
            )
            inputs = {k: v.to(predict_device) for k, v in inputs.items()}
            with torch.inference_mode():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)
                for j in range(len(batch_texts)):
                    pred_id = int(torch.argmax(probs[j]))
                    confidence = float(probs[j][pred_id])
                    pred_ids.append(pred_id)
                    confidences.append(confidence)

        # --- Attribution pass (always CPU to avoid OOM) ---
        explainer = SequenceClassificationExplainer(model, tokenizer)
        # xlm-roberta has type_vocab_size=1, but transformers-interpret
        # incorrectly sets accepts_token_type_ids=True, causing LIG interpolation
        # to produce out-of-range indices. Force it to False.
        if (
            hasattr(model, "config")
            and getattr(model.config, "model_type", "") == "xlm-roberta"
        ):
            explainer.accepts_token_type_ids = False

        sample_attributions: List[SampleAttribution] = []
        class_summaries: Dict[str, ClassAttributionSummary] = {}

        skipped_count = 0
        for rec, text in tqdm(
            zip(records, texts),
            total=len(records),
            desc=f"Attribution [{task_name}]",
        ):
            true_label = rec.labels.get(task_name, "")
            idx = len(sample_attributions) + skipped_count
            pred_id = pred_ids[idx]
            confidence = confidences[idx]
            predicted_label = id2label.get(pred_id, str(pred_id))
            # print(f"pred_id: {pred_id}, true_label: {true_label}, text: {text}")
            # print(f"predicted_label: {predicted_label}, confidence: {confidence:.4f}")
            # print(id2label)

            try:
                raw_attrs = explainer(
                    text,
                    index=pred_id,
                    n_steps=self.n_steps,
                    internal_batch_size=self.internal_batch_size,
                )
            except (RuntimeError, IndexError) as e:
                # transformers-interpret may produce out-of-bounds indices for
                # certain texts (e.g. special chars, very long tokens) leading to
                # CUDA/CPU embedding lookup failures. Skip and continue.
                logger.warning(
                    f"Attribution failed for {rec.request_id} (pred={predicted_label}): {e}"
                )
                skipped_count += 1
                continue

            # Sort by score, skip special tokens
            positive = [
                (t, s) for t, s in raw_attrs if s > 0 and t not in ("[CLS]", "[SEP]")
            ]
            negative = [
                (t, s) for t, s in raw_attrs if s < 0 and t not in ("[CLS]", "[SEP]")
            ]
            positive.sort(key=lambda x: x[1], reverse=True)
            negative.sort(key=lambda x: x[1])

            sample_attr = SampleAttribution(
                request_id=rec.request_id,
                task_name=task_name,
                text=text,
                predicted_label=predicted_label,
                true_label=true_label,
                confidence=confidence,
                top_positive=[TokenAttribution(t, s) for t, s in positive[:top_k]],
                top_negative=[TokenAttribution(t, s) for t, s in negative[:top_k]],
            )
            sample_attributions.append(sample_attr)

            # Aggregate into class summary
            summary = class_summaries.setdefault(
                predicted_label,
                ClassAttributionSummary(class_label=predicted_label, sample_count=0),
            )
            summary.sample_count += 1
            for token, score in raw_attrs:
                if token in ("[CLS]", "[SEP]"):
                    continue
                summary.token_scores.setdefault(token, []).append(score)

        return TaskAttributionReport(
            task_name=task_name,
            sample_attributions=sample_attributions,
            class_summaries=class_summaries,
        )
