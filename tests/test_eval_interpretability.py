"""Tests for eval.interpretability — token-level attribution evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest
import torch

from ticket_router_base.data import BaseDataset, ClassificationTask, TaskDescriptor
from ticket_router_base.types import Record
from ticket_router_eval.interpretability import (
    ClassAttributionSummary,
    HFInterpretabilityEvaluator,
    SampleAttribution,
    TaskAttributionReport,
    TokenAttribution,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeDataset(BaseDataset):
    """Minimal fake dataset for interpretability tests."""

    name = "fake"
    csv_path = Path("/dev/null")
    body_column = "body"
    task_descriptor = TaskDescriptor(
        classification_tasks=[
            ClassificationTask(name="queue", target_column="queue", labels=["A", "B"]),
        ],
    )
    sensitive_columns = []
    stratified_columns = []
    discrete_feature_columns = []

    def load_df(self, dataset_path=None, sample_num=None):
        raise NotImplementedError

    def load(self, dataset_path=None, sample_num=0):
        raise NotImplementedError


class _FakeHFPredictor:
    """Minimal mock predictor compatible with HFInterpretabilityEvaluator."""

    MAX_LENGTH = 128
    name = "fake-hf"

    def __init__(self, model_paths: Dict[str, Path]):
        self._model_paths = model_paths



def _make_record(request_id: str, body: str, label: str) -> Record:
    return Record(
        request_id=request_id,
        title=None,
        body=body,
        language=None,
        labels={"queue": label},
        discrete_features={},
        sensitive_attributes={},
        generation_target=None,
    )


# ---------------------------------------------------------------------------
# ClassAttributionSummary
# ---------------------------------------------------------------------------

class TestClassAttributionSummary:
    def test_top_tokens_sorts_by_mean_abs_score(self) -> None:
        summary = ClassAttributionSummary(class_label="A", sample_count=2)
        summary.token_scores = {
            "refund": [0.8, 0.6],      # mean=0.7, abs=0.7
            "urgent": [-0.9, -0.1],    # mean=-0.5, abs=0.5
            "hello": [0.1, 0.1],       # mean=0.1, abs=0.1
        }
        top = summary.top_tokens(k=2)

        assert len(top) == 2
        assert top[0][0] == "refund"
        assert top[0][1] == pytest.approx(0.7)
        assert top[1][0] == "urgent"
        assert top[1][1] == pytest.approx(-0.5)

    def test_top_tokens_empty(self) -> None:
        summary = ClassAttributionSummary(class_label="A", sample_count=0)
        assert summary.top_tokens(k=5) == []

    def test_top_tokens_k_larger_than_tokens(self) -> None:
        summary = ClassAttributionSummary(class_label="A", sample_count=1)
        summary.token_scores = {"only": [0.5]}
        top = summary.top_tokens(k=10)
        assert len(top) == 1


# ---------------------------------------------------------------------------
# Dataclass construction
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_token_attribution_immutable(self) -> None:
        ta = TokenAttribution(token="refund", score=0.5)
        assert ta.token == "refund"
        assert ta.score == pytest.approx(0.5)
        # frozen dataclass
        with pytest.raises(AttributeError):
            ta.score = 0.6  # type: ignore[misc]

    def test_sample_attribution_defaults(self) -> None:
        sa = SampleAttribution(
            request_id="r1",
            task_name="queue",
            text="I want a refund",
            predicted_label="A",
            true_label="A",
            confidence=0.9,
        )
        assert sa.top_positive == []
        assert sa.top_negative == []

    def test_task_attribution_report_defaults(self) -> None:
        report = TaskAttributionReport(task_name="queue")
        assert report.sample_attributions == []
        assert report.class_summaries == {}


# ---------------------------------------------------------------------------
# HFInterpretabilityEvaluator (mocked, no real model loading)
# ---------------------------------------------------------------------------

class TestHFInterpretabilityEvaluator:
    def _setup_mocks(self) -> tuple[MagicMock, MagicMock, MagicMock]:
        """Return (mock_tokenizer, mock_model, mock_explainer_instance)."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.ones(1, 10, dtype=torch.long),
            "attention_mask": torch.ones(1, 10, dtype=torch.long),
        }

        mock_model = MagicMock()
        # logits shape (batch, num_labels=2); predict class 1 with high confidence
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.tensor([[0.1, 2.0]])
        mock_model.return_value = mock_outputs
        # model.to("cpu") should return self so the variable stays the same mock
        mock_model.to.return_value = mock_model

        mock_explainer_instance = MagicMock()
        # Return attribution tuples: (token, score)
        mock_explainer_instance.return_value = [
            ("[CLS]", 0.0),
            ("refund", 0.8),
            ("please", -0.3),
            ("[SEP]", 0.0),
        ]

        return mock_tokenizer, mock_model, mock_explainer_instance

    @patch("ticket_router_eval.interpretability.SequenceClassificationExplainer")
    @patch("ticket_router_eval.interpretability.AutoModelForSequenceClassification")
    @patch("ticket_router_eval.interpretability.AutoTokenizer")
    def test_evaluate_single_task(
        self,
        mock_auto_tokenizer: MagicMock,
        mock_auto_model: MagicMock,
        mock_explainer_cls: MagicMock,
    ) -> None:
        mock_tokenizer, mock_model, mock_explainer = self._setup_mocks()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_explainer_cls.return_value = mock_explainer

        dataset = _FakeDataset()
        predictor = _FakeHFPredictor(model_paths={"queue": Path("/fake/model")})
        evaluator = HFInterpretabilityEvaluator(
            predictor,
            dataset,
            device="cpu",
            n_steps=10,
            internal_batch_size=5,
        )

        records = [
            _make_record("r1", "I want a refund", "A"),
        ]
        reports = evaluator.evaluate(records, top_k=2)

        assert "queue" in reports
        report = reports["queue"]
        assert len(report.sample_attributions) == 1

        sa = report.sample_attributions[0]
        assert sa.request_id == "r1"
        assert sa.predicted_label == "B"  # argmax of [0.1, 2.0] -> index 1 -> label "B"
        assert sa.true_label == "A"
        # [CLS] and [SEP] should be filtered out
        assert all(t.token not in ("[CLS]", "[SEP]") for t in sa.top_positive)
        assert all(t.token not in ("[CLS]", "[SEP]") for t in sa.top_negative)

        # Verify explainer called with correct n_steps and internal_batch_size
        mock_explainer.assert_called_once()
        _, kwargs = mock_explainer.call_args
        assert kwargs["n_steps"] == 10
        assert kwargs["internal_batch_size"] == 5

        # Class summary aggregated
        assert "B" in report.class_summaries
        summary = report.class_summaries["B"]
        assert summary.sample_count == 1
        assert "refund" in summary.token_scores
        assert "please" in summary.token_scores

    @patch("ticket_router_eval.interpretability.SequenceClassificationExplainer")
    @patch("ticket_router_eval.interpretability.AutoModelForSequenceClassification")
    @patch("ticket_router_eval.interpretability.AutoTokenizer")
    def test_task_names_filter(
        self,
        mock_auto_tokenizer: MagicMock,
        mock_auto_model: MagicMock,
        mock_explainer_cls: MagicMock,
    ) -> None:
        mock_tokenizer, mock_model, mock_explainer = self._setup_mocks()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_explainer_cls.return_value = mock_explainer

        dataset = _FakeDataset()
        predictor = _FakeHFPredictor(model_paths={"queue": Path("/fake/model")})
        evaluator = HFInterpretabilityEvaluator(predictor, dataset, device="cpu")

        records = [_make_record("r1", "hello", "A")]
        # Filter with non-existent task name -> empty reports
        reports = evaluator.evaluate(records, task_names=["nonexistent"])
        assert reports == {}

        # Filter with existing task name -> works
        reports = evaluator.evaluate(records, task_names=["queue"])
        assert "queue" in reports

    @patch("ticket_router_eval.interpretability.SequenceClassificationExplainer")
    @patch("ticket_router_eval.interpretability.AutoModelForSequenceClassification")
    @patch("ticket_router_eval.interpretability.AutoTokenizer")
    def test_max_samples_truncation(
        self,
        mock_auto_tokenizer: MagicMock,
        mock_auto_model: MagicMock,
        mock_explainer_cls: MagicMock,
    ) -> None:
        mock_tokenizer, mock_model, mock_explainer = self._setup_mocks()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_explainer_cls.return_value = mock_explainer

        dataset = _FakeDataset()
        predictor = _FakeHFPredictor(model_paths={"queue": Path("/fake/model")})
        evaluator = HFInterpretabilityEvaluator(predictor, dataset, device="cpu")

        records = [
            _make_record("r1", "hello", "A"),
            _make_record("r2", "world", "B"),
            _make_record("r3", "foo", "A"),
        ]
        reports = evaluator.evaluate(records, max_samples=2)
        assert len(reports["queue"].sample_attributions) == 2

    @patch("ticket_router_eval.interpretability.tqdm")
    @patch("ticket_router_eval.interpretability.SequenceClassificationExplainer")
    @patch("ticket_router_eval.interpretability.AutoModelForSequenceClassification")
    @patch("ticket_router_eval.interpretability.AutoTokenizer")
    def test_positive_negative_sorting(
        self,
        mock_auto_tokenizer: MagicMock,
        mock_auto_model: MagicMock,
        mock_explainer_cls: MagicMock,
        mock_tqdm: MagicMock,
    ) -> None:
        # Make tqdm transparent so it passes through the iterable unchanged
        mock_tqdm.side_effect = lambda iterable, **kwargs: iterable
        mock_tokenizer, mock_model, mock_explainer = self._setup_mocks()
        # Custom attribution with multiple positive/negative tokens
        mock_explainer.return_value = [
            ("[CLS]", 0.0),
            ("a", 0.1),
            ("b", 0.5),
            ("c", 0.3),
            ("d", -0.4),
            ("e", -0.2),
            ("f", -0.6),
            ("[SEP]", 0.0),
        ]
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_explainer_cls.return_value = mock_explainer

        dataset = _FakeDataset()
        predictor = _FakeHFPredictor(model_paths={"queue": Path("/fake/model")})
        evaluator = HFInterpretabilityEvaluator(predictor, dataset, device="cpu")

        records = [_make_record("r1", "text", "A")]
        reports = evaluator.evaluate(records, top_k=2)

        sa = reports["queue"].sample_attributions[0]
        # top_positive sorted descending by score
        assert [t.token for t in sa.top_positive] == ["b", "c"]
        assert [t.score for t in sa.top_positive] == pytest.approx([0.5, 0.3])
        # top_negative sorted ascending by score (most negative first)
        assert [t.token for t in sa.top_negative] == ["f", "d"]
        assert [t.score for t in sa.top_negative] == pytest.approx([-0.6, -0.4])

    @patch("ticket_router_eval.interpretability.tqdm")
    @patch("ticket_router_eval.interpretability.SequenceClassificationExplainer")
    @patch("ticket_router_eval.interpretability.AutoModelForSequenceClassification")
    @patch("ticket_router_eval.interpretability.AutoTokenizer")
    def test_explainer_failure_skips_sample(
        self,
        mock_auto_tokenizer: MagicMock,
        mock_auto_model: MagicMock,
        mock_explainer_cls: MagicMock,
        mock_tqdm: MagicMock,
    ) -> None:
        # Make tqdm transparent so it passes through the iterable unchanged
        mock_tqdm.side_effect = lambda iterable, **kwargs: iterable
        mock_tokenizer, mock_model, mock_explainer = self._setup_mocks()

        # Second call raises RuntimeError (simulating CUDA index OOB)
        mock_explainer.side_effect = [
            [("[CLS]", 0.0), ("ok", 0.5), ("[SEP]", 0.0)],
            RuntimeError("CUDA index out of bounds"),
            [("[CLS]", 0.0), ("fine", 0.3), ("[SEP]", 0.0)],
        ]

        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_explainer_cls.return_value = mock_explainer

        dataset = _FakeDataset()
        predictor = _FakeHFPredictor(model_paths={"queue": Path("/fake/model")})
        evaluator = HFInterpretabilityEvaluator(predictor, dataset, device="cpu")

        records = [
            _make_record("r1", "first", "A"),
            _make_record("r2", "second", "A"),
            _make_record("r3", "third", "A"),
        ]
        reports = evaluator.evaluate(records, top_k=2)

        report = reports["queue"]
        # r2 failed, so only 2 successful attributions
        assert len(report.sample_attributions) == 2
        assert report.sample_attributions[0].request_id == "r1"
        assert report.sample_attributions[1].request_id == "r3"
        # Class summary should only aggregate successful samples
        assert report.class_summaries["B"].sample_count == 2
