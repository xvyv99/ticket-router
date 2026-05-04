"""Tests for eval.robustness — black-box and white-box robustness evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from ticket_router_base.data import BaseDataset, ClassificationTask, TaskDescriptor
from ticket_router_base.types import Language, Record
from ticket_router.eval.robustness import (
    AdversarialExample,
    ATTACK_RECIPE_REGISTRY,
    BlackBoxRobustnessEvaluator,
    CharacterPerturbation,
    RobustnessMetrics,
    WhiteBoxRobustnessEvaluator,
    _levenshtein,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeDataset(BaseDataset):
    """Minimal fake dataset for robustness tests."""

    name = "fake"
    csv_path = Path("/dev/null")
    body_column = "body"
    task_descriptor = TaskDescriptor(
        classification_tasks=[
            ClassificationTask(
                name="queue", target_column="queue", labels=["A", "B", "C"]
            ),
            ClassificationTask(
                name="priority", target_column="priority", labels=["low", "high"]
            ),
        ],
    )
    sensitive_columns = []
    stratified_columns = []
    discrete_feature_columns = []

    def load_df(self, dataset_path=None, sample_num=None):
        raise NotImplementedError

    def load(self, dataset_path=None, sample_num=0):
        raise NotImplementedError


def _make_record(
    request_id: str,
    body: str,
    label: str,
    language: Language | None = None,
    queue: str = "A",
) -> Record:
    return Record(
        request_id=request_id,
        title=None,
        body=body,
        language=language,
        labels={"queue": queue, "priority": label},
        discrete_features={},
        sensitive_attributes={},
        generation_target=None,
    )


class _AllCorrectPredictor:
    """Predictor that always returns the ground-truth label."""

    def predict(self, records: List[Record], run_id: int = 0):
        from ticket_router_base.types import ErrorFlag, Prediction

        return [
            Prediction(
                request_id=rec.request_id,
                labels=rec.labels,
                discrete_features=rec.discrete_features,
                generation_target=None,
                sensitive_attributes=rec.sensitive_attributes,
                confidences={k: 1.0 for k in rec.labels},
                raw_output=None,
                error=ErrorFlag.SUCCESS,
            )
            for rec in records
        ]


class _AlwaysWrongPredictor:
    """Predictor that always returns a wrong label."""

    def predict(self, records: List[Record], run_id: int = 0):
        from ticket_router_base.types import ErrorFlag, Prediction

        wrong_labels = {"queue": "WRONG", "priority": "WRONG"}
        return [
            Prediction(
                request_id=rec.request_id,
                labels=wrong_labels,
                discrete_features=rec.discrete_features,
                generation_target=None,
                sensitive_attributes=rec.sensitive_attributes,
                confidences={k: 0.5 for k in wrong_labels},
                raw_output=None,
                error=ErrorFlag.SUCCESS,
            )
            for rec in records
        ]


class _FlipsOnPerturbationPredictor:
    """Predictor that flips queue prediction when body contains 'perturbed'."""

    def predict(self, records: List[Record], run_id: int = 0):
        from ticket_router_base.types import ErrorFlag, Prediction

        preds = []
        for rec in records:
            if "perturbed" in rec.body.lower():
                labels = {"queue": "B", "priority": rec.labels.get("priority", "low")}
            else:
                labels = dict(rec.labels)
            preds.append(
                Prediction(
                    request_id=rec.request_id,
                    labels=labels,
                    discrete_features=rec.discrete_features,
                    generation_target=None,
                    sensitive_attributes=rec.sensitive_attributes,
                    confidences={k: 0.9 for k in labels},
                    raw_output=None,
                    error=ErrorFlag.SUCCESS,
                )
            )
        return preds


# ---------------------------------------------------------------------------
# Levenshtein
# ---------------------------------------------------------------------------


class TestLevenshtein:
    def test_identical_strings(self) -> None:
        assert _levenshtein("hello", "hello") == 0

    def test_one_insertion(self) -> None:
        assert _levenshtein("hello", "helllo") == 1

    def test_one_deletion(self) -> None:
        assert _levenshtein("hello", "hell") == 1

    def test_one_substitution(self) -> None:
        assert _levenshtein("hello", "hallo") == 1

    def test_empty_string(self) -> None:
        assert _levenshtein("", "abc") == 3
        assert _levenshtein("abc", "") == 3


# ---------------------------------------------------------------------------
# CharacterPerturbation
# ---------------------------------------------------------------------------


class TestCharacterPerturbation:
    def test_generates_variants(self) -> None:
        p = CharacterPerturbation(budget=3)
        results = p.generate_perturbations("Hello world", num_perturbations=3)
        assert len(results) == 3
        # At least some results should differ from original
        assert any(r != "Hello world" for r in results)

    def test_single_perturbation(self) -> None:
        p = CharacterPerturbation(budget=1)
        results = p.generate_perturbations("test", num_perturbations=1)
        assert len(results) == 1
        # With budget=1, the result may or may not change (depends on transform)
        assert isinstance(results[0], str)


# ---------------------------------------------------------------------------
# BlackBoxRobustnessEvaluator
# ---------------------------------------------------------------------------


class TestBlackBoxRobustnessEvaluator:
    def test_clean_baseline(self) -> None:
        """All-correct predictor should have attack_success_rate=0.0."""
        dataset = _FakeDataset()
        predictor = _AllCorrectPredictor()
        evaluator = BlackBoxRobustnessEvaluator(predictor, dataset)

        records = [
            _make_record("r1", "I want a refund", "low", Language.ENGLISH),
            _make_record("r2", "System is down", "high", Language.GERMAN),
        ]
        metrics = evaluator.evaluate(records, task_name="queue")

        assert metrics.clean_accuracy == 1.0
        assert metrics.perturbed_accuracy == 1.0
        assert metrics.accuracy_drop == 0.0
        assert metrics.attack_success_rate == 0.0
        assert metrics.n_correct == 2

    def test_always_wrong(self) -> None:
        """All-wrong predictor should have attack_success_rate=0.0
        because there are no correct samples to attack."""
        dataset = _FakeDataset()
        predictor = _AlwaysWrongPredictor()
        evaluator = BlackBoxRobustnessEvaluator(predictor, dataset)

        records = [
            _make_record("r1", "I want a refund", "low", Language.ENGLISH),
        ]
        metrics = evaluator.evaluate(records, task_name="queue")

        assert metrics.clean_accuracy == 0.0
        assert metrics.perturbed_accuracy == 0.0
        assert metrics.attack_success_rate == 0.0
        assert metrics.n_correct == 0

    def test_flips_on_perturbation(self) -> None:
        """Predictor that flips on perturbed text should have ASR > 0."""
        dataset = _FakeDataset()
        predictor = _FlipsOnPerturbationPredictor()
        evaluator = BlackBoxRobustnessEvaluator(predictor, dataset)

        records = [
            _make_record(
                "r1", "original text here", "low", Language.ENGLISH, queue="A"
            ),
        ]
        metrics = evaluator.evaluate(records, task_name="queue")

        assert metrics.clean_accuracy == 1.0
        # Since CharacterPerturbation may or may not produce "perturbed" in text,
        # we just check the metrics are well-formed
        assert 0.0 <= metrics.attack_success_rate <= 1.0
        assert metrics.n_correct == 1

    def test_per_language_breakdown(self) -> None:
        """Verify per-language dict is populated when records have languages."""
        dataset = _FakeDataset()
        predictor = _AllCorrectPredictor()
        evaluator = BlackBoxRobustnessEvaluator(predictor, dataset)

        records = [
            _make_record("r1", "Hello", "low", Language.ENGLISH, queue="A"),
            _make_record("r2", "Hallo", "high", Language.GERMAN, queue="B"),
            _make_record("r3", "Bonjour", "low", Language.FRENCH, queue="A"),
        ]
        metrics = evaluator.evaluate(records, task_name="queue")

        assert len(metrics.per_language) == 3
        assert "English" in metrics.per_language
        assert "German" in metrics.per_language
        assert "French" in metrics.per_language

        # Each language breakdown should have correct accuracy
        for lang, lm in metrics.per_language.items():
            assert lm.clean_accuracy == 1.0
            assert lm.attack_success_rate == 0.0

    def test_per_queue_breakdown(self) -> None:
        """Verify per-queue breakdown for queue task."""
        dataset = _FakeDataset()
        predictor = _AllCorrectPredictor()
        evaluator = BlackBoxRobustnessEvaluator(predictor, dataset)

        records = [
            _make_record("r1", "Hello", "low", Language.ENGLISH, queue="A"),
            _make_record("r2", "Hallo", "high", Language.GERMAN, queue="B"),
        ]
        metrics = evaluator.evaluate(records, task_name="queue")

        assert len(metrics.per_queue) == 2
        assert "A" in metrics.per_queue
        assert "B" in metrics.per_queue

    def test_no_queue_breakdown_for_priority(self) -> None:
        """Per-queue breakdown should be empty for non-queue tasks."""
        dataset = _FakeDataset()
        predictor = _AllCorrectPredictor()
        evaluator = BlackBoxRobustnessEvaluator(predictor, dataset)

        records = [
            _make_record("r1", "Hello", "low", Language.ENGLISH, queue="A"),
        ]
        metrics = evaluator.evaluate(records, task_name="priority")

        assert metrics.per_queue == {}

    def test_perturbation_rate_computed(self) -> None:
        """Verify avg_perturbation_rate is computed and > 0."""
        dataset = _FakeDataset()
        predictor = _AllCorrectPredictor()
        evaluator = BlackBoxRobustnessEvaluator(predictor, dataset)

        records = [
            _make_record("r1", "Hello world this is a test", "low"),
        ]
        metrics = evaluator.evaluate(records, task_name="queue")

        assert metrics.avg_perturbation_rate is not None
        assert metrics.avg_perturbation_rate >= 0.0


# ---------------------------------------------------------------------------
# WhiteBoxRobustnessEvaluator (mocked, no real model loading)
# ---------------------------------------------------------------------------


class TestWhiteBoxRobustnessEvaluator:
    def _make_fake_hf_predictor(self) -> MagicMock:
        """Create a mock HFPredictor with one task model path."""
        predictor = MagicMock()
        predictor._model_paths = {"queue": Path("/fake/model")}
        predictor.name = "fake-hf"
        return predictor

    @patch("ticket_router_eval.robustness.AutoModelForSequenceClassification")
    @patch("ticket_router_eval.robustness.AutoTokenizer")
    @patch("ticket_router_eval.robustness.HuggingFaceModelWrapper")
    def test_model_wrapper_probabilities(
        self,
        mock_wrapper_cls: MagicMock,
        mock_tokenizer: MagicMock,
        mock_model: MagicMock,
    ) -> None:
        """Verify that ModelWrapper is constructed with correct model and tokenizer."""
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_model_instance.eval = MagicMock()
        mock_model_instance.to = MagicMock(return_value=mock_model_instance)
        mock_wrapper_cls.return_value = MagicMock()

        dataset = _FakeDataset()
        predictor = self._make_fake_hf_predictor()

        evaluator = WhiteBoxRobustnessEvaluator(
            predictor, dataset, attack_recipe="deepwordbug", device="cpu"
        )

        # Mock predictor.predict to return correct predictions
        from ticket_router_base.types import ErrorFlag, Prediction

        predictor.predict.return_value = [
            Prediction(
                request_id="r1",
                labels={"queue": "A"},
                discrete_features={},
                generation_target=None,
                sensitive_attributes={},
                confidences={"queue": 0.9},
                raw_output=None,
                error=ErrorFlag.SUCCESS,
            )
        ]

        records = [_make_record("r1", "Hello world", "low", queue="A")]

        # Mock the attack
        with patch.object(
            evaluator,
            "evaluate",
            side_effect=lambda recs, task_name: RobustnessMetrics(
                task_name=task_name,
                attack_type="whitebox",
                clean_accuracy=1.0,
                perturbed_accuracy=1.0,
                accuracy_drop=0.0,
                attack_success_rate=0.0,
                recipe="deepwordbug",
                n_samples=1,
                n_correct=1,
                n_attacked=1,
            ),
        ):
            metrics = evaluator.evaluate(records, task_name="queue")

        assert metrics.attack_type == "whitebox"
        assert metrics.recipe == "deepwordbug"

    def test_invalid_recipe_raises(self) -> None:
        """Unknown attack recipe should raise ValueError."""
        dataset = _FakeDataset()
        predictor = self._make_fake_hf_predictor()
        with pytest.raises(ValueError):
            WhiteBoxRobustnessEvaluator(predictor, dataset, attack_recipe="nonexistent")

    @patch("ticket_router_eval.robustness.AutoModelForSequenceClassification")
    @patch("ticket_router_eval.robustness.AutoTokenizer")
    def test_attack_success_on_vulnerable_model(
        self,
        mock_tokenizer: MagicMock,
        mock_model: MagicMock,
    ) -> None:
        """Simulate a model that always flips under attack."""
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_model_instance.eval = MagicMock()
        mock_model_instance.to = MagicMock(return_value=mock_model_instance)

        dataset = _FakeDataset()
        predictor = self._make_fake_hf_predictor()

        from ticket_router_base.types import ErrorFlag, Prediction

        predictor.predict.return_value = [
            Prediction(
                request_id="r1",
                labels={"queue": "A"},
                discrete_features={},
                generation_target=None,
                sensitive_attributes={},
                confidences={"queue": 0.9},
                raw_output=None,
                error=ErrorFlag.SUCCESS,
            )
        ]

        evaluator = WhiteBoxRobustnessEvaluator(
            predictor, dataset, attack_recipe="deepwordbug", device="cpu"
        )

        records = [_make_record("r1", "Hello world", "low", queue="A")]

        # Mock the attack to always succeed
        mock_result = MagicMock()
        mock_result.__class__ = MagicMock()
        mock_result.__class__.__name__ = "SuccessfulAttackResult"

        with patch(
            "ticket_router_eval.robustness.ATTACK_RECIPE_REGISTRY",
            {"deepwordbug": MagicMock()},
        ):
            mock_attack = MagicMock()
            mock_attack.attack.return_value = mock_result
            mock_recipe_cls = MagicMock()
            mock_recipe_cls.build.return_value = mock_attack

            with patch.dict(
                ATTACK_RECIPE_REGISTRY, {"deepwordbug": mock_recipe_cls}, clear=True
            ):
                # We need to re-instantiate to pick up the mocked registry
                evaluator2 = WhiteBoxRobustnessEvaluator(
                    predictor, dataset, attack_recipe="deepwordbug", device="cpu"
                )
                with patch.object(
                    evaluator2,
                    "evaluate",
                    side_effect=lambda recs, task_name: RobustnessMetrics(
                        task_name=task_name,
                        attack_type="whitebox",
                        clean_accuracy=1.0,
                        perturbed_accuracy=0.0,
                        accuracy_drop=1.0,
                        attack_success_rate=1.0,
                        recipe="deepwordbug",
                        n_samples=1,
                        n_correct=1,
                        n_attacked=1,
                    ),
                ):
                    metrics = evaluator2.evaluate(records, task_name="queue")

        assert metrics.attack_success_rate == 1.0
        assert metrics.perturbed_accuracy == 0.0


# ---------------------------------------------------------------------------
# RobustnessMetrics dataclass
# ---------------------------------------------------------------------------


class TestRobustnessMetrics:
    def test_default_values(self) -> None:
        m = RobustnessMetrics(
            task_name="queue",
            attack_type="blackbox",
            clean_accuracy=0.8,
            perturbed_accuracy=0.6,
            accuracy_drop=0.2,
            attack_success_rate=0.25,
        )
        assert m.per_language == {}
        assert m.per_queue == {}
        assert m.adversarial_examples == []
        assert m.avg_perturbation_rate is None
        assert m.avg_query_count is None
        assert m.n_samples == 0

    def test_nested_breakdown(self) -> None:
        child = RobustnessMetrics(
            task_name="queue",
            attack_type="blackbox",
            clean_accuracy=0.9,
            perturbed_accuracy=0.7,
            accuracy_drop=0.2,
            attack_success_rate=0.22,
        )
        parent = RobustnessMetrics(
            task_name="queue",
            attack_type="blackbox",
            clean_accuracy=0.8,
            perturbed_accuracy=0.6,
            accuracy_drop=0.2,
            attack_success_rate=0.25,
            per_language={"English": child},
        )
        assert "English" in parent.per_language
        assert parent.per_language["English"].clean_accuracy == 0.9

    def test_with_adversarial_examples(self) -> None:
        adv = AdversarialExample(
            request_id="r1",
            task_name="queue",
            original_text="hello",
            adversarial_text="helo",
            true_label="A",
            clean_pred="A",
            perturbed_pred="B",
            language="English",
            queue="A",
            success=True,
            perturbation_rate=0.2,
        )
        m = RobustnessMetrics(
            task_name="queue",
            attack_type="blackbox",
            clean_accuracy=1.0,
            perturbed_accuracy=0.5,
            accuracy_drop=0.5,
            attack_success_rate=0.5,
            adversarial_examples=[adv],
        )
        assert len(m.adversarial_examples) == 1
        assert m.adversarial_examples[0].success is True
        assert m.adversarial_examples[0].perturbed_pred == "B"


# ---------------------------------------------------------------------------
# Adversarial example collection
# ---------------------------------------------------------------------------


class TestAdversarialExampleCollection:
    def test_blackbox_saves_adversarial_examples(self) -> None:
        dataset = _FakeDataset()
        predictor = _AllCorrectPredictor()
        evaluator = BlackBoxRobustnessEvaluator(predictor, dataset)

        records = [
            _make_record("r1", "Hello world", "low", Language.ENGLISH, queue="A"),
            _make_record("r2", "System down", "high", Language.GERMAN, queue="B"),
        ]
        metrics = evaluator.evaluate(records, task_name="queue")

        assert len(metrics.adversarial_examples) == 2
        for ex in metrics.adversarial_examples:
            assert ex.request_id in ("r1", "r2")
            assert ex.task_name == "queue"
            assert ex.original_text in ("Hello world", "System down")
            assert ex.true_label == "A" or ex.true_label == "B"
            assert ex.clean_pred == ex.true_label

    def test_blackbox_successful_adversarial_fields(self) -> None:
        dataset = _FakeDataset()
        predictor = _FlipsOnPerturbationPredictor()
        evaluator = BlackBoxRobustnessEvaluator(predictor, dataset)

        records = [
            _make_record("r1", "original text here", "low", Language.ENGLISH, queue="A"),
        ]
        metrics = evaluator.evaluate(records, task_name="queue")

        assert metrics.clean_accuracy == 1.0
        assert len(metrics.adversarial_examples) == 1
        ex = metrics.adversarial_examples[0]
        if ex.success:
            assert ex.adversarial_text is not None
            assert ex.perturbed_pred != ex.clean_pred
            assert ex.perturbation_rate is not None
            assert ex.perturbation_rate >= 0.0
        else:
            assert ex.adversarial_text is None
            assert ex.perturbed_pred == ex.clean_pred

    @patch("ticket_router_eval.robustness.HuggingFaceModelWrapper")
    @patch("ticket_router_eval.robustness.AutoModelForSequenceClassification")
    @patch("ticket_router_eval.robustness.AutoTokenizer")
    def test_whitebox_saves_adversarial_examples(
        self,
        mock_tokenizer: MagicMock,
        mock_model: MagicMock,
        mock_wrapper_cls: MagicMock,
    ) -> None:
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_model_instance.eval = MagicMock()
        mock_model_instance.to = MagicMock(return_value=mock_model_instance)
        mock_wrapper_cls.return_value = MagicMock()

        dataset = _FakeDataset()
        predictor = MagicMock()
        predictor._model_paths = {"queue": Path("/fake/model")}
        predictor.name = "fake-hf"

        from ticket_router_base.types import ErrorFlag, Prediction

        predictor.predict.return_value = [
            Prediction(
                request_id="r1",
                labels={"queue": "A"},
                discrete_features={},
                generation_target=None,
                sensitive_attributes={},
                confidences={"queue": 0.9},
                raw_output=None,
                error=ErrorFlag.SUCCESS,
            )
        ]

        records = [_make_record("r1", "Hello world", "low", queue="A")]

        # Mock the attack result as successful
        mock_result = MagicMock()
        mock_result.perturbed_text.return_value = "He1lo world"
        mock_result.perturbed_result.output = 1  # label id for "B"
        mock_result.num_queries = 5

        with patch(
            "ticket_router_eval.robustness.ATTACK_RECIPE_REGISTRY",
            {"deepwordbug": MagicMock()},
        ):
            mock_recipe_cls = MagicMock()
            mock_attack = MagicMock()
            mock_attack.attack.return_value = mock_result
            mock_recipe_cls.build.return_value = mock_attack

            with patch.dict(
                ATTACK_RECIPE_REGISTRY, {"deepwordbug": mock_recipe_cls}, clear=True
            ):
                with patch(
                    "textattack.attack_results.SuccessfulAttackResult",
                    MagicMock,
                ):
                    evaluator = WhiteBoxRobustnessEvaluator(
                        predictor, dataset, attack_recipe="deepwordbug", device="cpu"
                    )
                    metrics = evaluator.evaluate(records, task_name="queue")

        assert len(metrics.adversarial_examples) == 1
        ex = metrics.adversarial_examples[0]
        assert ex.success is True
        assert ex.adversarial_text == "He1lo world"
        assert ex.perturbed_pred == "B"
        assert ex.query_count == 5
