"""Robustness evaluation using TextAttack.

Provides black-box character-level perturbation for Rule-Based models
and white-box gradient attacks for HF transformer models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Dict, List, Type

import numpy as np
import torch
from tqdm import tqdm

from textattack import Attack
from textattack.shared import AttackedText
from textattack.attack_recipes import (
    BAEGarg2019,
    DeepWordBugGao2018,
    Pruthi2019,
)
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.transformations import (
    CompositeTransformation,
    WordSwapHomoglyphSwap,
    WordSwapNeighboringCharacterSwap,
    WordSwapRandomCharacterDeletion,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ticket_router_base.data import BaseDataset
from ticket_router_base.types import Record

from .custom_recipe import TextFoolerJin2019Torch

logger = getLogger(__name__)

# Attack recipe registry for white-box attacks
ATTACK_RECIPE_REGISTRY: Dict[str, Type] = {
    "textfooler": TextFoolerJin2019Torch,
    "bae": BAEGarg2019,
    "deepwordbug": DeepWordBugGao2018,
    "pruthi": Pruthi2019,
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class RobustnessMetrics:
    """Robustness metrics for a single task."""

    task_name: str
    attack_type: str  # "blackbox" or "whitebox"

    clean_accuracy: float
    perturbed_accuracy: float
    accuracy_drop: float
    attack_success_rate: float

    # Per-language breakdown
    per_language: Dict[str, "RobustnessMetrics"] = field(default_factory=dict)

    # Per-queue breakdown (only populated for queue task)
    per_queue: Dict[str, "RobustnessMetrics"] = field(default_factory=dict)

    # Black-box specific: average Levenshtein distance / original length
    avg_perturbation_rate: float | None = None

    # White-box specific
    recipe: str | None = None
    avg_query_count: float | None = None

    # Sample counts
    n_samples: int = 0
    n_correct: int = 0
    n_attacked: int = 0


# ---------------------------------------------------------------------------
# Perturbation strategies
# ---------------------------------------------------------------------------


class PerturbationStrategy(ABC):
    """Base class for text perturbation strategies."""

    @abstractmethod
    def generate_perturbations(
        self, text: str, num_perturbations: int = 1
    ) -> List[str]:
        """Generate perturbed versions of the input text.

        Args:
            text: Original text to perturb.
            num_perturbations: Number of perturbed variants to generate.

        Returns:
            List of perturbed text strings.
        """
        raise NotImplementedError()


class CharacterPerturbation(PerturbationStrategy):
    """Character-level perturbation using TextAttack transformations.

    Combines random character deletion, neighboring character swap, and
    homoglyph swap. Each transformation is applied iteratively up to the
    configured budget.
    """

    def __init__(self, budget: int = 3):
        self.budget = budget
        self.transformation = CompositeTransformation(
            [
                WordSwapRandomCharacterDeletion(),
                WordSwapNeighboringCharacterSwap(),
                WordSwapHomoglyphSwap(),
            ]
        )

    def generate_perturbations(
        self, text: str, num_perturbations: int = 1
    ) -> List[str]:
        results: List[str] = []
        for _ in range(num_perturbations):
            attacked = AttackedText(text)
            # Apply transformations iteratively up to budget
            for _step in range(self.budget):
                transformed = self.transformation(attacked)
                if not transformed:
                    break
                attacked = transformed[0]
            results.append(attacked.text)
        return results


class WordPerturbation(PerturbationStrategy):
    """Word-level perturbation using TextAttack WordDeletion.

    This is kept simple and English-centric; CharacterPerturbation is
    preferred for multilingual datasets.
    """

    def __init__(self, budget: int = 3):
        self.budget = budget
        from textattack.transformations import WordDeletion

        self.transformation = WordDeletion()

    def generate_perturbations(
        self, text: str, num_perturbations: int = 1
    ) -> List[str]:
        results: List[str] = []
        for _ in range(num_perturbations):
            attacked = AttackedText(text)
            for _step in range(self.budget):
                transformed = self.transformation(attacked)
                if not transformed:
                    break
                attacked = transformed[0]
            results.append(attacked.text)
        return results


# ---------------------------------------------------------------------------
# Levenshtein distance helper
# ---------------------------------------------------------------------------


def _levenshtein(s1: str, s2: str) -> int:
    """Compute Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


# ---------------------------------------------------------------------------
# Black-box evaluator
# ---------------------------------------------------------------------------


class BlackBoxRobustnessEvaluator:
    """Black-box robustness evaluator using character-level perturbations.

    Works with any Predictor (Rule-Based, Supervised, Goal-Based).
    Only perturbs the ``body`` field of each record.
    """

    def __init__(
        self,
        predictor,
        dataset: BaseDataset,
        strategy: PerturbationStrategy | None = None,
        num_perturbations: int = 1,
    ):
        """Args:
        predictor: Any Predictor instance.
        dataset: Dataset descriptor for task definitions.
        strategy: Perturbation strategy. Default CharacterPerturbation.
        num_perturbations: Number of perturbed variants per sample.
        """
        self.predictor = predictor
        self.dataset = dataset
        self.strategy = strategy or CharacterPerturbation(budget=3)
        self.num_perturbations = num_perturbations

    def evaluate(self, records: List[Record], task_name: str) -> RobustnessMetrics:
        """Evaluate robustness on the given records for a single task.

        Args:
            records: Test records to evaluate.
            task_name: Task name (e.g. "queue", "priority").

        Returns:
            RobustnessMetrics with global + per-language + per-queue breakdowns.
        """
        n_total = len(records)

        # 1. Clean predictions
        clean_preds = self.predictor.predict(records)

        # 2. Identify correctly predicted samples
        correct_indices: List[int] = []
        correct_records: List[Record] = []
        for i, (rec, pred) in enumerate(zip(records, clean_preds)):
            true_label = rec.labels.get(task_name, "")
            pred_label = pred.labels.get(task_name, "")
            if true_label and pred_label == true_label:
                correct_indices.append(i)
                correct_records.append(rec)

        n_correct = len(correct_records)

        if n_correct == 0:
            return RobustnessMetrics(
                task_name=task_name,
                attack_type="blackbox",
                clean_accuracy=0.0,
                perturbed_accuracy=0.0,
                accuracy_drop=0.0,
                attack_success_rate=0.0,
                n_samples=n_total,
                n_correct=0,
                n_attacked=0,
            )

        # 3. Generate perturbations for correct samples
        perturbed_records: List[Record] = []
        perturbation_rates: List[float] = []
        for rec in tqdm(correct_records, desc=f"Perturbing [{task_name}]"):
            text = rec.body
            perturbed_texts = self.strategy.generate_perturbations(
                text, num_perturbations=self.num_perturbations
            )
            for pt in perturbed_texts:
                perturbed_rec = Record(
                    request_id=rec.request_id,
                    title=None,
                    body=pt,
                    language=rec.language,
                    labels=rec.labels,
                    discrete_features=rec.discrete_features,
                    sensitive_attributes=rec.sensitive_attributes,
                    generation_target=rec.generation_target,
                )
                perturbed_records.append(perturbed_rec)
                perturbation_rates.append(_levenshtein(text, pt) / max(len(text), 1))

        # 4. Perturbed predictions
        perturbed_preds = self.predictor.predict(perturbed_records)

        # 5. Compare clean vs perturbed
        sample_results: List[dict] = []
        for i, rec in enumerate(correct_records):
            clean_label = clean_preds[correct_indices[i]].labels.get(task_name, "")
            flipped = False
            start = i * self.num_perturbations
            end = start + self.num_perturbations
            for j in range(start, end):
                perturbed_label = perturbed_preds[j].labels.get(task_name, "")
                if perturbed_label != clean_label:
                    flipped = True
                    break
            sample_results.append(
                {
                    "language": rec.language.value if rec.language else "unknown",
                    "queue": rec.labels.get("queue", "unknown"),
                    "flipped": flipped,
                }
            )

        n_flipped = sum(r["flipped"] for r in sample_results)
        clean_acc = n_correct / n_total
        perturbed_acc = (n_correct - n_flipped) / n_total
        acc_drop = clean_acc - perturbed_acc
        asr = n_flipped / n_correct if n_correct > 0 else 0.0
        avg_perturb_rate = (
            sum(perturbation_rates) / len(perturbation_rates)
            if perturbation_rates
            else 0.0
        )

        # 6. Breakdowns
        per_language = self._compute_breakdown(
            records, correct_records, sample_results, task_name, "language"
        )
        per_queue = {}
        if task_name == "queue":
            per_queue = self._compute_breakdown(
                records, correct_records, sample_results, task_name, "queue"
            )

        return RobustnessMetrics(
            task_name=task_name,
            attack_type="blackbox",
            clean_accuracy=clean_acc,
            perturbed_accuracy=perturbed_acc,
            accuracy_drop=acc_drop,
            attack_success_rate=asr,
            per_language=per_language,
            per_queue=per_queue,
            avg_perturbation_rate=avg_perturb_rate,
            n_samples=n_total,
            n_correct=n_correct,
            n_attacked=n_correct,
        )

    @staticmethod
    def _compute_breakdown(
        all_records: List[Record],
        correct_records: List[Record],
        sample_results: List[dict],
        task_name: str,
        attr_key: str,
    ) -> Dict[str, RobustnessMetrics]:
        """Compute per-attribute breakdown metrics."""
        # Count totals per attribute
        attr_totals: Dict[str, int] = {}
        for rec in all_records:
            val = rec.language.value if rec.language else "unknown"
            if attr_key == "queue":
                val = rec.labels.get("queue", "unknown")
            attr_totals[val] = attr_totals.get(val, 0) + 1

        # Count correct per attribute
        attr_correct: Dict[str, int] = {}
        for rec in correct_records:
            val = rec.language.value if rec.language else "unknown"
            if attr_key == "queue":
                val = rec.labels.get("queue", "unknown")
            attr_correct[val] = attr_correct.get(val, 0) + 1

        # Count flipped per attribute
        attr_flipped: Dict[str, int] = {}
        for rec, sr in zip(correct_records, sample_results):
            val = rec.language.value if rec.language else "unknown"
            if attr_key == "queue":
                val = rec.labels.get("queue", "unknown")
            if sr["flipped"]:
                attr_flipped[val] = attr_flipped.get(val, 0) + 1

        breakdown: Dict[str, RobustnessMetrics] = {}
        for val in sorted(attr_totals.keys()):
            n_tot = attr_totals[val]
            n_cor = attr_correct.get(val, 0)
            n_fli = attr_flipped.get(val, 0)
            if n_cor == 0:
                continue
            c_acc = n_cor / n_tot
            p_acc = (n_cor - n_fli) / n_tot
            breakdown[val] = RobustnessMetrics(
                task_name=task_name,
                attack_type="blackbox",
                clean_accuracy=c_acc,
                perturbed_accuracy=p_acc,
                accuracy_drop=c_acc - p_acc,
                attack_success_rate=n_fli / n_cor,
                n_samples=n_tot,
                n_correct=n_cor,
                n_attacked=n_cor,
            )
        return breakdown


# ---------------------------------------------------------------------------
# White-box evaluator
# ---------------------------------------------------------------------------


class WhiteBoxRobustnessEvaluator:
    """White-box robustness evaluator for HF transformer models.

    Uses TextAttack's Attack framework with gradient-guided search.
    Requires the predictor to be an HFPredictor subclass.
    """

    def __init__(
        self,
        predictor,
        dataset: BaseDataset,
        attack_recipe: str = "textfooler",
        query_budget: int = 100,
        device: str = "cpu",
    ):
        """Args:
        predictor: HFPredictor subclass instance.
        dataset: Dataset descriptor.
        attack_recipe: One of "textfooler", "bae", "deepwordbug", "pruthi".
        query_budget: Maximum number of model queries per sample.
        device: Device for model inference ("cpu" or "cuda").
        """
        self.predictor = predictor
        self.dataset = dataset
        self.attack_recipe = attack_recipe
        self.query_budget = query_budget
        self.device = device

        if attack_recipe not in ATTACK_RECIPE_REGISTRY:
            raise ValueError(
                f"Unknown attack recipe: {attack_recipe}. "
                f"Choose from: {list(ATTACK_RECIPE_REGISTRY.keys())}"
            )

    def evaluate(self, records: List[Record], task_name: str) -> RobustnessMetrics:
        """Evaluate white-box robustness on the given records for a single task."""
        n_total = len(records)
        model_path = self.predictor._model_paths.get(task_name)
        if model_path is None:
            raise ValueError(f"No model path found for task: {task_name}")

        id2label = self.dataset.get_id2label(task_name)
        label2id = self.dataset.get_label2id(task_name)

        # 1. Clean predictions (using predictor's own predict method)
        clean_preds = self.predictor.predict(records)

        # 2. Identify correctly predicted samples
        correct_indices: List[int] = []
        correct_records: List[Record] = []
        for i, (rec, pred) in enumerate(zip(records, clean_preds)):
            true_label = rec.labels.get(task_name, "")
            pred_label = pred.labels.get(task_name, "")
            if true_label and pred_label == true_label:
                correct_indices.append(i)
                correct_records.append(rec)

        n_correct = len(correct_records)

        if n_correct == 0:
            return RobustnessMetrics(
                task_name=task_name,
                attack_type="whitebox",
                clean_accuracy=0.0,
                perturbed_accuracy=0.0,
                accuracy_drop=0.0,
                attack_success_rate=0.0,
                recipe=self.attack_recipe,
                n_samples=n_total,
                n_correct=0,
                n_attacked=0,
            )

        # 3. Build TextAttack model wrapper and attack
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        if self.device == "cuda" and torch.cuda.is_available():
            model = model.to("cuda")
        else:
            model = model.to("cpu")

        model_wrapper = HuggingFaceModelWrapper(model, tokenizer)
        recipe_cls = ATTACK_RECIPE_REGISTRY[self.attack_recipe]
        attack = recipe_cls.build(model_wrapper)

        # 4. Run attack on correct samples
        sample_results: List[dict] = []
        query_counts: List[int] = []

        for rec in tqdm(correct_records, desc=f"WhiteBox Attack [{task_name}]"):
            text = rec.body
            true_label = rec.labels.get(task_name, "")
            true_label_id = label2id.get(true_label, 0)

            result = attack.attack(text, true_label_id)

            # Determine if attack succeeded
            from textattack.attack_results import SuccessfulAttackResult

            flipped = isinstance(result, SuccessfulAttackResult)
            sample_results.append(
                {
                    "language": rec.language.value if rec.language else "unknown",
                    "queue": rec.labels.get("queue", "unknown"),
                    "flipped": flipped,
                }
            )
            logger.info(
                f"Original: {text} | Perturbed: {result.perturbed_text()} | "
                f"Flipped: {flipped}"
            )

            # Estimate query count from result internals
            qc = 0
            if hasattr(result, "num_queries"):
                qc = result.num_queries
            elif hasattr(attack, "search_method") and hasattr(
                attack.search_method, "get_num_queries"
            ):
                qc = attack.search_method.get_num_queries()
            query_counts.append(qc)

        n_flipped = sum(r["flipped"] for r in sample_results)
        clean_acc = n_correct / n_total
        perturbed_acc = (n_correct - n_flipped) / n_total
        acc_drop = clean_acc - perturbed_acc
        asr = n_flipped / n_correct if n_correct > 0 else 0.0
        avg_queries = sum(query_counts) / len(query_counts) if query_counts else 0.0

        # 5. Breakdowns
        per_language = self._compute_breakdown(
            records, correct_records, sample_results, task_name, "language"
        )
        per_queue = {}
        if task_name == "queue":
            per_queue = self._compute_breakdown(
                records, correct_records, sample_results, task_name, "queue"
            )

        return RobustnessMetrics(
            task_name=task_name,
            attack_type="whitebox",
            clean_accuracy=clean_acc,
            perturbed_accuracy=perturbed_acc,
            accuracy_drop=acc_drop,
            attack_success_rate=asr,
            per_language=per_language,
            per_queue=per_queue,
            recipe=self.attack_recipe,
            avg_query_count=avg_queries,
            n_samples=n_total,
            n_correct=n_correct,
            n_attacked=n_correct,
        )

    @staticmethod
    def _compute_breakdown(
        all_records: List[Record],
        correct_records: List[Record],
        sample_results: List[dict],
        task_name: str,
        attr_key: str,
    ) -> Dict[str, RobustnessMetrics]:
        """Compute per-attribute breakdown metrics."""
        attr_totals: Dict[str, int] = {}
        for rec in all_records:
            val = rec.language.value if rec.language else "unknown"
            if attr_key == "queue":
                val = rec.labels.get("queue", "unknown")
            attr_totals[val] = attr_totals.get(val, 0) + 1

        attr_correct: Dict[str, int] = {}
        for rec in correct_records:
            val = rec.language.value if rec.language else "unknown"
            if attr_key == "queue":
                val = rec.labels.get("queue", "unknown")
            attr_correct[val] = attr_correct.get(val, 0) + 1

        attr_flipped: Dict[str, int] = {}
        for rec, sr in zip(correct_records, sample_results):
            val = rec.language.value if rec.language else "unknown"
            if attr_key == "queue":
                val = rec.labels.get("queue", "unknown")
            if sr["flipped"]:
                attr_flipped[val] = attr_flipped.get(val, 0) + 1

        breakdown: Dict[str, RobustnessMetrics] = {}
        for val in sorted(attr_totals.keys()):
            n_tot = attr_totals[val]
            n_cor = attr_correct.get(val, 0)
            n_fli = attr_flipped.get(val, 0)
            if n_cor == 0:
                continue
            c_acc = n_cor / n_tot
            p_acc = (n_cor - n_fli) / n_tot
            breakdown[val] = RobustnessMetrics(
                task_name=task_name,
                attack_type="whitebox",
                clean_accuracy=c_acc,
                perturbed_accuracy=p_acc,
                accuracy_drop=c_acc - p_acc,
                attack_success_rate=n_fli / n_cor,
                n_samples=n_tot,
                n_correct=n_cor,
                n_attacked=n_cor,
            )
        return breakdown
