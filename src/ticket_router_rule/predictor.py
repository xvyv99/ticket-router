"""Rule-based Predictor and Trainer using weighted keyword matching."""

from __future__ import annotations

import math
import pickle
from logging import getLogger
from pathlib import Path
from typing import Dict, List

from ticket_router_base.config import OUTPUT_DIR
from ticket_router_base.data import BaseDataset
from ticket_router_base.predictor import Predictor, Trainer, register_model
from ticket_router_base.types import ErrorFlag, Prediction, Record

from .cfg import RuleBasedCfg
from .weighted_rule_model import (
    CandidateConfig,
    WeightedRuleModel,
    make_features,
    run_candidate_search,
)

logger = getLogger(__name__)

# Default candidate grids for hyperparameter search
_QUEUE_CANDIDATES = [
    CandidateConfig("queue_text_balanced", "text", 8, 0.20, 300),
    CandidateConfig("queue_text_strict", "text", 8, 0.35, 500),
    CandidateConfig("queue_text_meta_balanced", "text+meta", 8, 0.20, 300),
    CandidateConfig("queue_text_meta_strict", "text+meta", 8, 0.35, 500),
    CandidateConfig("queue_all_balanced", "text+meta+tags", 5, 0.20, 300),
    CandidateConfig("queue_all_strict", "text+meta+tags", 5, 0.35, 500),
]

_PRIORITY_CANDIDATES = [
    CandidateConfig("priority_text_balanced", "text", 8, 0.20, 300),
    CandidateConfig("priority_text_strict", "text", 8, 0.35, 500),
    CandidateConfig("priority_text_meta_balanced", "text+meta", 8, 0.20, 300),
    CandidateConfig("priority_text_meta_strict", "text+meta", 8, 0.35, 500),
    CandidateConfig("priority_all_balanced", "text+meta+tags", 5, 0.20, 300),
    CandidateConfig("priority_tags_only", "tags", 5, 0.10, 500),
]


def _softmax(scores: Dict[str, float]) -> Dict[str, float]:
    """Convert raw scores to probability-like confidences."""
    max_score = max(scores.values())
    # Subtract max for numerical stability
    exp_scores = {k: math.exp(v - max_score) for k, v in scores.items()}
    total = sum(exp_scores.values())
    return {k: v / total for k, v in exp_scores.items()}


def _get_model_path(save_dir: Path, task_name: str) -> Path:
    return save_dir / "models" / f"{task_name}.pkl"


def _save_model(model: WeightedRuleModel, save_dir: Path, task_name: str) -> None:
    path = _get_model_path(save_dir, task_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Saved {task_name} model to {path}")


def _load_model(save_dir: Path, task_name: str) -> WeightedRuleModel:
    path = _get_model_path(save_dir, task_name)
    with open(path, "rb") as f:
        return pickle.load(f)


@register_model
class RuleBasedPredictor(Predictor[RuleBasedCfg]):
    """Predictor using weighted keyword rules."""

    name = "rule-based"
    DEFAULT_SAVE_DIR = OUTPUT_DIR / "rule_based"

    dataset: BaseDataset
    cfg: RuleBasedCfg | None

    _models: Dict[str, WeightedRuleModel]
    _feature_modes: Dict[str, str]

    def __init__(
        self,
        dataset: BaseDataset,
        models: Dict[str, WeightedRuleModel],
        feature_modes: Dict[str, str],
        cfg: RuleBasedCfg,
    ):
        self.dataset = dataset
        self._models = models
        self._feature_modes = feature_modes
        self.cfg = cfg

    def predict(self, records: List[Record], run_id: int = 0) -> List[Prediction]:
        """Run rule-based prediction on records."""
        # Pre-compute features per task
        task_features: Dict[str, List[List[str]]] = {}
        for task_name, feature_mode in self._feature_modes.items():
            task_features[task_name] = [make_features(r, feature_mode) for r in records]

        predictions: List[Prediction] = []
        for i, rec in enumerate(records):
            labels: Dict[str, str] = {}
            confidences: Dict[str, float] = {}

            for task_name in self.dataset.task_names:
                model = self._models[task_name]
                features = task_features[task_name][i]
                pred_label, scores = model.predict_with_scores(features)
                labels[task_name] = pred_label
                confidences[task_name] = _softmax(scores)[pred_label]

            pred = Prediction(
                request_id=rec.request_id,
                labels=labels,
                discrete_features=rec.discrete_features,
                generation_target=None,
                sensitive_attributes=rec.sensitive_attributes,
                confidences=confidences,
                raw_output=None,
                error=ErrorFlag.SUCCESS,
            )
            predictions.append(pred)

        return predictions


class RuleBasedTrainer(Trainer):
    """Trainer for weighted keyword rule models."""

    dataset: BaseDataset
    cfg: RuleBasedCfg

    def __init__(self, dataset: BaseDataset, cfg: RuleBasedCfg | None = None):
        self.dataset = dataset
        self.cfg = cfg or RuleBasedCfg()

    def train(
        self,
        records: List[Record],
        val_records: List[Record] | None = None,
    ) -> RuleBasedPredictor:
        """Train rule-based models for all tasks.

        If candidate search is enabled and val_records are provided,
        runs hyperparameter search on the validation set first.
        """
        models: Dict[str, WeightedRuleModel] = {}
        feature_modes: Dict[str, str] = {}

        for task in self.dataset.all_tasks:
            task_name = task.name
            labels = task.labels

            # Determine best config
            if self.cfg.enable_candidate_search and val_records:
                candidates = (
                    _QUEUE_CANDIDATES if task_name == "queue" else _PRIORITY_CANDIDATES
                )
                logger.info(f"Running candidate search for {task_name}...")
                results, best_cfg = run_candidate_search(
                    train_records=records,
                    val_records=val_records,
                    task_name=task_name,
                    labels=labels,
                    candidates=candidates,
                )
                logger.info(
                    f"Best config for {task_name}: {best_cfg.name} "
                    f"(feature_mode={best_cfg.feature_mode}, "
                    f"min_count={best_cfg.min_count}, "
                    f"min_log_odds={best_cfg.min_log_odds}, "
                    f"max_features={best_cfg.max_features})"
                )
                feature_mode = best_cfg.feature_mode
                min_count = best_cfg.min_count
                min_log_odds = best_cfg.min_log_odds
                max_features = best_cfg.max_features
            else:
                # Use a sensible default (first candidate in the grid)
                default_cfg = (
                    _QUEUE_CANDIDATES[0]
                    if task_name == "queue"
                    else _PRIORITY_CANDIDATES[0]
                )
                feature_mode = default_cfg.feature_mode
                min_count = default_cfg.min_count
                min_log_odds = default_cfg.min_log_odds
                max_features = default_cfg.max_features

            # Train final model on all provided records
            features_list = [make_features(r, feature_mode) for r in records]
            labels_list = [r.labels.get(task_name, "") for r in records]

            model = WeightedRuleModel.fit(
                features_list=features_list,
                labels_list=labels_list,
                all_labels=labels,
                min_count=min_count,
                min_log_odds=min_log_odds,
                max_features=max_features,
            )

            models[task_name] = model
            feature_modes[task_name] = feature_mode

            # Save model
            _save_model(model, RuleBasedPredictor.DEFAULT_SAVE_DIR, task_name)

        return RuleBasedPredictor(
            dataset=self.dataset,
            models=models,
            feature_modes=feature_modes,
            cfg=self.cfg,
        )
