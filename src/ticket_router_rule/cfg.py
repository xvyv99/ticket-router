"""Configuration for rule-based predictor."""

from dataclasses import dataclass
from typing import Any, Dict

from ticket_router_base.cfg import Cfg


@dataclass(frozen=True)
class RuleBasedCfg(Cfg):
    """Configuration for the weighted rule-based predictor.

    When enable_candidate_search=True and val_records are provided to the Trainer,
    the trainer will run grid search over candidate configs on the validation set
    and select the best one per task by (accuracy + macro_f1).
    """

    # --- queue task ---
    queue_feature_mode: str = "text+meta+tags"
    queue_min_count: int = 5
    queue_min_log_odds: float = 0.20
    queue_max_features: int = 300

    # --- priority task ---
    priority_feature_mode: str = "text+meta"
    priority_min_count: int = 5
    priority_min_log_odds: float = 0.20
    priority_max_features: int = 300

    # --- candidate search ---
    enable_candidate_search: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "queue_feature_mode": self.queue_feature_mode,
            "queue_min_count": self.queue_min_count,
            "queue_min_log_odds": self.queue_min_log_odds,
            "queue_max_features": self.queue_max_features,
            "priority_feature_mode": self.priority_feature_mode,
            "priority_min_count": self.priority_min_count,
            "priority_min_log_odds": self.priority_min_log_odds,
            "priority_max_features": self.priority_max_features,
            "enable_candidate_search": self.enable_candidate_search,
        }
