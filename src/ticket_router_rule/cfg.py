"""Configuration for rule-based predictor."""

from dataclasses import dataclass
from typing import Any, Dict

from ticket_router_base.cfg import Cfg


@dataclass(frozen=True)
class RuleBasedCfg(Cfg):
    """Configuration for the weighted rule-based predictor.

    The only exposed parameter is whether to run candidate search.
    When enabled, the trainer searches over a fixed grid on the validation set
    and selects the best config per task by (accuracy + macro_f1).
    When disabled, a sensible default config is used.
    """

    enable_candidate_search: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {"enable_candidate_search": self.enable_candidate_search}
