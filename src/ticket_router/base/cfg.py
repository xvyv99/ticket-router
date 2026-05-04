"""Configuration base class for predictors.

Each predictor declares its own frozen dataclass Cfg subclass alongside its implementation.
Serialization and deterministic cfg_id hashing are centralized in the base class.
"""

from abc import ABC
from dataclasses import dataclass, asdict
from typing import Any
import hashlib
import json

@dataclass(frozen=True)
class Cfg(ABC):
    """Base configuration class for predictors.

    Subclasses are frozen dataclasses declared alongside their Predictor.
    Serialization and cfg_id hashing logic lives in the base class.
    """

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation."""
        return asdict(self)

    def cfg_id(self) -> str:
        """Deterministic 8-char hex hash."""
        payload = json.dumps(
            self.to_dict(), sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
        return hashlib.md5(payload.encode()).hexdigest()[:8]
