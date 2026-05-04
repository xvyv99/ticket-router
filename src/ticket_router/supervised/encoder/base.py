"""Abstract base class for text encoders used in supervised learning."""

from abc import ABC, abstractmethod
from typing import ClassVar, List

import numpy as np


class TextEncoder(ABC):
    """Unified interface for text-to-vector encoding.

    Follows the sklearn fit/transform convention so that supervised trainers
    can swap encoders without code changes.
    """

    name: ClassVar[str]
    need_fit: ClassVar[bool]

    @abstractmethod
    def fit(self, texts: List[str]) -> None:
        """Fit the encoder on the given texts.

        For stateless encoders (e.g. sentence-transformers), this may only
        perform lazy model loading.
        """
        ...

    @abstractmethod
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts into a dense feature matrix."""
        ...

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit and transform in one call."""
        if self.need_fit:
            self.fit(texts)
        return self.transform(texts)
