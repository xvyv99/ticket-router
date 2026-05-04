"""TF-IDF based text encoder with dimensionality reduction."""

from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .base import TextEncoder


class AdaptiveSVD(BaseEstimator, TransformerMixin):
    """TruncatedSVD that auto-adjusts n_components to min(n_components, n_features) at fit time."""

    def __init__(self, n_components: int = 200, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y=None):
        actual = min(self.n_components, X.shape[1])
        self.svd_ = TruncatedSVD(n_components=actual, random_state=self.random_state)
        self.svd_.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.svd_.transform(X)


class TfidfEncoder(TextEncoder):
    """Text encoder using TF-IDF + TruncatedSVD + StandardScaler."""

    name = "tfidf"
    need_fit = True

    def __init__(
        self,
        max_features: int = 10000,
        n_components: int = 200,
        ngram_range: tuple[int, int] = (1, 2),
        sublinear_tf: bool = True,
        random_state: int = 42,
    ):
        self.max_features = max_features
        self.n_components = n_components
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf
        self.random_state = random_state
        self._pipeline = None

    def fit(self, texts: List[str]):
        self._pipeline = make_pipeline(
            TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                sublinear_tf=self.sublinear_tf,
            ),
            AdaptiveSVD(n_components=self.n_components, random_state=self.random_state),
            StandardScaler(with_mean=False),
        )
        self._pipeline.fit(texts)

    def transform(self, texts: List[str]) -> np.ndarray:
        if self._pipeline is None:
            raise RuntimeError("TfidfEncoder must be fit() before transform()")
        return self._pipeline.transform(texts)
