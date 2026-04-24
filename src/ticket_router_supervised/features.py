import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator, TransformerMixin


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


def build_tfidf_pipeline(
    max_features: int = 10000, n_components: int = 200
) -> Pipeline:
    return make_pipeline(
        TfidfVectorizer(
            max_features=max_features, ngram_range=(1, 2), sublinear_tf=True
        ),
        AdaptiveSVD(n_components=n_components, random_state=42),
        StandardScaler(with_mean=False),
    )
