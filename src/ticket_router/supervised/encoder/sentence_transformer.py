"""Sentence-transformers based dense text encoder."""

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from ticket_router.supervised.config import TORCH_DEVICE

from .base import TextEncoder

DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class SentenceTransformerEncoder(TextEncoder):
    """Text encoder using sentence-transformers pre-trained models.

    Defaults to paraphrase-multilingual-MiniLM-L12-v2 for multilingual support.
    """

    name = "sentence_transformer"
    need_fit = False

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        batch_size: int = 32,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device or TORCH_DEVICE
        self._model = None

    def _init_model(self):
        self._model = SentenceTransformer(self.model_name, device=self.device)

    def fit(self, texts: List[str]):
        raise NotImplementedError(
            "SentenceTransformerEncoder does not require fitting."
        )

    def transform(self, texts: List[str]) -> np.ndarray:
        self._init_model()

        if self._model is None:
            raise RuntimeError(
                "SentenceTransformerEncoder must be fit() before transform()"
            )
        return self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
