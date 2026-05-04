"""Configuration for supervised learning predictors."""

from dataclasses import dataclass
from typing import Literal

from ticket_router.base.cfg import Cfg

from .encoder import get_encoder, TextEncoder


@dataclass(frozen=True)
class SupervisedCfg(Cfg):
    """Supervised predictor configuration.

    Only encoder_type participates in cfg_id hashing; encoder-specific
    hyperparameters use their own defaults and are not part of the config.
    """

    encoder_type: Literal["tfidf", "sentence_transformer"]

    @property
    def encoder(self) -> TextEncoder:
        """Get the TextEncoder instance corresponding to the encoder_type."""
        return get_encoder(self.encoder_type)
