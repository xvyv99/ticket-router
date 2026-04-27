from typing import Dict, Type

from .base import TextEncoder
from .tfidf import TfidfEncoder
from .sentence_transformer import SentenceTransformerEncoder

TEXT_ENCODERS_LST = [TfidfEncoder, SentenceTransformerEncoder]

TEXT_ENCODERS: Dict[str, Type[TextEncoder]] = {
    encoder.name: encoder for encoder in TEXT_ENCODERS_LST
}

def get_encoder(name: str) -> TextEncoder:
    """Get the TextEncoder class corresponding to the given name."""
    if name not in TEXT_ENCODERS:
        raise ValueError(f"Unknown encoder type: {name}")
    return TEXT_ENCODERS[name]()

__all__ = ["TextEncoder", "TfidfEncoder", "SentenceTransformerEncoder"]
