"""Bilingual emotion feature extraction contracts."""

from translation_audit.emotion.backend import EmoAtlasWorkerBackend, EmotionBackend
from translation_audit.emotion.features import EMOTIONS, VALENCES, build_output_record

__all__ = [
    "EMOTIONS",
    "VALENCES",
    "EmoAtlasWorkerBackend",
    "EmotionBackend",
    "build_output_record",
]
