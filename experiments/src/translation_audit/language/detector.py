"""Small, typed adapter around Lingua's offline language detector."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class Detection:
    """One language decision and the confidence assigned to the expected language."""

    predicted: str | None
    expected_confidence: float


class LanguageDetector(Protocol):
    """Interface used by the verification stage and deterministic tests."""

    def detect(self, text: str, expected: str) -> Detection: ...


class LinguaDetector:
    """Detect Portuguese, English, or Spanish locally with frozen settings."""

    def __init__(self, minimum_relative_distance: float) -> None:
        from lingua import Language, LanguageDetectorBuilder

        self._languages = {
            "en": Language.ENGLISH,
            "pt": Language.PORTUGUESE,
            "es": Language.SPANISH,
        }
        self._detector = (
            LanguageDetectorBuilder.from_languages(*self._languages.values())
            .with_minimum_relative_distance(minimum_relative_distance)
            .with_preloaded_language_models()
            .build()
        )

    def detect(self, text: str, expected: str) -> Detection:
        """Return an expected-language confidence without retaining response text."""
        expected_language = self._languages.get(expected)
        if expected_language is None:
            raise ValueError(f"Unsupported expected language {expected!r}")
        predicted_language = self._detector.detect_language_of(text)
        predicted = None
        if predicted_language is not None:
            predicted = next(
                (code for code, language in self._languages.items() if language == predicted_language),
                "other",
            )
        confidence_values = self._detector.compute_language_confidence_values(text)
        expected_confidence = next(
            (float(item.value) for item in confidence_values if item.language == expected_language),
            0.0,
        )
        return Detection(predicted=predicted, expected_confidence=expected_confidence)
