"""Language-neutral schema and paired preservation features."""

import math
from collections.abc import Mapping, Sequence
from typing import Final

EMOTIONS: Final[tuple[str, ...]] = (
    "anger",
    "trust",
    "surprise",
    "disgust",
    "joy",
    "sadness",
    "fear",
    "anticipation",
)
VALENCES: Final[tuple[str, ...]] = ("positive", "negative", "ambivalent", "neutral")
IDENTIFIER_COLUMNS: Final[tuple[str, ...]] = (
    "pair_id",
    "participant_id",
    "cohort_id",
    "year",
    "question_index",
    "prompt_id",
    "prompt_family",
)
TEXT_COLUMNS: Final[tuple[str, ...]] = ("source_pt", "translation_en")


def language_feature_names(prefix: str) -> tuple[str, ...]:
    """Return the expected numeric feature names for one language."""
    emotion_names = tuple(
        name
        for emotion in EMOTIONS
        for name in (
            f"{prefix}_emotion_{emotion}_type_count",
            f"{prefix}_emotion_{emotion}_share",
        )
    )
    valence_names = tuple(
        name
        for valence in VALENCES
        for name in (
            f"{prefix}_valence_{valence}_node_count",
            f"{prefix}_valence_{valence}_share",
        )
    )
    return (
        f"{prefix}_semantic_node_count",
        f"{prefix}_semantic_edge_count",
        f"{prefix}_semantic_density",
        f"{prefix}_emotion_lexicon_coverage",
        *emotion_names,
        f"{prefix}_valence_lexicon_coverage",
        *valence_names,
    )


PT_FEATURES: Final[tuple[str, ...]] = language_feature_names("pt")
EN_FEATURES: Final[tuple[str, ...]] = language_feature_names("en")


def _as_float(value: object) -> float:
    if not isinstance(value, int | float):
        raise ValueError("Emotion worker returned a non-numeric feature")
    return float(value)


def _distribution(
    record: Mapping[str, object],
    prefix: str,
    family: str,
    labels: Sequence[str],
) -> list[float]:
    values = [_as_float(record[f"{prefix}_{family}_{label}_share"]) for label in labels]
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("Emotion worker returned an invalid probability distribution")
    return values


def _jensen_shannon_distance(left: Sequence[float], right: Sequence[float]) -> float:
    """Return base-2 Jensen-Shannon distance for possibly all-zero vectors."""
    left_total = sum(left)
    right_total = sum(right)
    if left_total == 0.0 and right_total == 0.0:
        return 0.0
    left_prob = [value / left_total if left_total else 0.0 for value in left]
    right_prob = [value / right_total if right_total else 0.0 for value in right]
    midpoint = [(a + b) / 2.0 for a, b in zip(left_prob, right_prob, strict=True)]

    def divergence(values: Sequence[float]) -> float:
        return sum(
            value * math.log2(value / center)
            for value, center in zip(values, midpoint, strict=True)
            if value > 0.0 and center > 0.0
        )

    return math.sqrt((divergence(left_prob) + divergence(right_prob)) / 2.0)


def build_output_record(
    identifiers: Mapping[str, object],
    worker_record: Mapping[str, object],
) -> dict[str, object]:
    """Validate one worker response and add bilingual preservation measures."""
    pair_id = str(identifiers["pair_id"])
    if str(worker_record.get("pair_id", "")) != pair_id:
        raise ValueError("Emotion worker returned a mismatched pair identifier")
    forbidden = set(TEXT_COLUMNS).intersection(worker_record)
    if forbidden:
        raise ValueError("Emotion worker returned response text")

    expected = {*PT_FEATURES, *EN_FEATURES, "pair_id"}
    missing = expected.difference(worker_record)
    extra = set(worker_record).difference(expected)
    if missing or extra:
        raise ValueError("Emotion worker returned an invalid feature schema")

    record: dict[str, object] = {name: identifiers[name] for name in IDENTIFIER_COLUMNS}
    for name in (*PT_FEATURES, *EN_FEATURES):
        value = _as_float(worker_record[name])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("Emotion worker returned a non-finite or negative feature")
        record[name] = value

    pt_emotion = _distribution(record, "pt", "emotion", EMOTIONS)
    en_emotion = _distribution(record, "en", "emotion", EMOTIONS)
    pt_valence = _distribution(record, "pt", "valence", VALENCES)
    en_valence = _distribution(record, "en", "valence", VALENCES)

    for index, emotion in enumerate(EMOTIONS):
        record[f"emotion_{emotion}_share_delta_en_minus_pt"] = en_emotion[index] - pt_emotion[index]
    for index, valence in enumerate(VALENCES):
        record[f"valence_{valence}_share_delta_en_minus_pt"] = en_valence[index] - pt_valence[index]
    record["emotion_mean_absolute_share_delta"] = sum(
        abs(left - right) for left, right in zip(pt_emotion, en_emotion, strict=True)
    ) / len(EMOTIONS)
    record["valence_mean_absolute_share_delta"] = sum(
        abs(left - right) for left, right in zip(pt_valence, en_valence, strict=True)
    ) / len(VALENCES)
    record["emotion_jensen_shannon_distance"] = _jensen_shannon_distance(pt_emotion, en_emotion)
    record["valence_jensen_shannon_distance"] = _jensen_shannon_distance(pt_valence, en_valence)
    record["semantic_density_absolute_delta"] = abs(
        _as_float(record["en_semantic_density"]) - _as_float(record["pt_semantic_density"])
    )
    return record
