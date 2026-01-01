"""Text-in-memory JSONL worker for bilingual EmoAtlas inference."""

import argparse
import contextlib
import importlib
import json
import math
import os
import sys
from collections.abc import Callable, Mapping, Sequence, Set
from pathlib import Path
from typing import Any, Protocol, TextIO, cast

EMOTIONS = (
    "anger",
    "trust",
    "surprise",
    "disgust",
    "joy",
    "sadness",
    "fear",
    "anticipation",
)
VALENCES = ("positive", "negative", "ambivalent", "neutral")


class SemanticNetwork(Protocol):
    """Minimal forma mentis network surface consumed by this worker."""

    @property
    def edges(self) -> Sequence[tuple[str, str]]: ...

    @property
    def vertices(self) -> Sequence[str]: ...


class Analyzer(Protocol):
    """Minimal EmoScores API needed by the worker."""

    def formamentis_network(
        self,
        text: str,
        *,
        max_distance: int,
        semantic_enrichment: Sequence[str],
        multiplex: bool,
    ) -> SemanticNetwork: ...

    def emotions(
        self,
        obj: SemanticNetwork,
        *,
        normalization_strategy: str,
        return_words: bool,
    ) -> Mapping[str, Mapping[str, object]]: ...


class AnalyzerFactory(Protocol):
    """Constructor surface exposed by EmoScores."""

    def __call__(self, *, language: str, spacy_model: str) -> Analyzer: ...


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _as_int(value: object, default: int) -> int:
    if value is None:
        return default
    if not isinstance(value, int | float | str):
        raise ValueError("EmoAtlas returned a non-numeric count")
    return int(value)


def compute_language_features(
    *,
    text: str,
    analyzer: Analyzer,
    valence_sets: tuple[Set[str], Set[str], Set[str]],
    prefix: str,
    max_distance: int,
) -> dict[str, float]:
    """Compute one language's text-free emotion, valence, and network features."""
    network = analyzer.formamentis_network(
        text,
        max_distance=max_distance,
        semantic_enrichment=(),
        multiplex=False,
    )
    nodes = {str(node).lower() for node in network.vertices}
    edges = list(network.edges)
    node_count = len(nodes)
    edge_count = len(edges)
    density = _safe_ratio(2 * edge_count, node_count * (node_count - 1)) if node_count > 1 else 0.0

    emotion_words = analyzer.emotions(
        network,
        normalization_strategy="none",
        return_words=True,
    )
    counts: dict[str, int] = {}
    matched_words: set[str] = set()
    for emotion in EMOTIONS:
        details = emotion_words.get(emotion, {})
        words_value = details.get("words", [])
        words = {str(word).lower() for word in cast(Sequence[object], words_value)}
        count = _as_int(details.get("count"), len(words))
        if count < 0:
            raise ValueError("EmoAtlas returned a negative emotion count")
        counts[emotion] = count
        matched_words.update(words)
    emotion_total = sum(counts.values())

    positive, negative, ambivalent = valence_sets
    valence_counts = dict.fromkeys(VALENCES, 0)
    for node in nodes:
        if node in ambivalent:
            valence_counts["ambivalent"] += 1
        elif node in positive:
            valence_counts["positive"] += 1
        elif node in negative:
            valence_counts["negative"] += 1
        else:
            valence_counts["neutral"] += 1
    labeled_valence_count = node_count - valence_counts["neutral"]

    features: dict[str, float] = {
        f"{prefix}_semantic_node_count": float(node_count),
        f"{prefix}_semantic_edge_count": float(edge_count),
        f"{prefix}_semantic_density": density,
        f"{prefix}_emotion_lexicon_coverage": _safe_ratio(len(matched_words.intersection(nodes)), node_count),
        f"{prefix}_valence_lexicon_coverage": _safe_ratio(labeled_valence_count, node_count),
    }
    for emotion in EMOTIONS:
        features[f"{prefix}_emotion_{emotion}_type_count"] = float(counts[emotion])
        features[f"{prefix}_emotion_{emotion}_share"] = _safe_ratio(counts[emotion], emotion_total)
    for valence in VALENCES:
        features[f"{prefix}_valence_{valence}_node_count"] = float(valence_counts[valence])
        features[f"{prefix}_valence_{valence}_share"] = _safe_ratio(valence_counts[valence], node_count)
    if any(not math.isfinite(value) or value < 0.0 for value in features.values()):
        raise ValueError("EmoAtlas produced an invalid numeric feature")
    return features


def analyze_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    portuguese_analyzer: Analyzer,
    english_analyzer: Analyzer,
    portuguese_valences: tuple[Set[str], Set[str], Set[str]],
    english_valences: tuple[Set[str], Set[str], Set[str]],
    max_distance: int,
) -> list[dict[str, object]]:
    """Analyze one shard while returning no source or translated text."""
    records: list[dict[str, object]] = []
    for row in rows:
        record: dict[str, object] = {"pair_id": str(row["pair_id"])}
        record.update(
            compute_language_features(
                text=str(row["source_pt"]),
                analyzer=portuguese_analyzer,
                valence_sets=portuguese_valences,
                prefix="pt",
                max_distance=max_distance,
            )
        )
        record.update(
            compute_language_features(
                text=str(row["translation_en"]),
                analyzer=english_analyzer,
                valence_sets=english_valences,
                prefix="en",
                max_distance=max_distance,
            )
        )
        records.append(record)
    return records


def _load_analyzers(
    portuguese_model: str,
    english_model: str,
) -> tuple[Analyzer, Analyzer, tuple[Set[str], Set[str], Set[str]], tuple[Set[str], Set[str], Set[str]]]:
    """Import the pinned runtime lazily so unit tests require no model download."""
    emoatlas_module = importlib.import_module("emoatlas")
    resources_module = importlib.import_module("emoatlas.resources")
    analyzer_factory = cast(AnalyzerFactory, emoatlas_module.EmoScores)
    valences_loader = cast(
        Callable[[str], tuple[Set[str], Set[str], Set[str]]],
        resources_module._valences,
    )
    portuguese_analyzer = analyzer_factory(language="portuguese", spacy_model=portuguese_model)
    english_analyzer = analyzer_factory(language="english", spacy_model=english_model)
    portuguese_valences = valences_loader("portuguese")
    english_valences = valences_loader("english")
    return portuguese_analyzer, english_analyzer, portuguese_valences, english_valences


def serve(
    input_stream: TextIO,
    output_stream: TextIO,
    *,
    portuguese_model: str,
    english_model: str,
    max_distance: int,
) -> int:
    """Serve a narrow JSONL protocol and redact all processing failures."""
    with Path(os.devnull).open("w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            analyzers = _load_analyzers(portuguese_model, english_model)
        portuguese_analyzer, english_analyzer, portuguese_valences, english_valences = analyzers
        for line in input_stream:
            try:
                decoded: object = json.loads(line)
                if not isinstance(decoded, dict):
                    raise ValueError("invalid control request")
                request = cast(dict[str, object], decoded)
                if request.get("command") == "shutdown":
                    return 0
                rows_value = request.get("rows")
                if request.get("command") != "analyze" or not isinstance(rows_value, list):
                    raise ValueError("invalid control command")
                rows_untyped = cast(list[object], rows_value)
                if not all(isinstance(row, dict) for row in rows_untyped):
                    raise ValueError("invalid row schema")
                rows = [cast(dict[str, object], row) for row in rows_untyped]
                with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                    records = analyze_rows(
                        rows,
                        portuguese_analyzer=portuguese_analyzer,
                        english_analyzer=english_analyzer,
                        portuguese_valences=portuguese_valences,
                        english_valences=english_valences,
                        max_distance=max_distance,
                    )
                response: dict[str, Any] = {"status": "ok", "records": records}
            except BaseException as error:
                response = {"status": "error", "error_type": type(error).__name__}
            output_stream.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n")
            output_stream.flush()
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the isolated bilingual EmoAtlas worker")
    parser.add_argument("--portuguese-model", required=True)
    parser.add_argument("--english-model", required=True)
    parser.add_argument("--max-distance", type=int, default=3)
    return parser


def main() -> int:
    args = _parser().parse_args()
    return serve(
        sys.stdin,
        sys.stdout,
        portuguese_model=str(args.portuguese_model),
        english_model=str(args.english_model),
        max_distance=int(args.max_distance),
    )


if __name__ == "__main__":
    raise SystemExit(main())
