import io
import json
from collections.abc import Mapping, Sequence
from typing import NamedTuple

import pytest

from translation_audit_emoatlas import worker


class _Network(NamedTuple):
    edges: Sequence[tuple[str, str]]
    vertices: Sequence[str]


class _FakeAnalyzer:
    def formamentis_network(
        self,
        text: str,
        *,
        max_distance: int,
        semantic_enrichment: Sequence[str],
        multiplex: bool,
    ) -> worker.SemanticNetwork:
        del text, max_distance, semantic_enrichment, multiplex
        return _Network(
            edges=[("hope", "support"), ("support", "uncertainty")],
            vertices=["hope", "support", "uncertainty"],
        )

    def emotions(
        self,
        obj: worker.SemanticNetwork,
        *,
        normalization_strategy: str,
        return_words: bool,
    ) -> Mapping[str, Mapping[str, object]]:
        del obj, normalization_strategy, return_words
        empty: Mapping[str, object] = {"count": 0, "words": []}
        result: dict[str, Mapping[str, object]] = {emotion: empty for emotion in worker.EMOTIONS}
        result["joy"] = {"count": 2, "words": ["hope", "support"]}
        result["fear"] = {"count": 1, "words": ["uncertainty"]}
        return result


def _runtime() -> tuple[
    worker.Analyzer,
    worker.Analyzer,
    tuple[set[str], set[str], set[str]],
    tuple[set[str], set[str], set[str]],
]:
    analyzer = _FakeAnalyzer()
    valences: tuple[set[str], set[str], set[str]] = ({"hope", "support"}, {"uncertainty"}, set())
    return analyzer, analyzer, valences, valences


def _fake_loader(
    portuguese_model: str,
    english_model: str,
) -> tuple[
    worker.Analyzer,
    worker.Analyzer,
    tuple[set[str], set[str], set[str]],
    tuple[set[str], set[str], set[str]],
]:
    del portuguese_model, english_model
    return _runtime()


def test_compute_features_are_comparable_and_text_free() -> None:
    analyzer = _FakeAnalyzer()
    features = worker.compute_language_features(
        text="synthetic private marker",
        analyzer=analyzer,
        valence_sets=({"hope", "support"}, {"uncertainty"}, set()),
        prefix="pt",
        max_distance=3,
    )
    assert features["pt_semantic_node_count"] == 3.0
    assert features["pt_emotion_joy_share"] == 2 / 3
    assert features["pt_valence_positive_share"] == 2 / 3
    assert "synthetic private marker" not in json.dumps(features)


def test_jsonl_protocol_returns_no_text_and_redacts_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker, "_load_analyzers", _fake_loader)
    marker = "synthetic-private-marker"
    request = {
        "command": "analyze",
        "rows": [{"pair_id": "pair-1", "source_pt": marker, "translation_en": marker}],
    }
    input_stream = io.StringIO(json.dumps(request) + "\n" + '{"command":"shutdown"}\n')
    output_stream = io.StringIO()
    assert (
        worker.serve(
            input_stream,
            output_stream,
            portuguese_model="synthetic_pt",
            english_model="synthetic_en",
            max_distance=3,
        )
        == 0
    )
    response_text = output_stream.getvalue()
    response = json.loads(response_text)
    assert response["status"] == "ok"
    assert response["records"][0]["pair_id"] == "pair-1"
    assert marker not in response_text

    malformed_output = io.StringIO()
    worker.serve(
        io.StringIO("not-json\n" + '{"command":"shutdown"}\n'),
        malformed_output,
        portuguese_model="synthetic_pt",
        english_model="synthetic_en",
        max_distance=3,
    )
    failure = json.loads(malformed_output.getvalue())
    assert failure == {"status": "error", "error_type": "JSONDecodeError"}
