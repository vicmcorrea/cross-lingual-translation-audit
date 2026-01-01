import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace, TracebackType
from typing import Self

import polars as pl
import pytest
from omegaconf import DictConfig, OmegaConf

from translation_audit.emotion.backend import EmotionWorkerError
from translation_audit.emotion.features import EMOTIONS, EN_FEATURES, PT_FEATURES, VALENCES
from translation_audit.runtime.run_workspace import RunWorkspace
from translation_audit.stages.compute_emotion_features import ComputeEmotionFeaturesStage


def _worker_record(pair_id: str) -> dict[str, object]:
    record: dict[str, object] = {name: 0.0 for name in (*PT_FEATURES, *EN_FEATURES)}
    record["pair_id"] = pair_id
    for prefix in ("pt", "en"):
        record[f"{prefix}_semantic_node_count"] = 4.0
        record[f"{prefix}_semantic_edge_count"] = 3.0
        record[f"{prefix}_semantic_density"] = 0.5
        record[f"{prefix}_emotion_lexicon_coverage"] = 0.5
        record[f"{prefix}_valence_lexicon_coverage"] = 0.75
        for emotion in EMOTIONS:
            record[f"{prefix}_emotion_{emotion}_type_count"] = 0.0
        for valence in VALENCES:
            record[f"{prefix}_valence_{valence}_node_count"] = 1.0
            record[f"{prefix}_valence_{valence}_share"] = 0.25
    record["pt_emotion_joy_type_count"] = 3.0
    record["pt_emotion_joy_share"] = 0.75
    record["pt_emotion_fear_type_count"] = 1.0
    record["pt_emotion_fear_share"] = 0.25
    record["en_emotion_joy_type_count"] = 2.0
    record["en_emotion_joy_share"] = 0.5
    record["en_emotion_fear_type_count"] = 2.0
    record["en_emotion_fear_share"] = 0.5
    return record


class _FakeBackend:
    def __init__(self, fail_on_call: int | None = None) -> None:
        self.fail_on_call = fail_on_call
        self.calls = 0

    def __enter__(self) -> Self:
        return self

    def analyze(self, rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise EmotionWorkerError("synthetic safe failure")
        return [_worker_record(str(row["pair_id"])) for row in rows]

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exception_type, exception, traceback


def _config(input_path: Path, *, max_retries: int = 2) -> DictConfig:
    return OmegaConf.create(
        {
            "stage": {"name": "compute_emotion_features"},
            "runtime": {
                "capture_environment": False,
                "checkpoint_every_rows": 2,
                "resume": True,
                "store_response_text_in_logs": False,
            },
            "logging": {"level": "INFO", "json_filename": "events.jsonl", "include_console": False},
            "emotion": {
                "name": "emoatlas",
                "repository": "https://example.invalid/emoatlas",
                "revision": "synthetic-pinned-revision",
                "input_path": str(input_path),
                "languages": ["portuguese", "english"],
                "features": {"max_distance": 3},
                "worker": {
                    "python_candidates": ["/unavailable/python"],
                    "module": "translation_audit_emoatlas.worker",
                    "portuguese_model": "synthetic_pt",
                    "english_model": "synthetic_en",
                    "max_shard_retries": max_retries,
                },
            },
        }
    )


def _patch_hydra(monkeypatch: pytest.MonkeyPatch, run_directory: Path) -> None:
    state = SimpleNamespace(
        runtime=SimpleNamespace(output_dir=str(run_directory)),
        overrides=SimpleNamespace(task=[]),
    )
    monkeypatch.setattr("translation_audit.runtime.run_workspace.HydraConfig.get", lambda: state)


def _write_synthetic_input(path: Path) -> list[str]:
    markers = [f"synthetic-private-marker-{index}" for index in range(5)]
    table = pl.DataFrame(
        {
            "pair_id": [f"pair-{index}" for index in range(5)],
            "participant_id": [f"participant-{index}" for index in range(5)],
            "cohort_id": ["synthetic-cohort"] * 5,
            "year": [2026] * 5,
            "question_index": [1, 1, 2, 2, 3],
            "prompt_id": ["prompt"] * 5,
            "prompt_family": ["synthetic"] * 5,
            "source_pt": markers,
            "translation_en": [f"translated-{marker}" for marker in markers],
        }
    )
    table.write_parquet(path)
    return markers


@pytest.mark.asyncio
async def test_emotion_stage_shards_resumes_and_never_persists_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "synthetic.parquet"
    markers = _write_synthetic_input(input_path)
    run_directory = tmp_path / "run"
    _patch_hydra(monkeypatch, run_directory)
    config = _config(input_path, max_retries=0)

    with RunWorkspace(config) as workspace:
        failing = _FakeBackend(fail_on_call=2)
        stage = ComputeEmotionFeaturesStage(config, backend_factory=lambda: failing)
        with pytest.raises(EmotionWorkerError):
            await stage.run(workspace)

        succeeding = _FakeBackend()
        resumed = ComputeEmotionFeaturesStage(config, backend_factory=lambda: succeeding)
        result = await resumed.run(workspace)
        assert result.metrics["processed_rows"] == 5
        assert succeeding.calls == 2

    output_path = run_directory / "artifacts" / "emotion_features" / "emotion_features.parquet"
    output = pl.read_parquet(output_path)
    assert output.height == 5
    assert "source_pt" not in output.columns
    assert "translation_en" not in output.columns
    assert "emotion_jensen_shannon_distance" in output.columns

    checkpoints = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((run_directory / "checkpoints").glob("*.json"))
    ]
    completed = [item for item in checkpoints if item["event"] == "emotion_shard_completed"]
    assert len(completed) == 3
    assert len({item["payload"]["shard_index"] for item in completed}) == 3

    persisted_control_text = "\n".join(
        path.read_text(encoding="utf-8")
        for folder in ("checkpoints", "logs", "manifests")
        for path in (run_directory / folder).glob("*.json*")
    )
    assert all(marker not in persisted_control_text for marker in markers)
    assert (run_directory / "run_seal.json").is_file()


@pytest.mark.asyncio
async def test_emotion_stage_restarts_worker_without_logging_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "synthetic.parquet"
    markers = _write_synthetic_input(input_path)
    run_directory = tmp_path / "retry-run"
    _patch_hydra(monkeypatch, run_directory)
    config = _config(input_path, max_retries=1)
    backends = iter([_FakeBackend(fail_on_call=1), _FakeBackend()])

    with RunWorkspace(config) as workspace:
        stage = ComputeEmotionFeaturesStage(config, backend_factory=lambda: next(backends))
        result = await stage.run(workspace)
        assert result.metrics["processed_rows"] == 5

    control_text = "\n".join(
        path.read_text(encoding="utf-8")
        for folder in ("checkpoints", "logs", "manifests")
        for path in (run_directory / folder).glob("*.json*")
    )
    assert "emotion_worker_restarted" in control_text
    assert all(marker not in control_text for marker in markers)
