from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from omegaconf import DictConfig, OmegaConf

from translation_audit.language import Detection
from translation_audit.runtime.run_workspace import RunWorkspace
from translation_audit.stages.validate_languages import ValidateLanguagesStage


class FakeDetector:
    def detect(self, text: str, expected: str) -> Detection:
        if text == "123":
            return Detection(None, 0.0)
        return Detection(expected, 0.9)


@pytest.mark.asyncio
async def test_language_stage_writes_text_free_flag_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "pairs.parquet"
    pl.DataFrame(
        {
            "pair_id": ["a", "b"],
            "source_pt": ["bom dia", "123"],
            "translation_en": ["good morning", "123"],
        }
    ).write_parquet(input_path)
    run_root = tmp_path / "runs" / "run-1"
    cfg = OmegaConf.create(
        {
            "project": {"artifact_root": str(tmp_path)},
            "stage": {
                "name": "validate_languages",
                "input_path": str(input_path),
                "input_sha256": "",
                "minimum_relative_distance": 0.05,
            },
            "runtime": {"capture_environment": False, "checkpoint_every_rows": 1},
            "logging": {"json_filename": "events.jsonl", "level": "INFO", "include_console": False},
        }
    )
    monkeypatch.setattr(
        "translation_audit.runtime.run_workspace.HydraConfig.get",
        lambda: SimpleNamespace(
            runtime=SimpleNamespace(output_dir=str(run_root)),
            overrides=SimpleNamespace(task=[]),
        ),
    )
    with RunWorkspace(cfg) as workspace:
        result = await ValidateLanguagesStage(DictConfig(cfg), FakeDetector()).run(workspace)
        shards = sorted((workspace.paths.artifacts / "language_verification" / "shards").glob("*.parquet"))
        assert result.metrics["rows"] == 2
        assert result.metrics["pt_ambiguous"] == 1
        assert len(shards) == 2
        for shard in shards:
            assert "source_pt" not in pl.read_parquet_schema(shard)
            assert "translation_en" not in pl.read_parquet_schema(shard)
