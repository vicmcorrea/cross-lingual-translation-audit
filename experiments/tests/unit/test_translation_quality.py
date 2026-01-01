import json
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import Self

import polars as pl
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from translation_audit.config import EXPERIMENT_ROOT
from translation_audit.resolvers import register_resolvers
from translation_audit.runtime.checkpoints import CheckpointManager
from translation_audit.runtime.files import sha256_file
from translation_audit.runtime.run_workspace import RunWorkspace
from translation_audit.stages.translation_quality import (
    CometWorkerClient,
    EstimateTranslationQualityStage,
    SyntheticQualityEstimator,
)
from translation_audit.types import RunPaths


def test_comet_hydra_contract_is_active_and_revision_pinned() -> None:
    register_resolvers()
    with initialize_config_dir(version_base="1.3", config_dir=str(EXPERIMENT_ROOT / "conf")):
        config = compose(config_name="config", overrides=["stage=estimate_translation_quality"])
    assert config.stage.implementation == "estimate_translation_quality"
    assert config.qe.reference_free is True
    assert list(config.qe.inputs) == ["source_pt", "translation_en"]
    assert config.qe.revision == "33858b2239a139d497d9c74952c88b89a8c06213"


def test_comet_client_preserves_virtual_environment_python_symlink(tmp_path: Path) -> None:
    real_python = tmp_path / "python3.11"
    real_python.write_text("placeholder", encoding="utf-8")
    virtualenv_python = tmp_path / "runtime" / ".venv" / "bin" / "python"
    virtualenv_python.parent.mkdir(parents=True)
    virtualenv_python.symlink_to(real_python)
    worker_script = tmp_path / "worker.py"
    worker_script.write_text("placeholder", encoding="utf-8")
    config = _configuration(tmp_path / "data")
    config.qe.worker.python_executable = str(virtualenv_python)
    config.qe.worker.script = str(worker_script)

    client = CometWorkerClient(config)

    assert client.python_executable == virtualenv_python.absolute()
    assert client.python_executable.is_symlink()


class _FailAfterOneEstimator:
    def __init__(self) -> None:
        self.calls = 0
        self.synthetic = SyntheticQualityEstimator()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    def score(self, sources: Sequence[str], translations: Sequence[str]) -> list[float]:
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError("synthetic interruption")
        return self.synthetic.score(sources, translations)


class _CountingEstimator(SyntheticQualityEstimator):
    def __init__(self) -> None:
        self.calls = 0

    def score(self, sources: Sequence[str], translations: Sequence[str]) -> list[float]:
        self.calls += 1
        return super().score(sources, translations)


class _FailIfCalledEstimator(SyntheticQualityEstimator):
    def score(self, sources: Sequence[str], translations: Sequence[str]) -> list[float]:
        del sources, translations
        raise AssertionError("A verified COMET shard must not be recomputed")


def _configuration(
    data_root: Path,
    *,
    project_root: Path | None = None,
    resume_from_run: Path | None = None,
) -> DictConfig:
    artifact_root = (project_root or data_root.parent) / "artifacts"
    return OmegaConf.create(
        {
            "project": {"artifact_root": str(artifact_root)},
            "stage": {
                "name": "estimate_translation_quality",
                "implementation": "estimate_translation_quality",
                "input_filename": "paired_responses.parquet",
                "output_directory": "translation_quality",
                "output_filename": "cometkiwi_scores.parquet",
                "resume_from_run": None if resume_from_run is None else str(resume_from_run),
            },
            "data": {
                "curated_dir": str(data_root / "curated"),
                "sanitized_manifest": str(data_root / "manifest.json"),
                "expected": {"paired_responses": 5},
            },
            "qe": {
                "repository": "Unbabel/wmt23-cometkiwi-da-xl",
                "revision": "33858b2239a139d497d9c74952c88b89a8c06213",
                "frozen": True,
                "batch_size": 2,
                "precision": "float32",
                "inputs": ["source_pt", "translation_en"],
                "reference_free": True,
                "worker": {"python_executable": "unused", "script": "unused"},
            },
            "runtime": {"checkpoint_every_rows": 2, "device": "cpu"},
        }
    )


def _workspace(root: Path) -> RunWorkspace:
    paths = RunPaths(
        root=root,
        artifacts=root / "artifacts",
        checkpoints=root / "checkpoints",
        figures=root / "figures",
        logs=root / "logs",
        manifests=root / "manifests",
        metrics=root / "metrics",
        tables=root / "tables",
    )
    paths.artifacts.mkdir(parents=True)
    workspace = object.__new__(RunWorkspace)
    workspace.paths = paths
    workspace.checkpoints = CheckpointManager(paths.checkpoints)
    return workspace


def _write_input(data_root: Path) -> None:
    input_path = data_root / "curated" / "paired_responses.parquet"
    input_path.parent.mkdir(parents=True)
    pl.DataFrame(
        {
            "pair_id": [f"pair-{index}" for index in range(5)],
            "source_pt": [f"MARCADOR_PRIVADO_PT_{index}" for index in range(5)],
            "translation_en": [f"PRIVATE_MARKER_EN_{index}" for index in range(5)],
        }
    ).write_parquet(input_path)
    (data_root / "manifest.json").write_text(
        json.dumps({"output_sha256": {input_path.name: sha256_file(input_path)}}),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_translation_quality_is_sharded_text_free_and_resumable(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_input(data_root)
    config = _configuration(data_root)
    workspace = _workspace(tmp_path / "run")

    interrupted = _FailAfterOneEstimator()
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        await EstimateTranslationQualityStage(config, interrupted).run(workspace)
    assert interrupted.calls == 2
    shard_root = workspace.paths.artifacts / "translation_quality" / "shards"
    assert [path.name for path in shard_root.glob("*.parquet")] == ["part-000000.parquet"]

    resumed = _CountingEstimator()
    result = await EstimateTranslationQualityStage(config, resumed).run(workspace)
    assert resumed.calls == 2
    assert result.metrics["rows_scored"] == 5
    assert result.metrics["shards"] == 3

    output = pl.read_parquet(
        workspace.paths.artifacts / "translation_quality" / "cometkiwi_scores.parquet"
    )
    assert output.columns == ["pair_id", "cometkiwi_score"]
    assert output.height == 5
    assert output["pair_id"].to_list() == [f"pair-{index}" for index in range(5)]
    assert output["cometkiwi_score"].is_finite().all()

    checkpoint_documents = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(workspace.paths.checkpoints.glob("*.json"))
    )
    assert "qe_shard_reused" in checkpoint_documents
    assert "MARCADOR_PRIVADO" not in checkpoint_documents
    assert "PRIVATE_MARKER" not in checkpoint_documents
    manifest_text = (
        workspace.paths.artifacts / "translation_quality" / "manifest.json"
    ).read_text(encoding="utf-8")
    assert "MARCADOR_PRIVADO" not in manifest_text
    assert "PRIVATE_MARKER" not in manifest_text


@pytest.mark.asyncio
async def test_translation_quality_rejects_non_reference_free_or_mutable_config(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    _write_input(data_root)
    workspace = _workspace(tmp_path / "run")
    config = _configuration(data_root)
    config.qe.reference_free = False
    with pytest.raises(ValueError, match="reference-free"):
        await EstimateTranslationQualityStage(config, SyntheticQualityEstimator()).run(workspace)

    config.qe.reference_free = True
    config.qe.revision = "main"
    with pytest.raises(ValueError, match="full 40-character commit"):
        await EstimateTranslationQualityStage(config, SyntheticQualityEstimator()).run(workspace)


@pytest.mark.asyncio
async def test_translation_quality_resumes_verified_shards_into_a_new_run(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_input(data_root)
    old_root = tmp_path / "artifacts" / "runs" / "old-run"
    old_config = _configuration(data_root, project_root=tmp_path)
    old_result = await EstimateTranslationQualityStage(
        old_config, SyntheticQualityEstimator()
    ).run(_workspace(old_root))
    assert old_result.metrics["shards_reused"] == 0

    new_root = tmp_path / "artifacts" / "runs" / "new-run"
    new_config = _configuration(data_root, project_root=tmp_path, resume_from_run=old_root)
    new_result = await EstimateTranslationQualityStage(
        new_config, _FailIfCalledEstimator()
    ).run(_workspace(new_root))
    assert new_result.metrics["shards_reused"] == 3
    old_output = old_root / "artifacts" / "translation_quality" / "cometkiwi_scores.parquet"
    new_output = new_root / "artifacts" / "translation_quality" / "cometkiwi_scores.parquet"
    assert sha256_file(old_output) == sha256_file(new_output)
