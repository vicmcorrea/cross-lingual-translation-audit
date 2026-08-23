import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import DictConfig, OmegaConf

from translation_audit.cli import enable_python314_argparse_compatibility
from translation_audit.pipeline import run_pipeline, validate_runtime_contract, validate_stage_dependencies
from translation_audit.registry import STAGE_REGISTRY, create_stage, register_stage
from translation_audit.runtime.checkpoints import CheckpointManager
from translation_audit.runtime.files import sha256_file, write_json_exclusive, write_text_exclusive
from translation_audit.runtime.manifest import collect_environment, summarize_exception
from translation_audit.runtime.run_workspace import RunWorkspace
from translation_audit.stages.contracts import ContractOnlyStage
from translation_audit.types import StageNotImplementedError, StageResult


class _LazyHelp:
    def __repr__(self) -> str:
        return "deferred help text"


@register_stage("runtime_test_success")
class SuccessfulStage:
    stage_name = "runtime_test_success"

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

    async def run(self, workspace: RunWorkspace) -> StageResult:
        artifact = write_text_exclusive(workspace.paths.artifacts / "result.txt", "complete\n")
        assert workspace.checkpoints is not None
        workspace.checkpoints.write("shard_completed", {"row_stop": 500})
        return StageResult(
            metrics={"processed_rows": 500},
            artifacts=(str(artifact.relative_to(workspace.paths.root)),),
            details={"shards": 1},
        )


def _cfg(*, execute: bool = False) -> DictConfig:
    return OmegaConf.create(
        {
            "stage": {"name": "compute_embeddings", "implementation": "contract_only"},
            "runtime": {
                "execute": execute,
                "capture_environment": False,
                "checkpoint_every_rows": 500,
                "resume": True,
                "store_response_text_in_logs": False,
            },
            "logging": {"level": "INFO", "json_filename": "events.jsonl", "include_console": False},
        }
    )


def _patch_hydra(monkeypatch: pytest.MonkeyPatch, run_directory: Path) -> None:
    hydra_state = SimpleNamespace(
        runtime=SimpleNamespace(output_dir=str(run_directory)),
        overrides=SimpleNamespace(task=[]),
    )
    monkeypatch.setattr("translation_audit.runtime.run_workspace.HydraConfig.get", lambda: hydra_state)


def test_cli_python314_help_compatibility() -> None:
    enable_python314_argparse_compatibility()
    enable_python314_argparse_compatibility()
    parser = argparse.ArgumentParser()
    parser.add_argument("--lazy", help=_LazyHelp())  # type: ignore[arg-type]
    assert "deferred help text" in parser.format_help()


def test_files_and_checkpoints_are_exclusive(tmp_path: Path) -> None:
    target = tmp_path / "artifact.txt"
    write_text_exclusive(target, "first")
    with pytest.raises(FileExistsError):
        write_text_exclusive(target, "second")
    assert sha256_file(target) == sha256_file(target)

    json_path = write_json_exclusive(tmp_path / "artifact.json", {"value": 1})
    assert json.loads(json_path.read_text(encoding="utf-8")) == {"value": 1}

    manager = CheckpointManager(tmp_path / "checkpoints")
    first = manager.write("stage_started", {"batch": 0})
    second = manager.write("stage_completed", {"batch": 1})
    assert first.name == "000001_stage_started.json"
    assert second.name == "000002_stage_completed.json"


def test_registry_contract() -> None:
    assert {"prepare_cohort", "contract_only"}.issubset(STAGE_REGISTRY)
    stage = create_stage("contract_only", _cfg())
    assert stage.stage_name == "contract_only"
    with pytest.raises(ValueError, match="Unknown stage"):
        create_stage("missing", _cfg())
    with pytest.raises(ValueError, match="already registered"):
        register_stage("contract_only")(ContractOnlyStage)


def test_runtime_contract_is_mandatory() -> None:
    config = _cfg()
    validate_runtime_contract(config)

    config.runtime.checkpoint_every_rows = 0
    with pytest.raises(ValueError, match="checkpoint_every_rows"):
        validate_runtime_contract(config)
    config.runtime.checkpoint_every_rows = 500

    config.runtime.resume = False
    with pytest.raises(ValueError, match="resume"):
        validate_runtime_contract(config)
    config.runtime.resume = True

    config.runtime.store_response_text_in_logs = True
    with pytest.raises(ValueError, match="Response text"):
        validate_runtime_contract(config)
    config.runtime.store_response_text_in_logs = False

    config.logging.json_filename = "../events.jsonl"
    with pytest.raises(ValueError, match="run-local"):
        validate_runtime_contract(config)


def test_dependency_manifests_are_required_and_fingerprinted(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    config = OmegaConf.create(
        {
            "project": {"artifact_root": str(artifact_root)},
            "stage": {
                "depends_on": ["prepare_cohort"],
                "upstream_manifests": {},
            },
        }
    )
    with pytest.raises(ValueError, match="Missing upstream"):
        validate_stage_dependencies(config)

    run_root = artifact_root / "runs" / "upstream-run"
    manifest = run_root / "manifests" / "999_end.json"
    (run_root / "artifacts" / "config").mkdir(parents=True)
    manifest.parent.mkdir(parents=True)
    (run_root / "artifacts" / "config" / "resolved.yaml").write_text("stage: prepare_cohort\n")
    (run_root / "artifacts" / "config" / "overrides.txt").write_text("\n")
    (run_root / "manifests" / "000_start.json").write_text(
        json.dumps({"stage": "prepare_cohort"}),
        encoding="utf-8",
    )
    manifest.write_text(
        json.dumps({"status": "completed", "stage": "prepare_cohort"}),
        encoding="utf-8",
    )
    sealed_files = {
        str(path.relative_to(run_root)): sha256_file(path) for path in run_root.rglob("*") if path.is_file()
    }
    seal = run_root / "run_seal.json"
    seal.write_text(json.dumps({"files_sha256": sealed_files}), encoding="utf-8")
    config.stage.upstream_manifests.prepare_cohort = str(manifest)
    assert validate_stage_dependencies(config) == {"prepare_cohort": sha256_file(seal)}

    imported_root = tmp_path / "imported-runs"
    imported_run = shutil.copytree(run_root, imported_root / run_root.name)
    imported_manifest = imported_run / "manifests" / "999_end.json"
    imported_seal = imported_run / "run_seal.json"
    config.project.imported_run_root = str(imported_root)
    config.stage.upstream_manifests.prepare_cohort = str(imported_manifest)
    assert validate_stage_dependencies(config) == {"prepare_cohort": sha256_file(imported_seal)}

    sibling_run = shutil.copytree(run_root, tmp_path / "sibling" / run_root.name)
    config.stage.upstream_manifests.prepare_cohort = str(
        sibling_run / "manifests" / "999_end.json"
    )
    with pytest.raises(ValueError, match="outside the artifact root"):
        validate_stage_dependencies(config)

    config.runtime = {"name": "runpod_secure"}
    config.stage.upstream_manifests.prepare_cohort = str(imported_manifest)
    with pytest.raises(ValueError, match="fixed tmpfs boundary"):
        validate_stage_dependencies(config)
    del config.runtime

    config.stage.upstream_manifests.prepare_cohort = str(manifest)
    manifest.write_text(json.dumps({"status": "failed", "stage": "prepare_cohort"}), encoding="utf-8")
    with pytest.raises(ValueError, match="not a successful"):
        validate_stage_dependencies(config)


def test_workspace_records_success_and_refuses_reuse(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "run-one"
    _patch_hydra(monkeypatch, run_directory)
    with RunWorkspace(_cfg()) as workspace:
        assert workspace.checkpoints is not None
        workspace.checkpoints.write("test_checkpoint", {"value": 1})
    end = json.loads((run_directory / "manifests" / "999_end.json").read_text(encoding="utf-8"))
    assert end["status"] == "completed"
    assert (run_directory / "run_seal.json").is_file()
    with pytest.raises(FileExistsError), RunWorkspace(_cfg()):
        pass


def test_workspace_records_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_directory = tmp_path / "failed-run"
    _patch_hydra(monkeypatch, run_directory)
    with pytest.raises(RuntimeError, match="deliberate"), RunWorkspace(_cfg()):
        raise RuntimeError("deliberate")
    end = json.loads((run_directory / "manifests" / "999_end.json").read_text(encoding="utf-8"))
    assert end["status"] == "failed"
    assert end["error_type"] == "RuntimeError"


@pytest.mark.asyncio
async def test_pipeline_disabled_and_contract_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    disabled = tmp_path / "disabled"
    _patch_hydra(monkeypatch, disabled)
    await run_pipeline(_cfg(execute=False))
    events = [
        json.loads(path.read_text(encoding="utf-8"))["event"]
        for path in sorted((disabled / "checkpoints").glob("*.json"))
    ]
    assert events == ["run_started", "execution_disabled", "run_completed"]

    failed = tmp_path / "failed"
    _patch_hydra(monkeypatch, failed)
    with pytest.raises(StageNotImplementedError):
        await run_pipeline(_cfg(execute=True))


@pytest.mark.asyncio
async def test_successful_pipeline_registers_results_and_seals(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "successful"
    _patch_hydra(monkeypatch, run_directory)
    config = _cfg(execute=True)
    config.stage.name = "runtime_test_success"
    config.stage.implementation = "runtime_test_success"
    await run_pipeline(config)

    result = json.loads(
        (run_directory / "manifests" / "100_stage_result.json").read_text(encoding="utf-8")
    )
    metrics = json.loads((run_directory / "metrics" / "stage_metrics.json").read_text(encoding="utf-8"))
    seal = json.loads((run_directory / "run_seal.json").read_text(encoding="utf-8"))
    assert result["metrics"] == {"processed_rows": 500}
    assert metrics["metrics"] == {"processed_rows": 500}
    assert "artifacts/result.txt" in seal["files_sha256"]
    assert "checkpoints/000003_shard_completed.json" in seal["files_sha256"]


def test_environment_and_exception_manifests_are_safe() -> None:
    environment = collect_environment()
    assert "environment" not in environment
    assert "dependency_file_sha256" in environment
    assert len(environment["scientific_source_sha256"]) == 64
    assert environment["scientific_source_file_count"] > 0
    error = summarize_exception(RuntimeError("sensitive message"))
    assert error["error_type"] == "RuntimeError"
    assert "sensitive message" not in json.dumps(error)
