"""Hydra-driven experiment stage runner."""

import importlib
from pathlib import Path

import structlog
from omegaconf import DictConfig

from translation_audit.registry import create_stage
from translation_audit.runtime import RunWorkspace
from translation_audit.runtime.dependencies import verified_upstream_runs
from translation_audit.runtime.files import write_json_exclusive
from translation_audit.runtime.manifest import summarize_exception


def validate_runtime_contract(cfg: DictConfig) -> None:
    """Reject executions that cannot satisfy the common scientific run contract."""
    checkpoint_every_rows = int(cfg.runtime.get("checkpoint_every_rows", 0))
    if checkpoint_every_rows <= 0:
        raise ValueError("runtime.checkpoint_every_rows must be a positive integer")
    if not bool(cfg.runtime.get("resume", False)):
        raise ValueError("runtime.resume must be enabled for every scientific run")
    if bool(cfg.runtime.get("store_response_text_in_logs", True)):
        raise ValueError("Response text must never be stored in logs")

    log_filename = str(cfg.logging.get("json_filename", ""))
    if Path(log_filename).name != log_filename or not log_filename.endswith(".jsonl"):
        raise ValueError("logging.json_filename must be a run-local JSONL filename")


def validate_stage_dependencies(cfg: DictConfig) -> dict[str, str]:
    """Require successful upstream run manifests before an executable stage starts."""
    dependencies = [str(value) for value in cfg.stage.get("depends_on", [])]
    if not dependencies:
        return {}
    _, fingerprints = verified_upstream_runs(cfg, tuple(dependencies))
    return fingerprints


async def run_pipeline(cfg: DictConfig) -> None:
    """Run one stage inside an immutable, checkpointed workspace."""
    validate_runtime_contract(cfg)
    importlib.import_module("translation_audit.stages")
    with RunWorkspace(cfg) as workspace:
        if workspace.checkpoints is None:
            raise RuntimeError("Checkpoint manager was not initialized")

        logger = structlog.get_logger().bind(stage=str(cfg.stage.name))
        if not bool(cfg.runtime.execute):
            workspace.checkpoints.write(
                "execution_disabled",
                {"reason": "runtime.execute is false", "stage": str(cfg.stage.name)},
            )
            logger.warning("execution_disabled", reason="runtime.execute is false")
            return

        dependency_fingerprints = validate_stage_dependencies(cfg)
        if dependency_fingerprints:
            workspace.checkpoints.write(
                "dependencies_validated",
                {"manifest_sha256": dependency_fingerprints},
            )
        stage = create_stage(str(cfg.stage.implementation), cfg)
        workspace.checkpoints.write("stage_started", {"stage": str(cfg.stage.name)})
        logger.info("stage_started")
        try:
            result = await stage.run(workspace)
        except BaseException as error:
            workspace.checkpoints.write(
                "stage_failed",
                {
                    "stage": str(cfg.stage.name),
                    **summarize_exception(error),
                },
            )
            logger.error("stage_failed", error_type=type(error).__name__)
            raise

        workspace.checkpoints.write(
            "stage_completed",
            {
                "stage": str(cfg.stage.name),
                "metrics": result.metrics,
                "artifacts": list(result.artifacts),
                "details": result.details,
            },
        )
        stage_result = {
            "stage": str(cfg.stage.name),
            "metrics": result.metrics,
            "artifacts": list(result.artifacts),
            "details": result.details,
        }
        write_json_exclusive(workspace.paths.manifests / "100_stage_result.json", stage_result)
        write_json_exclusive(
            workspace.paths.metrics / "stage_metrics.json",
            {"stage": str(cfg.stage.name), "metrics": result.metrics},
        )
        logger.info("stage_completed", metric_count=len(result.metrics), artifact_count=len(result.artifacts))
