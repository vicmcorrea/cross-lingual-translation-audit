"""Immutable Hydra run workspace lifecycle."""

import os
from pathlib import Path
from types import TracebackType
from typing import Any, cast

import structlog
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from translation_audit.runtime.checkpoints import CheckpointManager
from translation_audit.runtime.files import sha256_file, write_json_exclusive, write_text_exclusive
from translation_audit.runtime.logging import LoggingHandle, configure_structured_logging
from translation_audit.runtime.manifest import collect_environment, summarize_exception, utc_now
from translation_audit.types import RunPaths

_SECRET_KEY_PARTS = ("api_key", "password", "secret", "token")


def assert_config_is_secret_free(value: object, path: str = "config") -> None:
    """Reject secret-shaped configuration before Hydra persists it."""
    if isinstance(value, dict):
        for key, item in cast(dict[object, object], value).items():
            key_text = str(key).lower()
            if any(part in key_text for part in _SECRET_KEY_PARTS):
                raise ValueError(f"Secret-shaped key is forbidden in Hydra configuration at {path}.{key}")
            assert_config_is_secret_free(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(cast(list[object], value)):
            assert_config_is_secret_free(item, f"{path}[{index}]")


class RunWorkspace:
    """Own all mutable state for one unique Hydra run."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        run_root = Path(HydraConfig.get().runtime.output_dir).resolve()
        self.run_id = run_root.name
        self.paths = RunPaths(
            root=run_root,
            artifacts=run_root / "artifacts",
            checkpoints=run_root / "checkpoints",
            figures=run_root / "figures",
            logs=run_root / "logs",
            manifests=run_root / "manifests",
            metrics=run_root / "metrics",
            tables=run_root / "tables",
        )
        self.checkpoints: CheckpointManager | None = None
        self._lock_descriptor: int | None = None
        self._logging_handle: LoggingHandle | None = None

    def __enter__(self) -> RunWorkspace:
        """Lock and initialize an isolated run directory."""
        self.paths.root.mkdir(parents=True, exist_ok=True)
        lock_path = self.paths.root / ".run-lock"
        if lock_path.exists():
            raise FileExistsError(f"Run directory has already been initialized at {self.paths.root}")
        self._lock_descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.write(self._lock_descriptor, f"pid={os.getpid()}\n".encode())

        try:
            self._initialize_directories_and_metadata()
        except BaseException:
            self._release_resources()
            raise
        return self

    def _initialize_directories_and_metadata(self) -> None:
        """Create run-owned directories and initial provenance records."""

        for directory in (
            self.paths.figures,
            self.paths.logs,
            self.paths.manifests,
            self.paths.metrics,
            self.paths.tables,
        ):
            directory.mkdir(parents=True, exist_ok=False)
        self.paths.artifacts.mkdir(parents=True, exist_ok=True)
        config_directory = self.paths.artifacts / "config"
        config_directory.mkdir(parents=True, exist_ok=False)

        logging_cfg = self.cfg.logging
        self._logging_handle = configure_structured_logging(
            log_path=self.paths.logs / str(logging_cfg.json_filename),
            level=str(logging_cfg.level),
            include_console=bool(logging_cfg.include_console),
            run_id=self.run_id,
        )

        resolved_container = OmegaConf.to_container(self.cfg, resolve=True)
        assert_config_is_secret_free(resolved_container)
        resolved_config = OmegaConf.to_yaml(self.cfg, resolve=True)
        write_text_exclusive(config_directory / "resolved.yaml", resolved_config)
        overrides = HydraConfig.get().overrides.task
        write_text_exclusive(config_directory / "overrides.txt", "\n".join(overrides) + "\n")

        start_manifest: dict[str, Any] = {
            "run_id": self.run_id,
            "status": "started",
            "started_at": utc_now(),
            "stage": str(self.cfg.stage.name),
            "run_directory": str(self.paths.root),
            "overrides": list(overrides),
        }
        write_json_exclusive(self.paths.manifests / "000_start.json", start_manifest)
        if bool(self.cfg.runtime.capture_environment):
            write_json_exclusive(self.paths.manifests / "001_environment.json", collect_environment())

        self.checkpoints = CheckpointManager(self.paths.checkpoints)
        self.checkpoints.write("run_started", {"run_id": self.run_id, "stage": str(self.cfg.stage.name)})
        structlog.get_logger().info("run_initialized", stage=str(self.cfg.stage.name))

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Record final state, release resources, and propagate failures."""
        try:
            self._record_final_state(exception_type, exception)
        finally:
            self._release_resources()
        if exception_type is None:
            self._seal_completed_run()

    def _record_final_state(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
    ) -> None:
        """Append terminal checkpoint, manifest, and structured log event."""
        status = "completed" if exception_type is None else "failed"
        if self.checkpoints is not None:
            payload: dict[str, Any] = {"run_id": self.run_id, "status": status}
            if exception is not None:
                payload.update(summarize_exception(exception))
            self.checkpoints.write(f"run_{status}", payload)

        end_manifest: dict[str, Any] = {
            "run_id": self.run_id,
            "status": status,
            "finished_at": utc_now(),
            "stage": str(self.cfg.stage.name),
        }
        if exception is not None:
            end_manifest.update(summarize_exception(exception))
        write_json_exclusive(self.paths.manifests / "999_end.json", end_manifest)

        logger = structlog.get_logger()
        if exception is None:
            logger.info("run_finished", status=status)
            return
        logger.error("run_finished", status=status, error_type=type(exception).__name__)

    def _release_resources(self) -> None:
        """Close run-owned streams and file descriptors unconditionally."""
        if self._logging_handle is not None:
            self._logging_handle.close()
            self._logging_handle = None
        if self._lock_descriptor is not None:
            os.close(self._lock_descriptor)
            self._lock_descriptor = None

    def _seal_completed_run(self) -> None:
        """Checksum the complete run and make its files and directories read-only."""
        seal_path = self.paths.root / "run_seal.json"
        files = {
            str(path.relative_to(self.paths.root)): sha256_file(path)
            for path in sorted(self.paths.root.rglob("*"))
            if path.is_file() and path != seal_path
        }
        write_json_exclusive(
            seal_path,
            {
                "run_id": self.run_id,
                "sealed_at": utc_now(),
                "files_sha256": files,
            },
        )
        for path in self.paths.root.rglob("*"):
            if path.is_file():
                path.chmod(0o440)
        for path in sorted(
            (item for item in self.paths.root.rglob("*") if item.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            path.chmod(0o550)
        self.paths.root.chmod(0o550)
