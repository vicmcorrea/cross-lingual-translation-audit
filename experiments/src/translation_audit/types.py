"""Shared domain types for experiment stages."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from omegaconf import DictConfig

if TYPE_CHECKING:
    from translation_audit.runtime import RunWorkspace


@dataclass(frozen=True, slots=True)
class RunPaths:
    """Run-owned directories that no other run may modify."""

    root: Path
    artifacts: Path
    checkpoints: Path
    figures: Path
    logs: Path
    manifests: Path
    metrics: Path
    tables: Path


@dataclass(frozen=True, slots=True)
class StageResult:
    """Serializable summary returned by a completed stage."""

    metrics: dict[str, float | int] = field(default_factory=lambda: dict[str, float | int]())
    artifacts: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=lambda: dict[str, Any]())


class StageNotImplementedError(RuntimeError):
    """Raised when execution is enabled for a scaffold-only stage."""


class ExperimentStage(Protocol):
    """Config-driven asynchronous stage interface."""

    stage_name: str
    cfg: DictConfig

    def __init__(self, cfg: DictConfig) -> None: ...

    async def run(self, workspace: RunWorkspace) -> StageResult: ...
