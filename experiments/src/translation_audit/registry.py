"""Stage registry and factory."""

from collections.abc import Callable
from typing import TypeVar

from omegaconf import DictConfig

from translation_audit.types import ExperimentStage

StageType = TypeVar("StageType", bound=type[ExperimentStage])
STAGE_REGISTRY: dict[str, type[ExperimentStage]] = {}


def register_stage(name: str) -> Callable[[StageType], StageType]:
    """Register a unique experiment stage class."""

    def decorator(stage_class: StageType) -> StageType:
        if name in STAGE_REGISTRY:
            raise ValueError(f"Stage '{name}' is already registered")
        STAGE_REGISTRY[name] = stage_class
        return stage_class

    return decorator


def create_stage(name: str, cfg: DictConfig) -> ExperimentStage:
    """Construct a config-driven stage by registry name."""
    stage_class = STAGE_REGISTRY.get(name)
    if stage_class is None:
        available = ", ".join(sorted(STAGE_REGISTRY)) or "none"
        raise ValueError(f"Unknown stage '{name}'. Available stages: {available}")
    return stage_class(cfg)
