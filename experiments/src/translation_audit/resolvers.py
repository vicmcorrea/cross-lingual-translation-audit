"""OmegaConf resolvers used before Hydra composes a run configuration."""

from uuid import uuid4

from omegaconf import OmegaConf

from translation_audit.config import ARTIFACT_ROOT, DATA_ROOT, EXPERIMENT_ROOT, PROJECT_ROOT


def register_resolvers() -> None:
    """Register deterministic-per-composition path and UUID resolvers."""
    resolvers = {
        "run_uuid": lambda: uuid4().hex,
        "project_root": lambda: str(PROJECT_ROOT),
        "experiment_root": lambda: str(EXPERIMENT_ROOT),
        "data_root": lambda: str(DATA_ROOT),
        "artifact_root": lambda: str(ARTIFACT_ROOT),
    }
    for name, resolver in resolvers.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, resolver, use_cache=True)
