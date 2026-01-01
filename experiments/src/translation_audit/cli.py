"""Console entry point for the Hydra experiment runner."""

import argparse
import asyncio
import sys
from collections.abc import Callable
from typing import Any, cast

import hydra
from omegaconf import DictConfig

from translation_audit.config import EXPERIMENT_ROOT, load_project_env
from translation_audit.pipeline import run_pipeline
from translation_audit.resolvers import register_resolvers

load_project_env()
register_resolvers()
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))


def enable_python314_argparse_compatibility() -> None:
    """Allow Hydra 1.3's lazy help object under Python 3.14 argparse."""
    argument_parser_type = cast(Any, argparse.ArgumentParser)
    current = cast(
        Callable[[argparse.ArgumentParser, argparse.Action], None], argument_parser_type._check_help
    )
    if getattr(current, "_translation_audit_py314_compat", False):
        return

    original = current

    def check_help(parser: argparse.ArgumentParser, action: argparse.Action) -> None:
        action_with_runtime_help = cast(Any, action)
        if action_with_runtime_help.help is not None and not isinstance(action_with_runtime_help.help, str):
            action_with_runtime_help.help = repr(action_with_runtime_help.help)
        original(parser, action)

    cast(Any, check_help)._translation_audit_py314_compat = True
    argument_parser_type._check_help = check_help


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def _hydra_entrypoint(cfg: DictConfig) -> None:
    asyncio.run(run_pipeline(cfg))


def main() -> None:
    """Invoke the Hydra entry point."""
    enable_python314_argparse_compatibility()
    _hydra_entrypoint()
