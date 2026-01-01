"""Explicit contracts for model stages that are not executed during setup."""

from omegaconf import DictConfig

from translation_audit.registry import register_stage
from translation_audit.types import StageNotImplementedError, StageResult


@register_stage("contract_only")
class ContractOnlyStage:
    """Prevent an unreviewed model stage from running accidentally."""

    stage_name = "contract_only"

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

    async def run(self, workspace: object) -> StageResult:
        del workspace
        raise StageNotImplementedError(
            f"Stage {self.cfg.stage.name} has a frozen contract but no approved implementation"
        )
