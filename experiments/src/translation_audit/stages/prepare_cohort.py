"""Hydra stage for deterministic cohort construction."""

from pathlib import Path

from omegaconf import DictConfig

from translation_audit.data.cohort import build_curated_cohort, validate_transfer_cohort
from translation_audit.registry import register_stage
from translation_audit.types import StageResult


def _host_inputs_available(raw_snapshot: Path, private_selection: Path) -> bool:
    """Check whether the full host-only cohort inputs are present."""
    return raw_snapshot.is_dir() and private_selection.is_file()


@register_stage("prepare_cohort")
class PrepareCohortStage:
    """Build or checksum-validate the configured immutable cohort."""

    stage_name = "prepare_cohort"

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

    async def run(self, workspace: object) -> StageResult:
        del workspace
        raw_snapshot = Path(str(self.cfg.data.raw_snapshot))
        curated_dir = Path(str(self.cfg.data.curated_dir))
        sanitized_manifest = Path(str(self.cfg.data.sanitized_manifest))
        private_selection = Path(str(self.cfg.data.private_selection_manifest))
        if _host_inputs_available(raw_snapshot, private_selection):
            manifest = build_curated_cohort(
                raw_snapshot=raw_snapshot,
                destination=curated_dir,
                sanitized_manifest_path=sanitized_manifest,
                private_selection_path=private_selection,
            )
            validation_mode = "host_immutable_cohort"
        else:
            manifest = validate_transfer_cohort(curated_dir, sanitized_manifest)
            validation_mode = "minimal_gpu_transfer"
        counts = manifest["counts"]
        return StageResult(
            metrics={
                "participants": int(counts["participants"]),
                "paired_responses": int(counts["logical_pairs"]),
            },
            artifacts=(str(curated_dir),),
            details={
                "curated_version": str(manifest["curated_version"]),
                "validation_mode": validation_mode,
            },
        )
