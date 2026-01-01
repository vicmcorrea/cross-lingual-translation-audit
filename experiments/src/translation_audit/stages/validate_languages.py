"""Checkpointed language verification before model inference."""

from pathlib import Path
from typing import cast

import polars as pl
import structlog
from omegaconf import DictConfig

from translation_audit.language import LanguageDetector, LinguaDetector
from translation_audit.registry import register_stage
from translation_audit.runtime.files import sha256_file, write_json_exclusive
from translation_audit.runtime.run_workspace import RunWorkspace
from translation_audit.types import StageResult

_INPUT_COLUMNS = ("pair_id", "source_pt", "translation_en")


def _write_parquet_exclusive(frame: pl.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as output:
        frame.write_parquet(output, compression="zstd", statistics=True)


@register_stage("validate_languages")
class ValidateLanguagesStage:
    """Flag language mismatches and ambiguity without automatically removing rows."""

    stage_name = "validate_languages"

    def __init__(self, cfg: DictConfig, detector: LanguageDetector | None = None) -> None:
        self.cfg = cfg
        self._detector = detector

    async def run(self, workspace: RunWorkspace) -> StageResult:
        return self._run_sync(workspace)

    def _run_sync(self, workspace: RunWorkspace) -> StageResult:
        if workspace.checkpoints is None:
            raise RuntimeError("Language verification requires an initialized checkpoint manager")
        input_path = Path(str(self.cfg.stage.input_path)).resolve()
        if not input_path.is_file():
            raise FileNotFoundError(f"Configured paired-response input is unavailable at {input_path}")
        expected_sha = str(self.cfg.stage.input_sha256)
        actual_sha = sha256_file(input_path)
        if expected_sha and expected_sha != actual_sha:
            raise ValueError("Paired-response checksum does not match the frozen language-verification input")

        detector = self._detector or LinguaDetector(float(self.cfg.stage.minimum_relative_distance))
        frame = pl.read_parquet(input_path, columns=list(_INPUT_COLUMNS), low_memory=True)
        shard_size = int(self.cfg.runtime.checkpoint_every_rows)
        output_root = workspace.paths.artifacts / "language_verification"
        shard_root = output_root / "shards"
        shard_root.mkdir(parents=True, exist_ok=False)
        logger = structlog.get_logger().bind(stage=self.stage_name)
        records: list[dict[str, object]] = []
        total = pt_match = en_match = pt_ambiguous = en_ambiguous = 0

        for shard_index, batch in enumerate(frame.iter_slices(n_rows=shard_size)):
            values = batch.to_dict(as_series=False)
            pair_ids = cast(list[str], values["pair_id"])
            pt_texts = cast(list[str], values["source_pt"])
            en_texts = cast(list[str], values["translation_en"])
            pt_detections = [detector.detect(text, "pt") for text in pt_texts]
            en_detections = [detector.detect(text, "en") for text in en_texts]
            shard = pl.DataFrame(
                {
                    "pair_id": pair_ids,
                    "pt_predicted": [item.predicted for item in pt_detections],
                    "pt_expected_confidence": [item.expected_confidence for item in pt_detections],
                    "pt_matches_expected": [item.predicted == "pt" for item in pt_detections],
                    "pt_ambiguous": [item.predicted is None for item in pt_detections],
                    "en_predicted": [item.predicted for item in en_detections],
                    "en_expected_confidence": [item.expected_confidence for item in en_detections],
                    "en_matches_expected": [item.predicted == "en" for item in en_detections],
                    "en_ambiguous": [item.predicted is None for item in en_detections],
                }
            )
            shard_path = shard_root / f"shard_{shard_index:06d}.parquet"
            _write_parquet_exclusive(shard, shard_path)
            shard_sha = sha256_file(shard_path)
            row_start = total
            total += shard.height
            pt_match += int(shard["pt_matches_expected"].sum())
            en_match += int(shard["en_matches_expected"].sum())
            pt_ambiguous += int(shard["pt_ambiguous"].sum())
            en_ambiguous += int(shard["en_ambiguous"].sum())
            record: dict[str, object] = {
                "shard_index": shard_index,
                "row_start": row_start,
                "row_stop": total,
                "rows": shard.height,
                "artifact": str(shard_path.relative_to(workspace.paths.root)),
                "sha256": shard_sha,
            }
            records.append(record)
            workspace.checkpoints.write("language_shard_completed", record)
            logger.info("language_shard_completed", shard_index=shard_index, rows=shard.height)

        if total != frame.height or total == 0:
            raise ValueError("Language verification did not cover the complete non-empty cohort")
        manifest_path = write_json_exclusive(
            output_root / "manifest.json",
            {
                "input_sha256": actual_sha,
                "detector": {
                    "name": "lingua-language-detector",
                    "version": "2.2.0",
                    "candidate_languages": ["pt", "en", "es"],
                    "minimum_relative_distance": float(self.cfg.stage.minimum_relative_distance),
                },
                "policy": "flag_only_no_automatic_exclusion",
                "rows": total,
                "shards": records,
            },
        )
        metrics: dict[str, float | int] = {
            "rows": total,
            "pt_matches_expected": pt_match,
            "en_matches_expected": en_match,
            "pt_ambiguous": pt_ambiguous,
            "en_ambiguous": en_ambiguous,
        }
        write_json_exclusive(output_root / "summary.json", metrics)
        return StageResult(
            metrics=metrics,
            artifacts=(
                str(manifest_path.relative_to(workspace.paths.root)),
                str((output_root / "summary.json").relative_to(workspace.paths.root)),
            ),
            details={"policy": "flag_only_no_automatic_exclusion", "input_sha256": actual_sha},
        )
