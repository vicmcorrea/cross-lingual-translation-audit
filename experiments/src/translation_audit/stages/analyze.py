"""Final text-free analysis over sealed upstream scientific artifacts."""

import json
import os
import tempfile
from pathlib import Path
from typing import cast

import polars as pl
import structlog
from omegaconf import DictConfig

from translation_audit.analysis import (
    automated_convergent_validation,
    compute_retrieval_metrics,
    summarize_strata,
)
from translation_audit.registry import register_stage
from translation_audit.runtime.dependencies import verified_upstream_runs
from translation_audit.runtime.files import sha256_file, write_json_exclusive, write_text_exclusive
from translation_audit.runtime.run_workspace import RunWorkspace
from translation_audit.types import StageResult

_UPSTREAM_STAGES = (
    "compute_embeddings",
    "estimate_translation_quality",
    "compute_emotion_features",
)
_METADATA_COLUMNS = (
    "pair_id",
    "participant_id",
    "cohort_id",
    "prompt_family",
    "pt_word_count",
    "exact_normalized_copy",
    "pt_duplicate_group_id",
    "en_duplicate_group_id",
)
_EMOTION_COLUMNS = (
    "pair_id",
    "emotion_jensen_shannon_distance",
    "valence_jensen_shannon_distance",
    "emotion_mean_absolute_share_delta",
    "valence_mean_absolute_share_delta",
    "semantic_density_absolute_delta",
)


def _write_parquet_exclusive(path: Path, frame: pl.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        frame.write_parquet(temporary_path, compression="zstd", statistics=True)
        with temporary_path.open("rb") as stream:
            os.fsync(stream.fileno())
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def _write_csv_exclusive(path: Path, frame: pl.DataFrame) -> Path:
    return write_text_exclusive(path, frame.write_csv(float_precision=8))


def _json_object(path: Path) -> dict[str, object]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object in scientific artifact")
    return cast(dict[str, object], value)


def _verified_upstream_runs(cfg: DictConfig) -> dict[str, Path]:
    """Verify every sealed upstream byte before resolving stage artifacts."""
    run_roots, _ = verified_upstream_runs(cfg, _UPSTREAM_STAGES)
    return run_roots


def _sealed_artifact(run_root: Path, relative_path: str) -> Path:
    """Resolve one conventional artifact and require its exact hash in the run seal."""
    candidate = (run_root / relative_path).resolve()
    if not candidate.is_relative_to(run_root) or not candidate.is_file():
        raise ValueError("Required upstream artifact is unavailable")
    seal = _json_object(run_root / "run_seal.json")
    raw_files = seal.get("files_sha256")
    if not isinstance(raw_files, dict):
        raise ValueError("Upstream run seal is invalid")
    expected = cast(dict[str, object], raw_files).get(relative_path)
    if not isinstance(expected, str) or sha256_file(candidate) != expected:
        raise ValueError("Required upstream artifact is not covered by its run seal")
    return candidate


def _load_embeddings(run_root: Path) -> tuple[pl.DataFrame, dict[str, object]]:
    manifest_path = _sealed_artifact(run_root, "artifacts/embeddings/manifest.json")
    manifest = _json_object(manifest_path)
    raw_shards = manifest.get("shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise ValueError("Embedding manifest has no shards")
    frames: list[pl.DataFrame] = []
    for raw_record in cast(list[object], raw_shards):
        if not isinstance(raw_record, dict):
            raise ValueError("Embedding shard record is invalid")
        record = cast(dict[str, object], raw_record)
        artifact = record.get("artifact")
        expected_hash = record.get("sha256")
        if not isinstance(artifact, str) or not isinstance(expected_hash, str):
            raise ValueError("Embedding shard record lacks provenance")
        path = _sealed_artifact(run_root, artifact)
        if sha256_file(path) != expected_hash:
            raise ValueError("Embedding shard differs from its stage manifest")
        frames.append(pl.read_parquet(path))
    return pl.concat(frames, how="vertical"), manifest


def _load_conventional_parquet(
    run_root: Path,
    *,
    directory: str,
    filename: str,
) -> tuple[pl.DataFrame, dict[str, object]]:
    manifest_relative = f"artifacts/{directory}/manifest.json"
    manifest = _json_object(_sealed_artifact(run_root, manifest_relative))
    parquet_relative = f"artifacts/{directory}/{filename}"
    path = _sealed_artifact(run_root, parquet_relative)
    expected_hash = manifest.get("output_sha256")
    if not isinstance(expected_hash, str) or sha256_file(path) != expected_hash:
        raise ValueError("Upstream Parquet differs from its stage manifest")
    return pl.read_parquet(path), manifest


def _load_curated(cfg: DictConfig) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, str]]:
    curated_root = Path(str(cfg.data.curated_dir)).resolve()
    manifest = _json_object(Path(str(cfg.data.sanitized_manifest)).resolve())
    raw_hashes = manifest.get("output_sha256")
    if not isinstance(raw_hashes, dict):
        raise ValueError("Sanitized cohort manifest has no output fingerprints")
    hashes = cast(dict[str, object], raw_hashes)
    outputs: dict[str, str] = {}
    for filename in ("paired_responses.parquet", "splits.parquet"):
        path = curated_root / filename
        expected = hashes.get(filename)
        actual = sha256_file(path)
        if not isinstance(expected, str) or actual != expected:
            raise ValueError("Curated analysis input differs from its sanitized manifest")
        outputs[filename] = actual
    metadata = pl.read_parquet(
        curated_root / "paired_responses.parquet", columns=list(_METADATA_COLUMNS)
    ).with_row_index("source_row")
    splits = pl.read_parquet(curated_root / "splits.parquet")
    if metadata.height != int(cfg.data.expected.paired_responses):
        raise ValueError("Curated metadata row count differs from the study contract")
    return metadata, splits, outputs


def _wide_retrieval(retrieval: pl.DataFrame) -> pl.DataFrame:
    outputs: list[pl.DataFrame] = []
    for direction in ("en_to_pt", "pt_to_en"):
        selected = retrieval.filter(pl.col("direction") == direction).select(
            "pair_id",
            pl.col("paired_cosine").alias(f"paired_cosine_{direction}"),
            pl.col("retrieval_rank").alias(f"retrieval_rank_{direction}"),
            pl.col("reciprocal_rank").alias(f"reciprocal_rank_{direction}"),
            pl.col("hit_at_1").alias(f"hit_at_1_{direction}"),
            pl.col("hit_at_5").alias(f"hit_at_5_{direction}"),
        )
        outputs.append(selected)
    if len(outputs) != 2:
        raise ValueError("Retrieval output has no configured directions")
    return outputs[0].join(outputs[1], on="pair_id", how="inner")


@register_stage("analyze")
class AnalyzeStage:
    """Produce descriptive publication tables from verified, text-free model artifacts."""

    stage_name = "analyze"

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

    async def run(self, workspace: RunWorkspace) -> StageResult:
        if workspace.checkpoints is None:
            raise RuntimeError("Checkpoint manager was not initialized")
        logger = structlog.get_logger().bind(stage=self.stage_name)
        upstream = _verified_upstream_runs(self.cfg)
        metadata, splits, curated_hashes = _load_curated(self.cfg)
        embeddings, embedding_manifest = _load_embeddings(upstream["compute_embeddings"])
        comet, comet_manifest = _load_conventional_parquet(
            upstream["estimate_translation_quality"],
            directory="translation_quality",
            filename="cometkiwi_scores.parquet",
        )
        emotion, emotion_manifest = _load_conventional_parquet(
            upstream["compute_emotion_features"],
            directory="emotion_features",
            filename="emotion_features.parquet",
        )
        workspace.checkpoints.write(
            "analysis_inputs_validated",
            {
                "paired_rows": metadata.height,
                "upstream_stages": list(_UPSTREAM_STAGES),
                "curated_sha256": curated_hashes,
            },
        )
        logger.info("analysis_inputs_validated", paired_rows=metadata.height)

        retrieval_output = compute_retrieval_metrics(
            embeddings,
            metadata,
            batch_size=int(self.cfg.analysis.compute.retrieval_batch_size),
            requested_device=str(self.cfg.analysis.compute.device),
        )
        analysis_root = workspace.paths.artifacts / str(self.cfg.stage.output_directory)
        retrieval_path = _write_parquet_exclusive(
            analysis_root / "retrieval_per_pair.parquet", retrieval_output.frame
        )
        workspace.checkpoints.write(
            "analysis_retrieval_completed",
            {
                "rows": retrieval_output.frame.height,
                "device": retrieval_output.device,
                "sha256": sha256_file(retrieval_path),
            },
        )
        logger.info(
            "analysis_retrieval_completed",
            rows=retrieval_output.frame.height,
            device=retrieval_output.device,
        )

        if set(comet.columns) != {"pair_id", "cometkiwi_score"}:
            raise ValueError("COMET artifact has an invalid analysis schema")
        missing_emotion = set(_EMOTION_COLUMNS).difference(emotion.columns)
        if missing_emotion:
            raise ValueError("Emotion artifact lacks preservation features")
        frame = (
            metadata.join(_wide_retrieval(retrieval_output.frame), on="pair_id", how="inner")
            .join(comet, on="pair_id", how="inner")
            .join(emotion.select(_EMOTION_COLUMNS), on="pair_id", how="inner")
            .sort("source_row")
        )
        null_values = int(frame.null_count().select(pl.sum_horizontal(pl.all())).item())
        if frame.height != metadata.height or null_values != 0:
            raise ValueError("Analysis artifacts do not align one-to-one with curated pairs")
        analysis_dataset_path = _write_parquet_exclusive(
            analysis_root / "analysis_dataset.parquet", frame
        )

        summary = summarize_strata(
            frame,
            repetitions=int(self.cfg.analysis.validation.bootstrap_repetitions),
            confidence_level=float(self.cfg.analysis.validation.confidence_level),
            seed=int(self.cfg.analysis.validation.seed),
            bootstrap_batch_size=int(self.cfg.analysis.compute.bootstrap_batch_size),
        )
        primary = summary.filter(
            (pl.col("stratum_type") == "overall") & (pl.col("stratum_value") == "all")
        )
        primary_parquet = _write_parquet_exclusive(
            workspace.paths.tables / "primary_metrics.parquet", primary
        )
        primary_csv = _write_csv_exclusive(workspace.paths.tables / "primary_metrics.csv", primary)
        sensitivity_parquet = _write_parquet_exclusive(
            workspace.paths.tables / "sensitivity_metrics.parquet", summary
        )
        sensitivity_csv = _write_csv_exclusive(
            workspace.paths.tables / "sensitivity_metrics.csv", summary
        )
        workspace.checkpoints.write(
            "analysis_summaries_completed",
            {
                "primary_rows": primary.height,
                "sensitivity_rows": summary.height,
                "primary_sha256": sha256_file(primary_parquet),
                "sensitivity_sha256": sha256_file(sensitivity_parquet),
            },
        )

        validation = automated_convergent_validation(
            frame,
            splits,
            alphas=[float(value) for value in self.cfg.analysis.validation.ridge_alphas],
            default_alpha=float(self.cfg.analysis.validation.ridge_default_alpha),
        )
        validation_parquet = _write_parquet_exclusive(
            workspace.paths.tables / "automated_convergent_validation.parquet", validation.metrics
        )
        validation_csv = _write_csv_exclusive(
            workspace.paths.tables / "automated_convergent_validation.csv", validation.metrics
        )
        predictions_path = _write_parquet_exclusive(
            analysis_root / "comet_cross_validated_predictions.parquet", validation.predictions
        )
        for row in validation.metrics.iter_rows(named=True):
            workspace.checkpoints.write(
                "analysis_validation_fold_completed",
                {
                    "fold_id": str(row["fold_id"]),
                    "held_out_cohort": str(row["held_out_cohort"]),
                    "n_test_pairs": int(row["n_test_pairs"]),
                    "selected_alpha": float(row["selected_alpha"]),
                },
            )
        logger.info("analysis_validation_completed", folds=validation.metrics.height)

        artifact_paths = (
            retrieval_path,
            analysis_dataset_path,
            predictions_path,
            primary_parquet,
            primary_csv,
            sensitivity_parquet,
            sensitivity_csv,
            validation_parquet,
            validation_csv,
        )
        manifest_path = write_json_exclusive(
            analysis_root / "manifest.json",
            {
                "stage": self.stage_name,
                "protocol": str(self.cfg.analysis.protocol),
                "response_text_columns_persisted": False,
                "comet_interpretation": "automated reference-free criterion; not human gold",
                "claim_scope": "descriptive estimates and exploratory automated convergence only",
                "embedding_model": embedding_manifest.get("model"),
                "comet_model_repository": comet_manifest.get("model_repository"),
                "comet_model_revision": comet_manifest.get("model_revision"),
                "emotion_repository": emotion_manifest.get("repository"),
                "emotion_revision": emotion_manifest.get("revision"),
                "retrieval_device": retrieval_output.device,
                "bootstrap": {
                    "method": "survey-specific respondent-record-cluster percentile bootstrap",
                    "repetitions": int(self.cfg.analysis.validation.bootstrap_repetitions),
                    "confidence_level": float(self.cfg.analysis.validation.confidence_level),
                    "seed": int(self.cfg.analysis.validation.seed),
                },
                "artifacts_sha256": {
                    str(path.relative_to(workspace.paths.root)): sha256_file(path)
                    for path in artifact_paths
                },
            },
        )
        return StageResult(
            metrics={
                "paired_rows": frame.height,
                "retrieval_rows": retrieval_output.frame.height,
                "summary_rows": summary.height,
                "validation_folds": validation.metrics.height,
            },
            artifacts=tuple(
                str(path.relative_to(workspace.paths.root)) for path in (*artifact_paths, manifest_path)
            ),
            details={
                "comet_role": "automated criterion, not human gold",
                "claim_scope": "descriptive_only",
                "retrieval_device": retrieval_output.device,
            },
        )
