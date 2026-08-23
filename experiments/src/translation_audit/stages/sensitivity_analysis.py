"""Sealed sensitivity package over verified, text-free scientific artifacts."""

import json
import os
import tempfile
from pathlib import Path
from typing import cast

import polars as pl
import structlog
from omegaconf import DictConfig

from translation_audit.analysis.metrics import compute_retrieval_metrics
from translation_audit.analysis.sensitivity import (
    affective_sensitivities,
    length_stratified_r1,
    paired_encoder_differences,
    summarize_language_matched,
    summarize_retrieval_sensitivity,
)
from translation_audit.registry import register_stage
from translation_audit.runtime.dependencies import verified_named_upstream_runs
from translation_audit.runtime.files import sha256_file, write_json_exclusive, write_text_exclusive
from translation_audit.runtime.run_workspace import RunWorkspace
from translation_audit.types import StageResult

_EXPECTED_UPSTREAM_STAGES = {
    "primary_analysis": "analyze",
    "scale_analysis": "analyze",
    "primary_embeddings": "compute_embeddings",
    "language_verification": "validate_languages",
    "emotion_features": "compute_emotion_features",
}
_FORBIDDEN_TEXT_COLUMNS = {"source_pt", "translation_en"}


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


def _sealed_artifact(run_root: Path, relative_path: str) -> Path:
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


def _load_analysis(run_root: Path) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, object]]:
    manifest = _json_object(_sealed_artifact(run_root, "artifacts/analysis/manifest.json"))
    if manifest.get("response_text_columns_persisted") is not False:
        raise ValueError("Analysis input does not certify a text-free schema")
    frame = pl.read_parquet(
        _sealed_artifact(run_root, "artifacts/analysis/analysis_dataset.parquet")
    )
    summary = pl.read_parquet(_sealed_artifact(run_root, "tables/sensitivity_metrics.parquet"))
    if _FORBIDDEN_TEXT_COLUMNS.intersection(frame.columns):
        raise ValueError("Analysis input contains response text")
    return frame, summary, manifest


def _validate_embedding_model(
    analysis_manifest: dict[str, object],
    *,
    expected_repository: str,
    expected_revision: str,
) -> None:
    raw_model = analysis_manifest.get("embedding_model")
    if not isinstance(raw_model, dict):
        raise ValueError("Analysis manifest lacks an embedding model identity")
    model = cast(dict[str, object], raw_model)
    if model.get("repository") != expected_repository or model.get("revision") != expected_revision:
        raise ValueError("Analysis manifest has an unexpected embedding model identity")


def _validate_analysis_alignment(primary: pl.DataFrame, scale: pl.DataFrame) -> None:
    if (
        primary.height != scale.height
        or primary["pair_id"].n_unique() != primary.height
        or scale["pair_id"].n_unique() != scale.height
    ):
        raise ValueError("Encoder analyses do not contain aligned unique response pairs")
    retrieval_prefixes = (
        "paired_cosine_",
        "retrieval_rank_",
        "reciprocal_rank_",
        "hit_at_1_",
        "hit_at_5_",
    )
    shared_non_embedding = sorted(
        column
        for column in set(primary.columns).intersection(scale.columns)
        if not column.startswith(retrieval_prefixes)
    )
    if not primary.select(shared_non_embedding).equals(scale.select(shared_non_embedding)):
        raise ValueError("Encoder analyses disagree on shared non-embedding columns")


def _load_embeddings(run_root: Path) -> tuple[pl.DataFrame, dict[str, object]]:
    manifest = _json_object(_sealed_artifact(run_root, "artifacts/embeddings/manifest.json"))
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


def _load_language_verification(run_root: Path) -> pl.DataFrame:
    manifest = _json_object(
        _sealed_artifact(run_root, "artifacts/language_verification/manifest.json")
    )
    raw_shards = manifest.get("shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise ValueError("Language-verification manifest has no shards")
    raw_detector = manifest.get("detector")
    if not isinstance(raw_detector, dict):
        raise ValueError("Language-verification manifest lacks detector provenance")
    detector = cast(dict[str, object], raw_detector)
    if (
        detector.get("name") != "lingua-language-detector"
        or detector.get("version") != "2.2.0"
        or detector.get("candidate_languages") != ["pt", "en", "es"]
        or float(cast(float, detector.get("minimum_relative_distance"))) != 0.05
    ):
        raise ValueError("Language-verification detector differs from the frozen protocol")
    frames: list[pl.DataFrame] = []
    for raw_record in cast(list[object], raw_shards):
        if not isinstance(raw_record, dict):
            raise ValueError("Language-verification shard record is invalid")
        record = cast(dict[str, object], raw_record)
        artifact = record.get("artifact")
        expected_hash = record.get("sha256")
        if not isinstance(artifact, str) or not isinstance(expected_hash, str):
            raise ValueError("Language-verification shard record lacks provenance")
        path = _sealed_artifact(run_root, artifact)
        if sha256_file(path) != expected_hash:
            raise ValueError("Language-verification shard differs from its manifest")
        frames.append(pl.read_parquet(path))
    return pl.concat(frames, how="vertical")


def _load_emotion_features(run_root: Path) -> tuple[pl.DataFrame, dict[str, object]]:
    manifest = _json_object(
        _sealed_artifact(run_root, "artifacts/emotion_features/manifest.json")
    )
    path = _sealed_artifact(run_root, "artifacts/emotion_features/emotion_features.parquet")
    expected_hash = manifest.get("output_sha256")
    if not isinstance(expected_hash, str) or sha256_file(path) != expected_hash:
        raise ValueError("Emotion feature artifact differs from its stage manifest")
    frame = pl.read_parquet(path)
    if _FORBIDDEN_TEXT_COLUMNS.intersection(frame.columns):
        raise ValueError("Emotion feature input contains response text")
    return frame, manifest


@register_stage("sensitivity_analysis")
class SensitivityAnalysisStage:
    """Run the sealed robustness package without persisting response text."""

    stage_name = "sensitivity_analysis"

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

    async def run(self, workspace: RunWorkspace) -> StageResult:
        if workspace.checkpoints is None:
            raise RuntimeError("Checkpoint manager was not initialized")
        logger = structlog.get_logger().bind(stage=self.stage_name)
        upstream, upstream_seals = verified_named_upstream_runs(
            self.cfg, _EXPECTED_UPSTREAM_STAGES
        )
        primary, _, primary_manifest = _load_analysis(
            upstream["primary_analysis"]
        )
        scale, _, scale_manifest = _load_analysis(upstream["scale_analysis"])
        embeddings, embedding_manifest = _load_embeddings(upstream["primary_embeddings"])
        language = _load_language_verification(upstream["language_verification"])
        emotion, emotion_manifest = _load_emotion_features(upstream["emotion_features"])

        expected = self.cfg.stage.expected_models
        _validate_embedding_model(
            primary_manifest,
            expected_repository=str(expected.primary.repository),
            expected_revision=str(expected.primary.revision),
        )
        _validate_embedding_model(
            scale_manifest,
            expected_repository=str(expected.scale.repository),
            expected_revision=str(expected.scale.revision),
        )
        _validate_analysis_alignment(primary, scale)
        embedding_model = embedding_manifest.get("model")
        if not isinstance(embedding_model, dict) or cast(dict[str, object], embedding_model).get(
            "repository"
        ) != str(expected.primary.repository):
            raise ValueError("Target-deduplication embeddings are not the primary encoder")
        workspace.checkpoints.write(
            "sensitivity_inputs_validated",
            {
                "paired_rows": primary.height,
                "upstream_seals": upstream_seals,
                "response_text_columns_persisted": False,
            },
        )
        logger.info("sensitivity_inputs_validated", paired_rows=primary.height)

        repetitions = int(self.cfg.analysis.validation.bootstrap_repetitions)
        confidence_level = float(self.cfg.analysis.validation.confidence_level)
        seed = int(self.cfg.analysis.validation.seed)
        bootstrap_batch_size = int(self.cfg.analysis.compute.bootstrap_batch_size)

        figure_length = length_stratified_r1(
            primary,
            repetitions=repetitions,
            confidence_level=confidence_level,
            seed=seed,
            bootstrap_batch_size=bootstrap_batch_size,
        )

        language_matched, language_coverage = summarize_language_matched(
            primary,
            language,
            repetitions=repetitions,
            confidence_level=confidence_level,
            seed=seed,
            bootstrap_batch_size=bootstrap_batch_size,
        )
        model_differences = paired_encoder_differences(
            primary,
            scale,
            primary_model=str(expected.primary.repository),
            comparison_model=str(expected.scale.repository),
            repetitions=repetitions,
            confidence_level=confidence_level,
            seed=seed,
            bootstrap_batch_size=bootstrap_batch_size,
        )
        affective = affective_sensitivities(
            primary,
            emotion,
            repetitions=repetitions,
            confidence_level=confidence_level,
            seed=seed,
            bootstrap_batch_size=bootstrap_batch_size,
        )
        workspace.checkpoints.write(
            "sensitivity_statistical_summaries_completed",
            {
                "language_matched_rows": language_matched.height,
                "model_difference_rows": model_differences.height,
                "affective_nonzero_rows": affective.nonzero_metrics.height,
            },
        )

        metadata = primary.select(
            "pair_id",
            "participant_id",
            "pt_duplicate_group_id",
            "en_duplicate_group_id",
        ).with_row_index("source_row")
        deduplicated_retrieval = compute_retrieval_metrics(
            embeddings,
            metadata,
            batch_size=int(self.cfg.analysis.compute.retrieval_batch_size),
            requested_device=str(self.cfg.analysis.compute.device),
            deduplicate_targets=True,
        )
        candidate_counts = {
            "en_to_pt": metadata["pt_duplicate_group_id"].n_unique(),
            "pt_to_en": metadata["en_duplicate_group_id"].n_unique(),
        }
        for direction in ("en_to_pt", "pt_to_en"):
            original_rank = f"retrieval_rank_{direction}"
            comparison = deduplicated_retrieval.frame.filter(
                pl.col("direction") == direction
            ).join(primary.select("pair_id", original_rank), on="pair_id", validate="1:1")
            if comparison.filter(pl.col("retrieval_rank") > pl.col(original_rank)).height:
                raise ValueError("Target-language de-duplication worsened at least one rank")

        both_expected = pl.col("pt_matches_expected") & pl.col("en_matches_expected")
        explicit_mismatch = (
            (~pl.col("pt_matches_expected") & ~pl.col("pt_ambiguous"))
            | (~pl.col("en_matches_expected") & ~pl.col("en_ambiguous"))
        )
        ambiguous_only = ~both_expected & ~explicit_mismatch
        language_counts = language.select(
            both_expected.sum().alias("language_both_expected"),
            ambiguous_only.sum().alias("language_ambiguous_only"),
            explicit_mismatch.sum().alias("language_explicit_mismatch"),
        ).row(0, named=True)

        def _nonzero_count(profile_family: str, column: str) -> int:
            row = affective.nonzero_metrics.filter(
                pl.col("profile_family") == profile_family
            ).select(column).unique()
            if row.height != 1:
                raise ValueError("Affective nonzero summary has inconsistent subset counts")
            return int(row.item())

        def _binary_count(metric: str) -> int:
            row = affective.coverage_by_length.filter(
                (pl.col("stratum_type") == "overall") & (pl.col("metric") == metric)
            ).select("estimate", "n_pairs")
            if row.height != 1:
                raise ValueError("Affective coverage summary lacks an expected diagnostic")
            estimate, rows = row.row(0)
            return round(float(estimate) * int(rows))

        observed_counts = {
            "paired_rows": primary.height,
            **{key: int(value) for key, value in language_counts.items()},
            "target_candidates_en_to_pt": candidate_counts["en_to_pt"],
            "target_candidates_pt_to_en": candidate_counts["pt_to_en"],
            "emotion_both_nonzero_pairs": _nonzero_count("emotion", "n_pairs"),
            "emotion_both_nonzero_clusters": _nonzero_count("emotion", "n_participants"),
            "valence_both_nonzero_pairs": _nonzero_count("valence", "n_pairs"),
            "pt_semantic_nodes_zero": _binary_count("pt_semantic_node_count_zero"),
            "en_semantic_nodes_zero": _binary_count("en_semantic_node_count_zero"),
            "pt_emotion_coverage_profile_disagreement": _binary_count(
                "pt_emotion_coverage_profile_disagreement"
            ),
            "en_emotion_coverage_profile_disagreement": _binary_count(
                "en_emotion_coverage_profile_disagreement"
            ),
        }
        expected_counts = {
            str(key): int(value) for key, value in self.cfg.stage.expected_counts.items()
        }
        if observed_counts != expected_counts:
            raise ValueError("Sensitivity inputs differ from the frozen diagnostic counts")
        workspace.checkpoints.write(
            "sensitivity_acceptance_checks_completed",
            {"observed_counts": observed_counts, "deduplicated_ranks_never_worse": True},
        )
        deduplicated_metrics = summarize_retrieval_sensitivity(
            deduplicated_retrieval.frame,
            metadata,
            sensitivity_name="direction_specific_target_deduplication",
            candidate_counts=candidate_counts,
            repetitions=repetitions,
            confidence_level=confidence_level,
            seed=seed,
            bootstrap_batch_size=bootstrap_batch_size,
        )
        workspace.checkpoints.write(
            "sensitivity_target_deduplication_completed",
            {
                "device": deduplicated_retrieval.device,
                "candidate_counts": candidate_counts,
                "metric_rows": deduplicated_metrics.height,
            },
        )
        logger.info(
            "sensitivity_target_deduplication_completed",
            device=deduplicated_retrieval.device,
        )

        analysis_root = workspace.paths.artifacts / str(self.cfg.stage.output_directory)
        output_frames = {
            "language_matched_metrics": language_matched,
            "language_coverage_by_length": language_coverage,
            "target_deduplicated_retrieval_metrics": deduplicated_metrics,
            "paired_encoder_differences": model_differences,
            "affective_nonzero_metrics": affective.nonzero_metrics,
            "affective_coverage_by_length": affective.coverage_by_length,
            "figure1_length_r1": figure_length,
        }
        artifact_paths: list[Path] = []
        for stem, frame in output_frames.items():
            artifact_paths.append(_write_parquet_exclusive(workspace.paths.tables / f"{stem}.parquet", frame))
            artifact_paths.append(_write_csv_exclusive(workspace.paths.tables / f"{stem}.csv", frame))
        retrieval_path = _write_parquet_exclusive(
            analysis_root / "target_deduplicated_retrieval_per_pair.parquet",
            deduplicated_retrieval.frame,
        )
        artifact_paths.append(retrieval_path)

        manifest_path = write_json_exclusive(
            analysis_root / "manifest.json",
            {
                "stage": self.stage_name,
                "protocol": str(self.cfg.analysis.protocol),
                "response_text_columns_persisted": False,
                "upstream_run_ids": {
                    role: root.name for role, root in upstream.items()
                },
                "upstream_seal_sha256": upstream_seals,
                "primary_embedding_model": primary_manifest.get("embedding_model"),
                "scale_embedding_model": scale_manifest.get("embedding_model"),
                "emotion_repository": emotion_manifest.get("repository"),
                "target_deduplication": {
                    "policy": "one target-language duplicate group scored by the maximum member similarity",
                    "query_rows_retained": primary.height,
                    "candidate_counts": candidate_counts,
                    "device": deduplicated_retrieval.device,
                },
                "acceptance_checks": {
                    "observed_counts": observed_counts,
                    "deduplicated_ranks_never_worse": True,
                    "encoder_non_embedding_columns_identical": True,
                },
                "bootstrap": {
                    "method": "survey-specific respondent-record-clustered percentile bootstrap",
                    "filtered_estimands": "full cluster universe with zero eligible-row contributions",
                    "repetitions": repetitions,
                    "confidence_level": confidence_level,
                    "seed": seed,
                },
                "artifacts_sha256": {
                    str(path.relative_to(workspace.paths.root)): sha256_file(path)
                    for path in artifact_paths
                },
            },
        )
        return StageResult(
            metrics={
                "paired_rows": primary.height,
                "language_matched_rows": int(
                    language.filter(
                        pl.col("pt_matches_expected") & pl.col("en_matches_expected")
                    ).height
                ),
                "target_deduplicated_metric_rows": deduplicated_metrics.height,
                "paired_encoder_difference_rows": model_differences.height,
                "affective_nonzero_metric_rows": affective.nonzero_metrics.height,
            },
            artifacts=tuple(
                str(path.relative_to(workspace.paths.root))
                for path in (*artifact_paths, manifest_path)
            ),
            details={
                "claim_scope": "prespecified automated sensitivities only",
                "response_text_persisted": False,
                "retrieval_device": deduplicated_retrieval.device,
            },
        )
