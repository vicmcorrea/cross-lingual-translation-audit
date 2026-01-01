import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from omegaconf import DictConfig, OmegaConf

from translation_audit.analysis.metrics import (
    clustered_bootstrap_mean,
    compute_retrieval_metrics,
    iter_similarity_batches,
)
from translation_audit.runtime.checkpoints import CheckpointManager
from translation_audit.runtime.files import sha256_file
from translation_audit.runtime.run_workspace import RunWorkspace
from translation_audit.stages.analyze import AnalyzeStage
from translation_audit.types import RunPaths


def _retrieval_fixture() -> tuple[pl.DataFrame, pl.DataFrame]:
    metadata = pl.DataFrame(
        {
            "source_row": [0, 1, 2],
            "pair_id": ["p0", "p1", "p2"],
            "pt_duplicate_group_id": ["pt-a", "pt-a", "pt-b"],
            "en_duplicate_group_id": ["en-x", "en-y", "en-x"],
        }
    )
    rows: list[dict[str, object]] = []
    specifications = {
        "en_to_pt": (
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.8, 0.2, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ),
        "pt_to_en": (
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.8, 0.2, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        ),
    }
    for direction, (queries, documents) in specifications.items():
        rows.extend(
            {
                "pair_id": f"p{index}",
                "source_row": index,
                "direction": direction,
                "query_embedding": query,
                "document_embedding": document,
            }
            for index, (query, document) in enumerate(zip(queries, documents, strict=True))
        )
    return pl.DataFrame(rows), metadata


def test_duplicate_aware_retrieval_uses_the_document_language_group() -> None:
    embeddings, metadata = _retrieval_fixture()
    output = compute_retrieval_metrics(
        embeddings, metadata, batch_size=1, requested_device="cpu"
    ).frame
    pair_zero = output.filter(pl.col("pair_id") == "p0").sort("direction")
    assert pair_zero["direction"].to_list() == ["en_to_pt", "pt_to_en"]
    assert pair_zero["retrieval_rank"].to_list() == [1, 1]
    assert pair_zero["hit_at_1"].to_list() == [True, True]


def test_similarity_batches_are_lazy_and_bounded() -> None:
    queries = np.eye(7, dtype=np.float32)
    documents = np.eye(7, dtype=np.float32)
    batches, device = iter_similarity_batches(queries, documents, 2, "cpu")
    assert device == "cpu"
    assert not isinstance(batches, list)
    iterator = iter(batches)
    assert next(iterator).shape == (2, 7)
    assert [batch.shape for batch in iterator] == [(2, 7), (2, 7), (1, 7)]


def test_cluster_bootstrap_is_deterministic_and_participant_clustered() -> None:
    values = np.asarray([1.0, 3.0, 2.0, 4.0])
    participants = ["a", "a", "b", "b"]
    first = clustered_bootstrap_mean(
        values,
        participants,
        repetitions=200,
        confidence_level=0.95,
        seed=42,
        batch_size=17,
    )
    second = clustered_bootstrap_mean(
        values,
        participants,
        repetitions=200,
        confidence_level=0.95,
        seed=42,
        batch_size=17,
    )
    assert first == second
    assert first[0] == 2.5


def _write_curated(data_root: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    curated = data_root / "curated"
    curated.mkdir(parents=True)
    pair_ids = [f"pair-{index}" for index in range(8)]
    participants = [f"participant-{index}" for index in range(8)]
    cohorts = ["cohort-a"] * 4 + ["cohort-b"] * 4
    pairs = pl.DataFrame(
        {
            "pair_id": pair_ids,
            "participant_id": participants,
            "cohort_id": cohorts,
            "prompt_family": ["positive", "improvement"] * 4,
            "pt_word_count": list(range(1, 9)),
            "exact_normalized_copy": [True, False, False, False, False, False, False, False],
            "pt_duplicate_group_id": [f"pt-{index}" for index in range(8)],
            "en_duplicate_group_id": [f"en-{index}" for index in range(8)],
            "source_pt": [f"PRIVATE_PT_MARKER_{index}" for index in range(8)],
            "translation_en": [f"PRIVATE_EN_MARKER_{index}" for index in range(8)],
        }
    )
    pairs_path = curated / "paired_responses.parquet"
    pairs.write_parquet(pairs_path)
    split_rows: list[dict[str, str]] = []
    for fold_id, held_out in (("test-cohort-a", "cohort-a"), ("test-cohort-b", "cohort-b")):
        split_rows.extend(
            {
                "participant_id": participant,
                "fold_id": fold_id,
                "role": "test" if cohort == held_out else "train",
            }
            for participant, cohort in zip(participants, cohorts, strict=True)
        )
    splits = pl.DataFrame(split_rows)
    splits_path = curated / "splits.parquet"
    splits.write_parquet(splits_path)
    (data_root / "manifest.json").write_text(
        json.dumps(
            {
                "output_sha256": {
                    pairs_path.name: sha256_file(pairs_path),
                    splits_path.name: sha256_file(splits_path),
                }
            }
        ),
        encoding="utf-8",
    )
    return pairs, splits


def _seal_run(run_root: Path, stage_name: str) -> Path:
    config_root = run_root / "artifacts" / "config"
    config_root.mkdir(parents=True, exist_ok=True)
    (config_root / "resolved.yaml").write_text(f"stage:\n  name: {stage_name}\n", encoding="utf-8")
    (config_root / "overrides.txt").write_text("\n", encoding="utf-8")
    start = run_root / "manifests" / "000_start.json"
    start.parent.mkdir(parents=True, exist_ok=True)
    start.write_text(json.dumps({"status": "started", "stage": stage_name}), encoding="utf-8")
    terminal = run_root / "manifests" / "999_end.json"
    terminal.parent.mkdir(parents=True, exist_ok=True)
    terminal.write_text(
        json.dumps({"status": "completed", "stage": stage_name}), encoding="utf-8"
    )
    files = {
        str(path.relative_to(run_root)): sha256_file(path)
        for path in run_root.rglob("*")
        if path.is_file()
    }
    (run_root / "run_seal.json").write_text(
        json.dumps({"run_id": run_root.name, "files_sha256": files}), encoding="utf-8"
    )
    return terminal


def _write_upstream_runs(artifact_root: Path, pairs: pl.DataFrame) -> dict[str, Path]:
    pair_ids = pairs["pair_id"].to_list()
    dimension = len(pair_ids)
    identity = np.eye(dimension, dtype=np.float32)

    embedding_root = artifact_root / "runs" / "embeddings"
    embedding_shard = embedding_root / "artifacts" / "embeddings" / "shards" / "shard.parquet"
    embedding_shard.parent.mkdir(parents=True)
    embedding_rows: list[dict[str, object]] = []
    for direction in ("en_to_pt", "pt_to_en"):
        embedding_rows.extend(
            {
                "pair_id": str(pair_id),
                "source_row": index,
                "direction": direction,
                "query_embedding": identity[index].tolist(),
                "document_embedding": identity[index].tolist(),
            }
            for index, pair_id in enumerate(pair_ids)
        )
    pl.DataFrame(embedding_rows).write_parquet(embedding_shard)
    embedding_manifest = embedding_root / "artifacts" / "embeddings" / "manifest.json"
    embedding_manifest.write_text(
        json.dumps(
            {
                "model": {"repository": "synthetic", "revision": "0" * 40},
                "shards": [
                    {
                        "artifact": str(embedding_shard.relative_to(embedding_root)),
                        "sha256": sha256_file(embedding_shard),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    comet_root = artifact_root / "runs" / "comet"
    comet_path = comet_root / "artifacts" / "translation_quality" / "cometkiwi_scores.parquet"
    comet_path.parent.mkdir(parents=True)
    pl.DataFrame(
        {"pair_id": pair_ids, "cometkiwi_score": np.linspace(0.2, 0.9, len(pair_ids))}
    ).write_parquet(comet_path)
    (comet_path.parent / "manifest.json").write_text(
        json.dumps(
            {
                "output_sha256": sha256_file(comet_path),
                "model_repository": "Unbabel/synthetic-comet",
                "model_revision": "1" * 40,
            }
        ),
        encoding="utf-8",
    )

    emotion_root = artifact_root / "runs" / "emotion"
    emotion_path = emotion_root / "artifacts" / "emotion_features" / "emotion_features.parquet"
    emotion_path.parent.mkdir(parents=True)
    emotion_values = np.linspace(0.01, 0.08, len(pair_ids))
    pl.DataFrame(
        {
            "pair_id": pair_ids,
            "emotion_jensen_shannon_distance": emotion_values,
            "valence_jensen_shannon_distance": emotion_values * 0.8,
            "emotion_mean_absolute_share_delta": emotion_values * 0.6,
            "valence_mean_absolute_share_delta": emotion_values * 0.4,
            "semantic_density_absolute_delta": emotion_values * 0.2,
        }
    ).write_parquet(emotion_path)
    (emotion_path.parent / "manifest.json").write_text(
        json.dumps(
            {
                "output_sha256": sha256_file(emotion_path),
                "repository": "synthetic-emoatlas",
                "revision": "2" * 40,
            }
        ),
        encoding="utf-8",
    )
    return {
        "compute_embeddings": _seal_run(embedding_root, "compute_embeddings"),
        "estimate_translation_quality": _seal_run(comet_root, "estimate_translation_quality"),
        "compute_emotion_features": _seal_run(emotion_root, "compute_emotion_features"),
    }


def _workspace(root: Path) -> RunWorkspace:
    paths = RunPaths(
        root=root,
        artifacts=root / "artifacts",
        checkpoints=root / "checkpoints",
        figures=root / "figures",
        logs=root / "logs",
        manifests=root / "manifests",
        metrics=root / "metrics",
        tables=root / "tables",
    )
    for path in (paths.artifacts, paths.tables):
        path.mkdir(parents=True)
    workspace = object.__new__(RunWorkspace)
    workspace.paths = paths
    workspace.checkpoints = CheckpointManager(paths.checkpoints)
    return workspace


def _config(
    project_root: Path,
    data_root: Path,
    manifests: dict[str, Path],
) -> DictConfig:
    return OmegaConf.create(
        {
            "project": {"artifact_root": str(project_root / "artifacts")},
            "stage": {
                "name": "analyze",
                "implementation": "analyze",
                "output_directory": "analysis",
                "upstream_manifests": {key: str(value) for key, value in manifests.items()},
            },
            "data": {
                "curated_dir": str(data_root / "curated"),
                "sanitized_manifest": str(data_root / "manifest.json"),
                "expected": {"paired_responses": 8},
            },
            "analysis": {
                "protocol": "synthetic-test",
                "compute": {
                    "device": "cpu",
                    "retrieval_batch_size": 3,
                    "bootstrap_batch_size": 7,
                },
                "validation": {
                    "bootstrap_repetitions": 50,
                    "confidence_level": 0.95,
                    "seed": 1234,
                    "ridge_alphas": [0.1, 1.0, 10.0],
                    "ridge_default_alpha": 1.0,
                },
            },
        }
    )


@pytest.mark.asyncio
async def test_analysis_stage_uses_sealed_inputs_and_writes_descriptive_text_free_tables(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    pairs, _ = _write_curated(data_root)
    manifests = _write_upstream_runs(tmp_path / "artifacts", pairs)
    run_root = tmp_path / "artifacts" / "runs" / "analysis"
    result = await AnalyzeStage(_config(tmp_path, data_root, manifests)).run(_workspace(run_root))
    assert result.metrics["paired_rows"] == 8
    assert result.metrics["retrieval_rows"] == 16
    assert result.metrics["validation_folds"] == 2

    primary = pl.read_parquet(run_root / "tables" / "primary_metrics.parquet")
    comet_row = primary.filter(pl.col("metric") == "cometkiwi_score")
    assert comet_row.height == 1
    assert "not human gold" in comet_row["evidence_role"].item()
    validation = pl.read_parquet(
        run_root / "tables" / "automated_convergent_validation.parquet"
    )
    assert validation.height == 2
    assert all("not human gold" in value for value in validation["criterion"].to_list())

    analysis_dataset = pl.read_parquet(run_root / "artifacts" / "analysis" / "analysis_dataset.parquet")
    assert "source_pt" not in analysis_dataset.columns
    assert "translation_en" not in analysis_dataset.columns
    persisted = "\n".join(
        path.read_text(encoding="utf-8")
        for path in run_root.rglob("*")
        if path.is_file() and path.suffix in {".json", ".csv"}
    )
    assert "PRIVATE_PT_MARKER" not in persisted
    assert "PRIVATE_EN_MARKER" not in persisted


@pytest.mark.asyncio
async def test_analysis_accepts_verified_imported_runs(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    pairs, _ = _write_curated(data_root)
    imported_parent = tmp_path / "imported"
    manifests = _write_upstream_runs(imported_parent, pairs)
    config = _config(tmp_path, data_root, manifests)
    config.project.imported_run_root = str(imported_parent / "runs")
    result = await AnalyzeStage(config).run(
        _workspace(tmp_path / "artifacts" / "runs" / "imported-analysis")
    )
    assert result.metrics["paired_rows"] == 8


@pytest.mark.asyncio
async def test_analysis_stage_rejects_a_tampered_upstream_seal(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    pairs, _ = _write_curated(data_root)
    manifests = _write_upstream_runs(tmp_path / "artifacts", pairs)
    embedding_shard = (
        tmp_path
        / "artifacts"
        / "runs"
        / "embeddings"
        / "artifacts"
        / "embeddings"
        / "shards"
        / "shard.parquet"
    )
    embedding_shard.write_bytes(embedding_shard.read_bytes() + b"tamper")
    config = _config(tmp_path, data_root, manifests)
    with pytest.raises(ValueError, match="seal verification failed"):
        await AnalyzeStage(config).run(
            _workspace(tmp_path / "artifacts" / "runs" / "analysis")
        )
