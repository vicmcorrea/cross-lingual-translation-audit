import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import polars as pl
import pytest
from omegaconf import DictConfig, OmegaConf

from translation_audit.embeddings.backends import SyntheticEmbeddingBackend, validate_frozen_qwen_config
from translation_audit.runtime.checkpoints import CheckpointManager
from translation_audit.runtime.files import sha256_file
from translation_audit.runtime.run_workspace import RunWorkspace
from translation_audit.stages.compute_embeddings import ComputeEmbeddingsStage
from translation_audit.types import RunPaths

_REVISION = "5cf2132abc99cad020ac570b19d031efec650f2b"


def _write_input(data_root: Path) -> None:
    curated = data_root / "curated"
    curated.mkdir(parents=True)
    paired = curated / "paired_responses.parquet"
    pl.DataFrame(
        {
            "pair_id": ["pair_001", "pair_002", "pair_003"],
            "source_pt": ["pt_fixture_001", "pt_fixture_002", "pt_fixture_003"],
            "translation_en": ["en_fixture_001", "en_fixture_002", "en_fixture_003"],
        }
    ).write_parquet(paired)
    manifest = {
        "output_sha256": {"paired_responses.parquet": sha256_file(paired)},
    }
    (data_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _cfg(project_root: Path, data_root: Path, *, resume_from: Path | None = None) -> DictConfig:
    return OmegaConf.create(
        {
            "project": {"artifact_root": str(project_root / "artifacts")},
            "stage": {
                "name": "compute_embeddings",
                "implementation": "compute_embeddings",
                "backend": "synthetic",
                "resume_from_run": None if resume_from is None else str(resume_from),
                "synthetic_dimension": 8,
                "synthetic_seed": 1729,
            },
            "data": {
                "curated_dir": str(data_root / "curated"),
                "sanitized_manifest": str(data_root / "manifest.json"),
                "expected": {"paired_responses": 3},
            },
            "encoder": {
                "name": "qwen3_embedding_4b",
                "repository": "Qwen/Qwen3-Embedding-4B",
                "revision": _REVISION,
                "frozen": True,
                "trust_remote_code": False,
                "normalize_embeddings": True,
                "precision": "bfloat16",
                "batch_size": 2,
                "max_length": 32,
                "directions": {
                    "pt_to_en": {
                        "query_column": "source_pt",
                        "document_column": "translation_en",
                        "query_instruction": "fixture-direction-a",
                    },
                    "en_to_pt": {
                        "query_column": "translation_en",
                        "document_column": "source_pt",
                        "query_instruction": "fixture-direction-b",
                    },
                },
            },
            "runtime": {
                "device": "cpu",
                "checkpoint_every_rows": 2,
            },
        }
    )


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
    paths.artifacts.mkdir(parents=True)
    manager = CheckpointManager(paths.checkpoints)
    return cast(RunWorkspace, SimpleNamespace(paths=paths, checkpoints=manager))


@pytest.mark.asyncio
async def test_synthetic_embedding_stage_is_sharded_text_free_and_checkpointed(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_input(data_root)
    run_root = tmp_path / "artifacts" / "runs" / "synthetic-run"
    workspace = _workspace(run_root)
    result = await ComputeEmbeddingsStage(_cfg(tmp_path, data_root)).run(workspace)

    assert result.metrics == {
        "input_rows": 3,
        "output_records": 6,
        "embedding_dimension": 8,
        "shards": 2,
        "shards_reused": 0,
    }
    shard_paths = sorted((run_root / "artifacts" / "embeddings" / "shards").glob("*.parquet"))
    assert len(shard_paths) == 2
    shard = pl.read_parquet(shard_paths[0])
    assert shard.columns == [
        "pair_id",
        "source_row",
        "direction",
        "query_embedding",
        "document_embedding",
    ]
    assert shard.height == 4
    assert shard.schema["query_embedding"] == pl.Array(pl.Float32, 8)
    vectors = np.stack(shard["query_embedding"].to_list())
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)

    checkpoint_events = [
        json.loads(path.read_text(encoding="utf-8"))["event"]
        for path in sorted((run_root / "checkpoints").glob("*.json"))
    ]
    assert checkpoint_events == ["embedding_shard_completed", "embedding_shard_completed"]
    metadata_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in run_root.rglob("*.json")
        if path.is_file()
    )
    assert "pt_fixture" not in metadata_text
    assert "en_fixture" not in metadata_text


@pytest.mark.asyncio
async def test_embedding_stage_resumes_verified_shards_into_a_new_run(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_input(data_root)
    old_root = tmp_path / "artifacts" / "runs" / "old-run"
    await ComputeEmbeddingsStage(_cfg(tmp_path, data_root)).run(_workspace(old_root))

    new_root = tmp_path / "artifacts" / "runs" / "new-run"
    config = _cfg(tmp_path, data_root, resume_from=old_root)

    class _FailIfCalled(SyntheticEmbeddingBackend):
        def encode_queries(self, texts: object, instruction: str) -> np.ndarray:
            del texts, instruction
            raise AssertionError("A verified resumed shard must not be recomputed")

        def encode_documents(self, texts: object) -> np.ndarray:
            del texts
            raise AssertionError("A verified resumed shard must not be recomputed")

    result = await ComputeEmbeddingsStage(config, backend=_FailIfCalled(8, 1729)).run(
        _workspace(new_root)
    )
    assert result.metrics["shards_reused"] == 2
    for old_path in sorted((old_root / "artifacts" / "embeddings" / "shards").glob("*.parquet")):
        new_path = new_root / "artifacts" / "embeddings" / "shards" / old_path.name
        assert sha256_file(old_path) == sha256_file(new_path)


def test_qwen_encoder_contract_rejects_mutable_or_unapproved_models(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    config = _cfg(tmp_path, data_root)
    validate_frozen_qwen_config(config.encoder)

    config.encoder.revision = "main"
    with pytest.raises(ValueError, match="commit SHA"):
        validate_frozen_qwen_config(config.encoder)
    config.encoder.revision = _REVISION

    config.encoder.revision = "0" * 40
    with pytest.raises(ValueError, match="approved Qwen3 checkpoint"):
        validate_frozen_qwen_config(config.encoder)
    config.encoder.revision = _REVISION

    config.encoder.repository = "unapproved/model"
    with pytest.raises(ValueError, match="approved Qwen3"):
        validate_frozen_qwen_config(config.encoder)
