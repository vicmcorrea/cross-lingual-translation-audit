"""Resumable Hydra stage for frozen Qwen3 multilingual embeddings."""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
import structlog
from omegaconf import DictConfig, OmegaConf

from translation_audit.embeddings import EmbeddingBackend, create_embedding_backend
from translation_audit.embeddings.backends import validate_frozen_qwen_config
from translation_audit.registry import register_stage
from translation_audit.runtime.files import sha256_file, write_json_exclusive
from translation_audit.runtime.run_workspace import RunWorkspace
from translation_audit.types import StageResult

_INPUT_COLUMNS = ("pair_id", "source_pt", "translation_en")
_ALLOWED_TEXT_COLUMNS = frozenset({"source_pt", "translation_en"})


@dataclass(frozen=True, slots=True)
class ShardRecord:
    """Non-sensitive provenance for one embedding artifact."""

    shard_index: int
    row_start: int
    row_stop: int
    input_rows: int
    output_records: int
    embedding_dimension: int
    artifact: str
    sha256: str
    reused: bool


def _canonical_sha256(value: object) -> str:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    import hashlib

    return hashlib.sha256(rendered.encode()).hexdigest()


def _write_parquet_exclusive(frame: pl.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as output:
        frame.write_parquet(output, compression="zstd", statistics=True)
    return destination


def _fixed_size_vectors(name: str, matrix: np.ndarray) -> pl.Series:
    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] <= 0:
        raise ValueError("Cannot serialize an invalid embedding matrix")
    return pl.Series(name, array, dtype=pl.Array(pl.Float32, int(array.shape[1])))


def _build_shard_table(
    pair_ids: list[str],
    direction_vectors: list[tuple[str, np.ndarray, np.ndarray]],
    row_start: int,
) -> pl.DataFrame:
    directions: list[str] = []
    repeated_pair_ids: list[str] = []
    source_rows: list[int] = []
    query_parts: list[np.ndarray] = []
    document_parts: list[np.ndarray] = []
    for direction, query_vectors, document_vectors in direction_vectors:
        if query_vectors.shape != document_vectors.shape or query_vectors.shape[0] != len(pair_ids):
            raise ValueError("Embedding backend returned inconsistent paired matrix shapes")
        directions.extend([direction] * len(pair_ids))
        repeated_pair_ids.extend(pair_ids)
        source_rows.extend(range(row_start, row_start + len(pair_ids)))
        query_parts.append(query_vectors)
        document_parts.append(document_vectors)
    query_matrix = np.concatenate(query_parts, axis=0)
    document_matrix = np.concatenate(document_parts, axis=0)
    return pl.DataFrame(
        [
            pl.Series("pair_id", repeated_pair_ids, dtype=pl.String),
            pl.Series("source_row", source_rows, dtype=pl.Int64),
            pl.Series("direction", directions, dtype=pl.String),
            _fixed_size_vectors("query_embedding", query_matrix),
            _fixed_size_vectors("document_embedding", document_matrix),
        ]
    )


def _copy_exclusive(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_file, destination.open("xb") as output_file:
        shutil.copyfileobj(input_file, output_file)


@register_stage("compute_embeddings")
class ComputeEmbeddingsStage:
    """Encode paired PT and EN responses into text-free Parquet shards."""

    stage_name = "compute_embeddings"

    def __init__(self, cfg: DictConfig, backend: EmbeddingBackend | None = None) -> None:
        self.cfg = cfg
        self._backend = backend

    async def run(self, workspace: RunWorkspace) -> StageResult:
        if workspace.checkpoints is None:
            raise RuntimeError("Embedding stage requires an initialized checkpoint manager")

        input_path, dataset_sha256 = self._validated_input()
        validate_frozen_qwen_config(self.cfg.encoder)
        model_config = OmegaConf.to_container(self.cfg.encoder, resolve=True)
        model_fingerprint = _canonical_sha256(model_config)
        backend = self._backend or create_embedding_backend(self.cfg)
        shard_size = int(self.cfg.runtime.checkpoint_every_rows)
        if shard_size <= 0:
            raise ValueError("runtime.checkpoint_every_rows must be positive")

        embedding_root = workspace.paths.artifacts / "embeddings"
        shard_root = embedding_root / "shards"
        shard_root.mkdir(parents=True, exist_ok=False)
        resume_root = self._validated_resume_root(workspace)
        logger = structlog.get_logger().bind(stage=self.stage_name)
        records: list[ShardRecord] = []
        row_start = 0

        input_frame = pl.read_parquet(input_path, columns=list(_INPUT_COLUMNS), low_memory=True)
        for shard_index, batch in enumerate(input_frame.iter_slices(n_rows=shard_size)):
            row_stop = row_start + batch.height
            resumed = self._resume_shard(
                resume_root=resume_root,
                destination_root=shard_root,
                shard_index=shard_index,
                row_start=row_start,
                row_stop=row_stop,
                dataset_sha256=dataset_sha256,
                model_fingerprint=model_fingerprint,
            )
            if resumed is not None:
                records.append(resumed)
                workspace.checkpoints.write(
                    "embedding_shard_reused",
                    {
                        "shard_index": shard_index,
                        "row_start": row_start,
                        "row_stop": row_stop,
                        "artifact_sha256": resumed.sha256,
                    },
                )
                logger.info(
                    "embedding_shard_reused",
                    shard_index=shard_index,
                    row_start=row_start,
                    row_stop=row_stop,
                )
                row_start = row_stop
                continue

            values = batch.to_dict(as_series=False)
            pair_ids = self._validated_strings(values, "pair_id")
            source_pt = self._validated_strings(values, "source_pt")
            translation_en = self._validated_strings(values, "translation_en")
            column_values = {"source_pt": source_pt, "translation_en": translation_en}
            direction_vectors: list[tuple[str, np.ndarray, np.ndarray]] = []
            for direction_name in sorted(self.cfg.encoder.directions):
                direction = self.cfg.encoder.directions[direction_name]
                query_column = str(direction.query_column)
                document_column = str(direction.document_column)
                if query_column not in _ALLOWED_TEXT_COLUMNS or document_column not in _ALLOWED_TEXT_COLUMNS:
                    raise ValueError("Embedding direction references a non-approved response column")
                instruction = str(direction.query_instruction)
                query_vectors = backend.encode_queries(column_values[query_column], instruction)
                document_vectors = backend.encode_documents(column_values[document_column])
                direction_vectors.append((str(direction_name), query_vectors, document_vectors))

            table = _build_shard_table(pair_ids, direction_vectors, row_start)
            dimension = int(direction_vectors[0][1].shape[1])
            shard_path = shard_root / f"shard_{shard_index:06d}.parquet"
            _write_parquet_exclusive(table, shard_path)
            artifact_hash = sha256_file(shard_path)
            record = ShardRecord(
                shard_index=shard_index,
                row_start=row_start,
                row_stop=row_stop,
                input_rows=batch.height,
                output_records=table.height,
                embedding_dimension=dimension,
                artifact=str(shard_path.relative_to(workspace.paths.root)),
                sha256=artifact_hash,
                reused=False,
            )
            self._write_shard_metadata(
                shard_root=shard_root,
                record=record,
                dataset_sha256=dataset_sha256,
                model_fingerprint=model_fingerprint,
            )
            records.append(record)
            workspace.checkpoints.write(
                "embedding_shard_completed",
                {
                    "shard_index": shard_index,
                    "row_start": row_start,
                    "row_stop": row_stop,
                    "artifact_sha256": artifact_hash,
                },
            )
            logger.info(
                "embedding_shard_completed",
                shard_index=shard_index,
                row_start=row_start,
                row_stop=row_stop,
            )
            row_start = row_stop

        if not records:
            raise ValueError("The paired response input is empty")

        self._validate_record_continuity(records, input_frame.height)
        index_path = self._write_index(embedding_root, records)
        manifest_path = write_json_exclusive(
            embedding_root / "manifest.json",
            {
                "dataset_sha256": dataset_sha256,
                "model": {
                    "repository": str(self.cfg.encoder.repository),
                    "revision": str(self.cfg.encoder.revision),
                    "fingerprint": model_fingerprint,
                    "normalize_embeddings": bool(self.cfg.encoder.normalize_embeddings),
                },
                "backend": str(self.cfg.stage.backend),
                "input_rows": row_start,
                "output_records": sum(record.output_records for record in records),
                "embedding_dimension": records[0].embedding_dimension,
                "shards": [
                    {
                        "shard_index": record.shard_index,
                        "row_start": record.row_start,
                        "row_stop": record.row_stop,
                        "artifact": record.artifact,
                        "sha256": record.sha256,
                        "reused": record.reused,
                    }
                    for record in records
                ],
            },
        )
        reused_count = sum(record.reused for record in records)
        return StageResult(
            metrics={
                "input_rows": row_start,
                "output_records": sum(record.output_records for record in records),
                "embedding_dimension": records[0].embedding_dimension,
                "shards": len(records),
                "shards_reused": reused_count,
            },
            artifacts=(
                str(index_path.relative_to(workspace.paths.root)),
                str(manifest_path.relative_to(workspace.paths.root)),
                str(shard_root.relative_to(workspace.paths.root)),
            ),
            details={
                "encoder_repository": str(self.cfg.encoder.repository),
                "encoder_revision": str(self.cfg.encoder.revision),
                "dataset_sha256": dataset_sha256,
            },
        )

    def _validated_input(self) -> tuple[Path, str]:
        input_path = Path(str(self.cfg.data.curated_dir)) / "paired_responses.parquet"
        sanitized_manifest = Path(str(self.cfg.data.sanitized_manifest))
        if not input_path.is_file() or not sanitized_manifest.is_file():
            raise FileNotFoundError("Curated responses or their sanitized manifest are missing")
        manifest = json.loads(sanitized_manifest.read_text(encoding="utf-8"))
        expected_hash = str(manifest.get("output_sha256", {}).get(input_path.name, ""))
        actual_hash = sha256_file(input_path)
        if not expected_hash or expected_hash != actual_hash:
            raise ValueError("Curated response checksum does not match its sanitized manifest")
        metadata_rows = int(pl.scan_parquet(input_path).select(pl.len()).collect().item())
        expected_rows = int(self.cfg.data.expected.paired_responses)
        if metadata_rows != expected_rows:
            raise ValueError("Curated response row count does not match the data contract")
        return input_path, actual_hash

    def _validated_resume_root(self, workspace: RunWorkspace) -> Path | None:
        configured = self.cfg.stage.get("resume_from_run")
        if configured in (None, "", "null"):
            return None
        candidate = Path(str(configured)).resolve()
        allowed_root = (Path(str(self.cfg.project.artifact_root)).resolve() / "runs").resolve()
        if candidate == workspace.paths.root or not candidate.is_relative_to(allowed_root):
            raise ValueError("Embedding resume source must be another run inside the artifact root")
        source = candidate / "artifacts" / "embeddings" / "shards"
        if not source.is_dir():
            raise ValueError("Embedding resume source has no shard artifacts")
        return source

    def _resume_shard(
        self,
        *,
        resume_root: Path | None,
        destination_root: Path,
        shard_index: int,
        row_start: int,
        row_stop: int,
        dataset_sha256: str,
        model_fingerprint: str,
    ) -> ShardRecord | None:
        if resume_root is None:
            return None
        stem = f"shard_{shard_index:06d}"
        source_parquet = resume_root / f"{stem}.parquet"
        source_metadata = resume_root / f"{stem}.json"
        if not source_parquet.is_file() or not source_metadata.is_file():
            return None
        metadata = json.loads(source_metadata.read_text(encoding="utf-8"))
        required = {
            "shard_index": shard_index,
            "row_start": row_start,
            "row_stop": row_stop,
            "dataset_sha256": dataset_sha256,
            "model_fingerprint": model_fingerprint,
        }
        if any(metadata.get(key) != value for key, value in required.items()):
            return None
        expected_hash = str(metadata.get("sha256", ""))
        if not expected_hash or sha256_file(source_parquet) != expected_hash:
            return None
        schema = pl.read_parquet_schema(source_parquet)
        schema_names = set(schema)
        expected_schema = {
            "pair_id",
            "source_row",
            "direction",
            "query_embedding",
            "document_embedding",
        }
        if schema_names != expected_schema:
            return None
        query_type = schema["query_embedding"]
        document_type = schema["document_embedding"]
        if not isinstance(query_type, pl.Array) or not isinstance(document_type, pl.Array):
            return None
        expected_input_rows = row_stop - row_start
        expected_output_records = expected_input_rows * len(self.cfg.encoder.directions)
        observed_output_records = int(
            pl.scan_parquet(source_parquet).select(pl.len()).collect().item()
        )
        if (
            int(metadata.get("input_rows", -1)) != expected_input_rows
            or int(metadata.get("output_records", -1)) != expected_output_records
            or observed_output_records != expected_output_records
            or query_type != document_type
            or int(metadata.get("embedding_dimension", -1)) != int(query_type.shape[0])
        ):
            return None

        destination_parquet = destination_root / source_parquet.name
        _copy_exclusive(source_parquet, destination_parquet)
        record = ShardRecord(
            shard_index=shard_index,
            row_start=row_start,
            row_stop=row_stop,
            input_rows=int(metadata["input_rows"]),
            output_records=int(metadata["output_records"]),
            embedding_dimension=int(metadata["embedding_dimension"]),
            artifact=str(destination_parquet.relative_to(destination_root.parents[2])),
            sha256=expected_hash,
            reused=True,
        )
        self._write_shard_metadata(
            shard_root=destination_root,
            record=record,
            dataset_sha256=dataset_sha256,
            model_fingerprint=model_fingerprint,
        )
        return record

    @staticmethod
    def _validated_strings(values: dict[str, Any], column: str) -> list[str]:
        column_values = cast(list[object], values[column])
        if any(not isinstance(value, str) or not value for value in column_values):
            raise ValueError(f"Input column {column} contains a missing or invalid value")
        return cast(list[str], column_values)

    @staticmethod
    def _write_shard_metadata(
        *,
        shard_root: Path,
        record: ShardRecord,
        dataset_sha256: str,
        model_fingerprint: str,
    ) -> None:
        write_json_exclusive(
            shard_root / f"shard_{record.shard_index:06d}.json",
            {
                "shard_index": record.shard_index,
                "row_start": record.row_start,
                "row_stop": record.row_stop,
                "input_rows": record.input_rows,
                "output_records": record.output_records,
                "embedding_dimension": record.embedding_dimension,
                "dataset_sha256": dataset_sha256,
                "model_fingerprint": model_fingerprint,
                "sha256": record.sha256,
            },
        )

    @staticmethod
    def _validate_record_continuity(records: list[ShardRecord], expected_rows: int) -> None:
        cursor = 0
        dimensions = {record.embedding_dimension for record in records}
        for expected_index, record in enumerate(records):
            if record.shard_index != expected_index or record.row_start != cursor:
                raise ValueError("Embedding shard sequence is incomplete or discontinuous")
            if record.row_stop <= record.row_start or record.input_rows != record.row_stop - record.row_start:
                raise ValueError("Embedding shard row bounds are invalid")
            cursor = record.row_stop
        if cursor != expected_rows or len(dimensions) != 1:
            raise ValueError("Embedding shards do not cover the complete input with one dimension")

    @staticmethod
    def _write_index(embedding_root: Path, records: list[ShardRecord]) -> Path:
        index = pl.DataFrame(
            {
                "shard_index": [record.shard_index for record in records],
                "row_start": [record.row_start for record in records],
                "row_stop": [record.row_stop for record in records],
                "input_rows": [record.input_rows for record in records],
                "output_records": [record.output_records for record in records],
                "embedding_dimension": [record.embedding_dimension for record in records],
                "artifact": [record.artifact for record in records],
                "sha256": [record.sha256 for record in records],
                "reused": [record.reused for record in records],
            }
        )
        return _write_parquet_exclusive(index, embedding_root / "index.parquet")
