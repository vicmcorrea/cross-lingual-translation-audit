"""Checkpointed bilingual EmoAtlas feature extraction."""

import asyncio
import json
import os
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import cast

import polars as pl
import structlog
from omegaconf import DictConfig

from translation_audit.emotion import EmoAtlasWorkerBackend, EmotionBackend, build_output_record
from translation_audit.emotion.backend import EmotionWorkerError
from translation_audit.emotion.features import IDENTIFIER_COLUMNS, TEXT_COLUMNS
from translation_audit.registry import register_stage
from translation_audit.runtime.files import sha256_file, write_json_exclusive
from translation_audit.runtime.run_workspace import RunWorkspace
from translation_audit.types import StageResult

BackendFactory = Callable[[], EmotionBackend]
_INPUT_COLUMNS = (*IDENTIFIER_COLUMNS, *TEXT_COLUMNS)


def _as_int(value: object) -> int:
    if not isinstance(value, int | float | str):
        raise ValueError("Emotion checkpoint contains a non-numeric field")
    return int(value)


def _write_parquet_exclusive(path: Path, frame: pl.DataFrame) -> Path:
    """Atomically create a Parquet artifact without overwriting another attempt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        frame.write_parquet(temporary_path, compression="zstd")
        with temporary_path.open("rb") as stream:
            os.fsync(stream.fileno())
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def _validate_input_schema(input_path: Path) -> None:
    missing = set(_INPUT_COLUMNS).difference(pl.scan_parquet(input_path).collect_schema().names())
    if missing:
        raise ValueError("Curated cohort does not satisfy the bilingual emotion input schema")


def _iter_input_shards(input_path: Path, shard_size: int) -> Iterator[tuple[int, int, pl.DataFrame]]:
    _validate_input_schema(input_path)
    frame = pl.read_parquet(input_path, columns=list(_INPUT_COLUMNS))
    for shard_index, row_start in enumerate(range(0, frame.height, shard_size)):
        yield shard_index, row_start, frame.slice(row_start, shard_size)


def _completed_shards(workspace: RunWorkspace) -> dict[int, tuple[Path, int]]:
    """Recover only checksum-validated shard artifacts recorded by this run."""
    completed: dict[int, tuple[Path, int]] = {}
    for checkpoint_path in sorted(workspace.paths.checkpoints.glob("*.json")):
        document = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if document.get("event") != "emotion_shard_completed":
            continue
        payload = cast(dict[str, object], document["payload"])
        shard_index = _as_int(payload["shard_index"])
        artifact = (workspace.paths.root / str(payload["artifact"])).resolve()
        if not artifact.is_relative_to(workspace.paths.root):
            raise ValueError("Emotion shard checkpoint points outside its run")
        if not artifact.is_file() or sha256_file(artifact) != str(payload["sha256"]):
            raise ValueError("Emotion shard checkpoint does not match its artifact")
        if set(TEXT_COLUMNS).intersection(pl.scan_parquet(artifact).collect_schema().names()):
            raise ValueError("Emotion shard artifact contains response text")
        completed[shard_index] = (artifact, _as_int(payload["row_count"]))
    return completed


@register_stage("compute_emotion_features")
class ComputeEmotionFeaturesStage:
    """Compute paired emotion, valence, and network features without persisting text."""

    stage_name = "compute_emotion_features"

    def __init__(self, cfg: DictConfig, backend_factory: BackendFactory | None = None) -> None:
        self.cfg = cfg
        self._backend_factory = backend_factory or self._configured_backend

    def _configured_backend(self) -> EmotionBackend:
        worker = self.cfg.emotion.worker
        return EmoAtlasWorkerBackend(
            python_candidates=[str(item) for item in worker.python_candidates],
            module=str(worker.module),
            portuguese_model=str(worker.portuguese_model),
            english_model=str(worker.english_model),
            max_distance=int(self.cfg.emotion.features.max_distance),
        )

    async def run(self, workspace: RunWorkspace) -> StageResult:
        """Run blocking NLP and Parquet work away from the orchestration loop."""
        return await asyncio.to_thread(self._run_sync, workspace)

    def _run_sync(self, workspace: RunWorkspace) -> StageResult:
        input_path = Path(str(self.cfg.emotion.input_path)).resolve()
        if not input_path.is_file():
            raise FileNotFoundError("Configured emotion input Parquet is unavailable")
        if workspace.checkpoints is None:
            raise RuntimeError("Checkpoint manager was not initialized")

        shard_size = int(self.cfg.runtime.checkpoint_every_rows)
        if shard_size <= 0:
            raise ValueError("Emotion shard size must be positive")
        max_retries = int(self.cfg.emotion.worker.max_shard_retries)
        if max_retries < 0:
            raise ValueError("Emotion worker retry count cannot be negative")

        artifact_directory = workspace.paths.artifacts / "emotion_features"
        shard_directory = artifact_directory / "shards"
        shard_directory.mkdir(parents=True, exist_ok=True)
        completed = _completed_shards(workspace)
        logger = structlog.get_logger().bind(stage=self.stage_name)

        backend = self._backend_factory()
        backend.__enter__()
        processed_rows = 0
        shard_count = 0
        try:
            for shard_index, row_start, batch in _iter_input_shards(input_path, shard_size):
                shard_count += 1
                existing = completed.get(shard_index)
                if existing is not None:
                    processed_rows += existing[1]
                    logger.info("emotion_shard_resumed", shard_index=shard_index, row_count=existing[1])
                    continue

                input_rows = cast(list[dict[str, object]], batch.to_dicts())
                worker_records: list[dict[str, object]] | None = None
                for attempt in range(max_retries + 1):
                    try:
                        worker_records = backend.analyze(input_rows)
                        break
                    except EmotionWorkerError as error:
                        backend.__exit__(type(error), error, error.__traceback__)
                        if attempt >= max_retries:
                            raise
                        workspace.checkpoints.write(
                            "emotion_worker_restarted",
                            {"shard_index": shard_index, "attempt": attempt + 1},
                        )
                        logger.warning(
                            "emotion_worker_restarted",
                            shard_index=shard_index,
                            attempt=attempt + 1,
                        )
                        backend = self._backend_factory()
                        backend.__enter__()
                if worker_records is None:
                    raise RuntimeError("Emotion worker produced no records")

                output_rows = [
                    build_output_record(identifiers, worker_record)
                    for identifiers, worker_record in zip(input_rows, worker_records, strict=True)
                ]
                frame = pl.from_dicts(output_rows)
                if set(TEXT_COLUMNS).intersection(frame.columns):
                    raise ValueError("Emotion feature output contains response text")
                shard_path = shard_directory / f"part-{shard_index:06d}.parquet"
                _write_parquet_exclusive(shard_path, frame)
                workspace.checkpoints.write(
                    "emotion_shard_completed",
                    {
                        "shard_index": shard_index,
                        "row_start": row_start,
                        "row_stop": row_start + batch.height,
                        "row_count": batch.height,
                        "artifact": str(shard_path.relative_to(workspace.paths.root)),
                        "sha256": sha256_file(shard_path),
                    },
                )
                processed_rows += batch.height
                logger.info(
                    "emotion_shard_completed",
                    shard_index=shard_index,
                    row_count=batch.height,
                )
        finally:
            backend.__exit__(None, None, None)

        completed = _completed_shards(workspace)
        if len(completed) != shard_count:
            raise RuntimeError("Emotion stage did not produce every expected shard")
        shard_paths = [completed[index][0] for index in range(shard_count)]
        frames = [pl.read_parquet(path) for path in shard_paths]
        combined = pl.concat(frames, how="vertical") if frames else pl.DataFrame()
        if combined.height != processed_rows:
            raise RuntimeError("Emotion shard row counts are inconsistent")

        output_path = artifact_directory / "emotion_features.parquet"
        _write_parquet_exclusive(output_path, combined)
        shard_manifest = [
            {
                "shard_index": index,
                "row_count": completed[index][1],
                "artifact": str(completed[index][0].relative_to(workspace.paths.root)),
                "sha256": sha256_file(completed[index][0]),
            }
            for index in range(shard_count)
        ]
        manifest_path = write_json_exclusive(
            artifact_directory / "manifest.json",
            {
                "stage": self.stage_name,
                "backend": str(self.cfg.emotion.name),
                "repository": str(self.cfg.emotion.repository),
                "revision": str(self.cfg.emotion.revision),
                "input_sha256": sha256_file(input_path),
                "output_sha256": sha256_file(output_path),
                "row_count": combined.height,
                "shard_size": shard_size,
                "shards": shard_manifest,
                "languages": [str(value) for value in self.cfg.emotion.languages],
                "response_text_columns_persisted": False,
            },
        )

        emotion_values = [float(value) for value in combined["emotion_jensen_shannon_distance"].to_list()]
        valence_values = [float(value) for value in combined["valence_jensen_shannon_distance"].to_list()]
        row_denominator = max(combined.height, 1)
        return StageResult(
            metrics={
                "processed_rows": combined.height,
                "shards": shard_count,
                "mean_emotion_jensen_shannon_distance": sum(emotion_values) / row_denominator,
                "mean_valence_jensen_shannon_distance": sum(valence_values) / row_denominator,
            },
            artifacts=(
                str(output_path.relative_to(workspace.paths.root)),
                str(manifest_path.relative_to(workspace.paths.root)),
            ),
            details={
                "feature_family": "bilingual_emoatlas",
                "response_text_persisted": False,
            },
        )
