"""Reference-free COMETKiwi quality estimation for PT-BR to English pairs."""

import hashlib
import json
import math
import os
import re
import select
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from types import TracebackType
from typing import Protocol, Self, TextIO, cast

import polars as pl
import structlog
from omegaconf import DictConfig

from translation_audit.registry import register_stage
from translation_audit.runtime.files import sha256_file, write_json_exclusive
from translation_audit.runtime.run_workspace import RunWorkspace
from translation_audit.types import StageResult

_FULL_COMMIT = re.compile(r"[0-9a-f]{40}")
_INPUT_COLUMNS = ("pair_id", "source_pt", "translation_en")
_OUTPUT_COLUMNS = ("pair_id", "cometkiwi_score")


def _canonical_sha256(value: object) -> str:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(rendered.encode()).hexdigest()


class QualityEstimator(Protocol):
    """Score translation pairs without retaining their text."""

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    def score(self, sources: Sequence[str], translations: Sequence[str]) -> list[float]: ...


class SyntheticQualityEstimator:
    """Deterministic, download-free estimator used only by unit tests."""

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    def score(self, sources: Sequence[str], translations: Sequence[str]) -> list[float]:
        if len(sources) != len(translations):
            raise ValueError("Synthetic estimator received misaligned inputs")
        scores: list[float] = []
        for source, translation in zip(sources, translations, strict=True):
            digest = hashlib.sha256(f"{source}\u0000{translation}".encode()).digest()
            scores.append(int.from_bytes(digest[:8], "big") / float((1 << 64) - 1))
        return scores


class CometWorkerClient:  # pragma: no cover - exercised by the pinned Pod integration smoke
    """Persistent, text-over-stdin client for the isolated Python 3.11 COMET worker."""

    def __init__(self, cfg: DictConfig) -> None:
        self.python_executable = Path(str(cfg.qe.worker.python_executable)).expanduser().absolute()
        self.worker_script = Path(str(cfg.qe.worker.script)).resolve()
        self.repository = str(cfg.qe.repository)
        self.revision = str(cfg.qe.revision)
        self.batch_size = int(cfg.qe.batch_size)
        self.precision = str(cfg.qe.precision)
        self.device = str(cfg.runtime.device)
        self._process: subprocess.Popen[str] | None = None
        self._stderr_stream: TextIO | None = None
        self._request_sequence = 0

    def _diagnostic(self, phase: str, sensitive_values: Sequence[str] = ()) -> str:
        process = self._process
        stderr_stream = self._stderr_stream
        exit_code = None if process is None else process.poll()
        diagnostic = "no worker diagnostic was emitted"
        if stderr_stream is not None:
            stderr_stream.flush()
            stderr_stream.seek(0)
            diagnostic = stderr_stream.read()[-8192:].strip() or diagnostic
        redactions = [*sensitive_values]
        redactions.extend(
            value
            for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
            if (value := os.environ.get(name))
        )
        for value in sorted(set(redactions), key=len, reverse=True):
            if value:
                diagnostic = diagnostic.replace(value, "[REDACTED]")
        diagnostic = re.sub(
            r"(?i)((?:authorization|token)\s*[:=]\s*)\S+",
            r"\1[REDACTED]",
            diagnostic,
        )
        return f"COMET worker failed during {phase} (exit_code={exit_code}). {diagnostic}"

    def __enter__(self) -> Self:
        if not self.python_executable.is_file():
            raise FileNotFoundError("The isolated COMET Python executable is unavailable")
        if not self.worker_script.is_file():
            raise FileNotFoundError("The isolated COMET worker script is unavailable")
        command = [
            str(self.python_executable),
            str(self.worker_script),
            "--repository",
            self.repository,
            "--revision",
            self.revision,
            "--batch-size",
            str(self.batch_size),
            "--device",
            self.device,
            "--precision",
            self.precision,
        ]
        self._stderr_stream = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_stream,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=os.environ.copy(),
        )
        process = self._process
        if process.stdout is None:
            raise RuntimeError("COMET worker stdout was not initialized")
        readable, _, _ = select.select([process.stdout], [], [], 1800)
        if not readable:
            diagnostic = self._diagnostic("startup timeout")
            self.__exit__(None, None, None)
            raise RuntimeError(diagnostic)
        ready_line = process.stdout.readline()
        try:
            ready_value: object = json.loads(ready_line)
        except json.JSONDecodeError as error:
            diagnostic = self._diagnostic("startup")
            self.__exit__(None, None, None)
            raise RuntimeError(diagnostic) from error
        ready = cast(dict[str, object], ready_value) if isinstance(ready_value, dict) else {}
        if ready.get("status") != "ready":
            diagnostic = self._diagnostic("startup")
            self.__exit__(None, None, None)
            raise RuntimeError(diagnostic)
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exception_type, exception, traceback
        process = self._process
        self._process = None
        if process is None:
            return
        if process.stdin is not None:
            with suppress(BrokenPipeError):
                process.stdin.close()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        if self._stderr_stream is not None:
            self._stderr_stream.close()
            self._stderr_stream = None

    def score(self, sources: Sequence[str], translations: Sequence[str]) -> list[float]:
        if len(sources) != len(translations):
            raise ValueError("COMET worker received misaligned inputs")
        process = self._process
        if process is None or process.stdin is None or process.stdout is None:
            raise RuntimeError("COMET worker is not running")
        if process.poll() is not None:
            raise RuntimeError(self._diagnostic("scoring", [*sources, *translations]))

        self._request_sequence += 1
        request = {
            "request_id": self._request_sequence,
            "samples": [
                {"src": source, "mt": translation}
                for source, translation in zip(sources, translations, strict=True)
            ],
        }
        try:
            process.stdin.write(json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n")
            process.stdin.flush()
            response_line = process.stdout.readline()
        except (BrokenPipeError, OSError) as error:
            raise RuntimeError(
                self._diagnostic("scoring communication", [*sources, *translations])
            ) from error
        if not response_line:
            raise RuntimeError(self._diagnostic("scoring response", [*sources, *translations]))
        try:
            response_value: object = json.loads(response_line)
        except json.JSONDecodeError as error:
            raise RuntimeError("COMET worker returned an invalid protocol response") from error
        if not isinstance(response_value, dict):
            raise RuntimeError("COMET worker returned an invalid protocol response")
        response = cast(dict[str, object], response_value)
        if response.get("request_id") != self._request_sequence:
            raise RuntimeError("COMET worker response did not match its request")
        raw_scores = response.get("scores")
        if not isinstance(raw_scores, list):
            raise RuntimeError("COMET worker returned an invalid score count")
        score_values = cast(list[object], raw_scores)
        if len(score_values) != len(sources):
            raise RuntimeError("COMET worker returned an invalid score count")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in score_values):
            raise RuntimeError("COMET worker returned a non-numeric score")
        scores = [float(value) for value in cast(list[int | float], score_values)]
        if any(not math.isfinite(score) for score in scores):
            raise RuntimeError("COMET worker returned a non-finite score")
        return scores


def _write_parquet_exclusive(path: Path, frame: pl.DataFrame) -> Path:
    """Atomically publish a Parquet artifact without replacing prior output."""
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


def _copy_exclusive(source: Path, destination: Path) -> None:
    """Copy one verified shard without permitting destination replacement."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_stream, destination.open("xb") as output_stream:
        shutil.copyfileobj(input_stream, output_stream)
        output_stream.flush()
        os.fsync(output_stream.fileno())


def _model_fingerprint(cfg: DictConfig) -> str:
    """Fingerprint every model option that can affect COMET scores."""
    return _canonical_sha256(
        {
            "repository": str(cfg.qe.repository),
            "revision": str(cfg.qe.revision),
            "batch_size": int(cfg.qe.batch_size),
            "precision": str(cfg.qe.precision),
            "inputs": [str(value) for value in cfg.qe.inputs],
            "reference_free": bool(cfg.qe.reference_free),
        }
    )


def _write_shard_metadata(
    metadata_path: Path,
    *,
    shard_index: int,
    row_start: int,
    row_stop: int,
    input_sha256: str,
    model_fingerprint: str,
    artifact_sha256: str,
) -> None:
    write_json_exclusive(
        metadata_path,
        {
            "shard_index": shard_index,
            "row_start": row_start,
            "row_stop": row_stop,
            "rows": row_stop - row_start,
            "input_sha256": input_sha256,
            "model_fingerprint": model_fingerprint,
            "artifact_sha256": artifact_sha256,
        },
    )


def _validated_shard(
    shard_path: Path,
    metadata_path: Path,
    *,
    expected_pair_ids: Sequence[str],
    shard_index: int,
    row_start: int,
    row_stop: int,
    input_sha256: str,
    model_fingerprint: str,
) -> tuple[pl.DataFrame, str] | None:
    """Return a shard only when its data and non-sensitive provenance both match."""
    if not shard_path.is_file() or not metadata_path.is_file():
        return None
    metadata_value: object = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata_value, dict):
        return None
    metadata = cast(dict[str, object], metadata_value)
    required: dict[str, object] = {
        "shard_index": shard_index,
        "row_start": row_start,
        "row_stop": row_stop,
        "rows": row_stop - row_start,
        "input_sha256": input_sha256,
        "model_fingerprint": model_fingerprint,
    }
    if any(metadata.get(key) != value for key, value in required.items()):
        return None
    artifact_hash = metadata.get("artifact_sha256")
    if not isinstance(artifact_hash, str) or sha256_file(shard_path) != artifact_hash:
        return None
    try:
        frame = pl.read_parquet(shard_path)
        _validate_score_frame(frame, expected_pair_ids)
    except (OSError, ValueError, pl.exceptions.PolarsError):
        return None
    return frame, artifact_hash


def _validate_config(cfg: DictConfig) -> None:
    """Reject mutable, reference-based, or incorrectly wired QE configurations."""
    if not bool(cfg.qe.frozen):
        raise ValueError("Translation quality model must be frozen")
    revision = str(cfg.qe.revision)
    if _FULL_COMMIT.fullmatch(revision) is None:
        raise ValueError("Translation quality model revision must be a full 40-character commit")
    if not bool(cfg.qe.reference_free):
        raise ValueError("COMETKiwi must remain reference-free")
    if [str(value) for value in cfg.qe.inputs] != ["source_pt", "translation_en"]:
        raise ValueError("COMETKiwi inputs must be source_pt followed by translation_en")
    if int(cfg.qe.batch_size) <= 0:
        raise ValueError("qe.batch_size must be positive")
    if str(cfg.qe.precision) not in {"float16", "bfloat16", "float32"}:
        raise ValueError("Unsupported COMET precision")


def _load_input(cfg: DictConfig) -> tuple[pl.DataFrame, Path, str]:
    """Load only the required columns and verify the curated file fingerprint."""
    input_path = Path(str(cfg.data.curated_dir)).resolve() / str(cfg.stage.input_filename)
    if not input_path.is_file():
        raise FileNotFoundError("Curated translation-pair input is unavailable")
    manifest_path = Path(str(cfg.data.sanitized_manifest)).resolve()
    manifest_value: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest_value, dict):
        raise ValueError("Curated manifest is not a JSON object")
    manifest = cast(dict[str, object], manifest_value)
    raw_hashes = manifest.get("output_sha256")
    if not isinstance(raw_hashes, dict):
        raise ValueError("Curated manifest has no output fingerprints")
    expected_hash = cast(dict[str, object], raw_hashes).get(input_path.name)
    input_hash = sha256_file(input_path)
    if not isinstance(expected_hash, str) or input_hash != expected_hash:
        raise ValueError("Curated translation-pair fingerprint does not match its manifest")

    schema = pl.read_parquet_schema(input_path)
    missing = set(_INPUT_COLUMNS).difference(schema)
    if missing:
        raise ValueError("Curated translation-pair input is missing required columns")
    pairs = pl.read_parquet(input_path, columns=list(_INPUT_COLUMNS))
    if pairs.is_empty():
        raise ValueError("Curated translation-pair input is empty")
    if pairs.null_count().select(pl.sum_horizontal(pl.all())).item() != 0:
        raise ValueError("Curated translation-pair input contains null values")
    if pairs["pair_id"].n_unique() != pairs.height:
        raise ValueError("Curated translation-pair identifiers are not unique")
    if pairs.height != int(cfg.data.expected.paired_responses):
        raise ValueError("Curated translation-pair row count does not match the data contract")
    return pairs, input_path, input_hash


def _validate_score_frame(frame: pl.DataFrame, expected_pair_ids: Sequence[str]) -> None:
    """Validate an existing or newly generated text-free score shard."""
    if frame.columns != list(_OUTPUT_COLUMNS):
        raise ValueError("Translation-quality shard has an invalid schema")
    pair_ids = cast(list[str], frame["pair_id"].to_list())
    if pair_ids != list(expected_pair_ids):
        raise ValueError("Translation-quality shard does not match its input boundary")
    scores = cast(list[float], frame["cometkiwi_score"].to_list())
    if any(not math.isfinite(float(score)) for score in scores):
        raise ValueError("Translation-quality shard contains a non-finite score")


def _validated_resume_root(cfg: DictConfig, workspace: RunWorkspace) -> Path | None:
    """Resolve an earlier run while preserving run isolation and path confinement."""
    configured = cfg.stage.get("resume_from_run")
    if configured in (None, "", "null"):
        return None
    candidate = Path(str(configured)).resolve()
    allowed_root = (Path(str(cfg.project.artifact_root)).resolve() / "runs").resolve()
    if candidate == workspace.paths.root or not candidate.is_relative_to(allowed_root):
        raise ValueError("COMET resume source must be another run inside the artifact root")
    shard_root = candidate / "artifacts" / str(cfg.stage.output_directory) / "shards"
    if not shard_root.is_dir():
        raise ValueError("COMET resume source has no shard artifacts")
    return shard_root


@register_stage("estimate_translation_quality")
class EstimateTranslationQualityStage:
    """Score deterministic PT-BR/English shards with reference-free COMETKiwi."""

    stage_name = "estimate_translation_quality"

    def __init__(self, cfg: DictConfig, estimator: QualityEstimator | None = None) -> None:
        self.cfg = cfg
        self._estimator = estimator

    async def run(self, workspace: RunWorkspace) -> StageResult:
        _validate_config(self.cfg)
        if workspace.checkpoints is None:
            raise RuntimeError("Checkpoint manager was not initialized")
        pairs, input_path, input_hash = _load_input(self.cfg)
        model_fingerprint = _model_fingerprint(self.cfg)
        shard_size = int(self.cfg.runtime.checkpoint_every_rows)
        shard_count = math.ceil(pairs.height / shard_size)
        output_root = workspace.paths.artifacts / str(self.cfg.stage.output_directory)
        shard_root = output_root / "shards"
        final_path = output_root / str(self.cfg.stage.output_filename)
        resume_root = _validated_resume_root(self.cfg, workspace)
        logger = structlog.get_logger().bind(stage=self.stage_name)

        workspace.checkpoints.write(
            "qe_input_validated",
            {
                "input_sha256": input_hash,
                "rows": pairs.height,
                "shards": shard_count,
                "model_repository": str(self.cfg.qe.repository),
                "model_revision": str(self.cfg.qe.revision),
                "model_fingerprint": model_fingerprint,
            },
        )
        logger.info("qe_input_validated", rows=pairs.height, shards=shard_count)

        pending: list[tuple[int, int, int, Path, Path]] = []
        shard_paths: list[Path] = []
        shard_records: list[dict[str, object]] = []
        for shard_index, row_start in enumerate(range(0, pairs.height, shard_size)):
            row_stop = min(row_start + shard_size, pairs.height)
            input_shard = pairs.slice(row_start, row_stop - row_start)
            shard_path = shard_root / f"part-{shard_index:06d}.parquet"
            metadata_path = shard_path.with_suffix(".json")
            shard_paths.append(shard_path)
            expected_ids = cast(list[str], input_shard["pair_id"].to_list())
            if shard_path.exists() or metadata_path.exists():
                verified = _validated_shard(
                    shard_path,
                    metadata_path,
                    expected_pair_ids=expected_ids,
                    shard_index=shard_index,
                    row_start=row_start,
                    row_stop=row_stop,
                    input_sha256=input_hash,
                    model_fingerprint=model_fingerprint,
                )
                if verified is None:
                    raise ValueError("Existing run-local COMET shard failed resume validation")
                existing, artifact_hash = verified
                workspace.checkpoints.write(
                    "qe_shard_reused",
                    {
                        "shard_index": shard_index,
                        "row_start": row_start,
                        "row_stop": row_stop,
                        "rows": existing.height,
                        "artifact_sha256": artifact_hash,
                    },
                )
                logger.info("qe_shard_reused", shard_index=shard_index, rows=existing.height)
                shard_records.append(
                    {
                        "shard_index": shard_index,
                        "row_start": row_start,
                        "row_stop": row_stop,
                        "rows": existing.height,
                        "artifact": str(shard_path.relative_to(workspace.paths.root)),
                        "artifact_sha256": artifact_hash,
                        "reused": True,
                    }
                )
                continue

            if resume_root is not None:
                source_path = resume_root / shard_path.name
                source_metadata = resume_root / metadata_path.name
                resumed = _validated_shard(
                    source_path,
                    source_metadata,
                    expected_pair_ids=expected_ids,
                    shard_index=shard_index,
                    row_start=row_start,
                    row_stop=row_stop,
                    input_sha256=input_hash,
                    model_fingerprint=model_fingerprint,
                )
                if resumed is not None:
                    resumed_frame, artifact_hash = resumed
                    _copy_exclusive(source_path, shard_path)
                    _write_shard_metadata(
                        metadata_path,
                        shard_index=shard_index,
                        row_start=row_start,
                        row_stop=row_stop,
                        input_sha256=input_hash,
                        model_fingerprint=model_fingerprint,
                        artifact_sha256=artifact_hash,
                    )
                    workspace.checkpoints.write(
                        "qe_shard_reused",
                        {
                            "shard_index": shard_index,
                            "row_start": row_start,
                            "row_stop": row_stop,
                            "rows": resumed_frame.height,
                            "artifact_sha256": artifact_hash,
                        },
                    )
                    logger.info(
                        "qe_shard_reused", shard_index=shard_index, rows=resumed_frame.height
                    )
                    shard_records.append(
                        {
                            "shard_index": shard_index,
                            "row_start": row_start,
                            "row_stop": row_stop,
                            "rows": resumed_frame.height,
                            "artifact": str(shard_path.relative_to(workspace.paths.root)),
                            "artifact_sha256": artifact_hash,
                            "reused": True,
                        }
                    )
                    continue
            pending.append((shard_index, row_start, row_stop, shard_path, metadata_path))

        estimator = self._estimator or CometWorkerClient(self.cfg)
        if pending:
            with estimator:
                for shard_index, row_start, row_stop, shard_path, metadata_path in pending:
                    input_shard = pairs.slice(row_start, row_stop - row_start)
                    sources = cast(list[str], input_shard["source_pt"].to_list())
                    translations = cast(list[str], input_shard["translation_en"].to_list())
                    scores = estimator.score(sources, translations)
                    if len(scores) != input_shard.height or any(
                        not math.isfinite(score) for score in scores
                    ):
                        raise RuntimeError("Translation-quality estimator returned invalid scores")
                    result = pl.DataFrame(
                        {
                            "pair_id": input_shard["pair_id"],
                            "cometkiwi_score": pl.Series(scores, dtype=pl.Float64),
                        }
                    )
                    expected_ids = cast(list[str], input_shard["pair_id"].to_list())
                    _validate_score_frame(result, expected_ids)
                    _write_parquet_exclusive(shard_path, result)
                    artifact_hash = sha256_file(shard_path)
                    _write_shard_metadata(
                        metadata_path,
                        shard_index=shard_index,
                        row_start=row_start,
                        row_stop=row_stop,
                        input_sha256=input_hash,
                        model_fingerprint=model_fingerprint,
                        artifact_sha256=artifact_hash,
                    )
                    workspace.checkpoints.write(
                        "qe_shard_completed",
                        {
                            "shard_index": shard_index,
                            "row_start": row_start,
                            "row_stop": row_stop,
                            "rows": result.height,
                            "artifact_sha256": artifact_hash,
                        },
                    )
                    logger.info("qe_shard_completed", shard_index=shard_index, rows=result.height)
                    shard_records.append(
                        {
                            "shard_index": shard_index,
                            "row_start": row_start,
                            "row_stop": row_stop,
                            "rows": result.height,
                            "artifact": str(shard_path.relative_to(workspace.paths.root)),
                            "artifact_sha256": artifact_hash,
                            "reused": False,
                        }
                    )

        shard_records.sort(key=lambda record: int(cast(int, record["shard_index"])))

        shard_frames = [pl.read_parquet(path) for path in shard_paths]
        combined = pl.concat(shard_frames, how="vertical")
        expected_all_ids = cast(list[str], pairs["pair_id"].to_list())
        _validate_score_frame(combined, expected_all_ids)
        if final_path.is_file():
            existing_final = pl.read_parquet(final_path)
            _validate_score_frame(existing_final, expected_all_ids)
            if not existing_final.equals(combined):
                raise ValueError("Existing translation-quality output differs from its shards")
        else:
            _write_parquet_exclusive(final_path, combined)

        scores = cast(list[float], combined["cometkiwi_score"].to_list())
        output_hash = sha256_file(final_path)
        metrics: dict[str, float | int] = {
            "rows_scored": combined.height,
            "shards": shard_count,
            "shards_reused": sum(bool(record["reused"]) for record in shard_records),
            "score_mean": float(sum(scores) / len(scores)),
            "score_min": float(min(scores)),
            "score_max": float(max(scores)),
        }
        manifest_path = output_root / "manifest.json"
        manifest = {
            "stage": self.stage_name,
            "reference_free": True,
            "source_language": "pt-BR",
            "target_language": "en",
            "input_file": input_path.name,
            "input_sha256": input_hash,
            "output_file": final_path.name,
            "output_sha256": output_hash,
            "rows": combined.height,
            "shards": shard_count,
            "model_repository": str(self.cfg.qe.repository),
            "model_revision": str(self.cfg.qe.revision),
            "model_fingerprint": model_fingerprint,
            "precision": str(self.cfg.qe.precision),
            "shard_records": shard_records,
        }
        if manifest_path.is_file():
            if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
                raise ValueError("Existing translation-quality manifest is inconsistent")
        else:
            write_json_exclusive(manifest_path, manifest)
        workspace.checkpoints.write(
            "qe_output_completed",
            {
                "rows": combined.height,
                "shards": shard_count,
                "output_sha256": output_hash,
            },
        )
        logger.info("qe_output_completed", rows=combined.height, shards=shard_count)
        return StageResult(
            metrics=metrics,
            artifacts=(
                str(final_path.relative_to(workspace.paths.root)),
                str(manifest_path.relative_to(workspace.paths.root)),
                str(shard_root.relative_to(workspace.paths.root)),
            ),
            details={
                "input_sha256": input_hash,
                "output_sha256": output_hash,
                "model_repository": str(self.cfg.qe.repository),
                "model_revision": str(self.cfg.qe.revision),
            },
        )
