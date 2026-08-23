"""Run provenance manifests that intentionally exclude environment secrets."""

import os
import platform
import sys
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from translation_audit.config import EXPERIMENT_ROOT
from translation_audit.runtime.files import sha256_file


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(UTC).isoformat()


def _source_tree_fingerprint(paths: tuple[Path, ...]) -> tuple[str, int]:
    """Hash the ordered path and content digest of non-sensitive scientific source files."""
    files = sorted(
        path
        for root in paths
        for path in root.rglob("*")
        if path.is_file() and path.suffix in {".py", ".yaml", ".yml"}
    )
    digest = sha256()
    for path in files:
        relative = path.relative_to(EXPERIMENT_ROOT).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), len(files)


def collect_environment() -> dict[str, Any]:
    """Collect reproducibility metadata without copying environment variables."""
    tracked_files = (EXPERIMENT_ROOT / "pyproject.toml", EXPERIMENT_ROOT / "uv.lock")
    fingerprints = {path.name: sha256_file(path) for path in tracked_files if path.is_file()}
    source_sha256, source_file_count = _source_tree_fingerprint(
        (EXPERIMENT_ROOT / "src", EXPERIMENT_ROOT / "conf")
    )
    return {
        "captured_at": utc_now(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "process_id": os.getpid(),
        "dependency_file_sha256": fingerprints,
        "scientific_source_sha256": source_sha256,
        "scientific_source_file_count": source_file_count,
    }


def summarize_exception(error: BaseException) -> dict[str, str | int]:
    """Identify an error without persisting text that may contain sensitive data."""
    message = str(error)
    return {
        "error_type": type(error).__name__,
        "error_message_sha256": sha256(message.encode("utf-8")).hexdigest(),
        "error_message_length": len(message),
    }
