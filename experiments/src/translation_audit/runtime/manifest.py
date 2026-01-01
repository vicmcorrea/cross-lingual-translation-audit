"""Run provenance manifests that intentionally exclude environment secrets."""

import os
import platform
import sys
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any

from translation_audit.config import EXPERIMENT_ROOT
from translation_audit.runtime.files import sha256_file


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(UTC).isoformat()


def collect_environment() -> dict[str, Any]:
    """Collect reproducibility metadata without copying environment variables."""
    tracked_files = (EXPERIMENT_ROOT / "pyproject.toml", EXPERIMENT_ROOT / "uv.lock")
    fingerprints = {path.name: sha256_file(path) for path in tracked_files if path.is_file()}
    return {
        "captured_at": utc_now(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "process_id": os.getpid(),
        "dependency_file_sha256": fingerprints,
    }


def summarize_exception(error: BaseException) -> dict[str, str | int]:
    """Identify an error without persisting text that may contain sensitive data."""
    message = str(error)
    return {
        "error_type": type(error).__name__,
        "error_message_sha256": sha256(message.encode("utf-8")).hexdigest(),
        "error_message_length": len(message),
    }
