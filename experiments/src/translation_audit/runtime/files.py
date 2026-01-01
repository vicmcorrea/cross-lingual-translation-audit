"""Exclusive and content-hashing file operations."""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def write_text_exclusive(path: Path, content: str) -> Path:
    """Atomically create a text file and fail if the destination exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> Path:
    """Atomically create a formatted JSON document without overwriting."""
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)
    return write_text_exclusive(path, f"{serialized}\n")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest without loading a file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
