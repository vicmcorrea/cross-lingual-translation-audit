"""Append-only lifecycle checkpoints."""

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from translation_audit.runtime.files import write_json_exclusive

_SAFE_EVENT = re.compile(r"[^a-z0-9_-]+")


class CheckpointManager:
    """Create ordered checkpoint records without modifying prior records."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=False)
        self._sequence = 0

    def write(self, event: str, payload: dict[str, Any] | None = None) -> Path:
        """Append a lifecycle checkpoint and return its path."""
        self._sequence += 1
        safe_event = _SAFE_EVENT.sub("_", event.lower()).strip("_") or "event"
        checkpoint_path = self.directory / f"{self._sequence:06d}_{safe_event}.json"
        document = {
            "sequence": self._sequence,
            "event": event,
            "created_at": datetime.now(UTC).isoformat(),
            "payload": payload or {},
        }
        return write_json_exclusive(checkpoint_path, document)
