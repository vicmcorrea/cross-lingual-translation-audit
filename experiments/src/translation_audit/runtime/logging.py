"""Structured run logging."""

import io
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO, cast

import structlog


class _TeeStream(io.TextIOBase):
    """Write structured events to a run file and optionally standard output."""

    def __init__(self, primary: TextIO, secondary: TextIO | None) -> None:
        self._primary = primary
        self._secondary = secondary

    def write(self, message: str) -> int:
        written = self._primary.write(message)
        if self._secondary is not None:
            self._secondary.write(message)
        return written

    def writable(self) -> bool:
        """Report that structlog can write to this stream."""
        return True

    def flush(self) -> None:
        self._primary.flush()
        if self._secondary is not None:
            self._secondary.flush()


@dataclass(slots=True)
class LoggingHandle:
    """Own the file stream used by structlog for a single run."""

    stream: TextIO

    def close(self) -> None:
        """Flush and close the run-owned log file."""
        structlog.reset_defaults()
        self.stream.flush()
        self.stream.close()


def configure_structured_logging(
    log_path: Path,
    level: str,
    include_console: bool,
    run_id: str,
) -> LoggingHandle:
    """Configure JSON logging bound to one immutable run."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stream = log_path.open("x", encoding="utf-8")
    target = _TeeStream(stream, sys.stdout if include_console else None)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(sort_keys=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level.upper(), logging.INFO)),
        logger_factory=structlog.PrintLoggerFactory(file=cast(TextIO, target)),
        cache_logger_on_first_use=True,
    )
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(run_id=run_id)
    return LoggingHandle(stream=stream)
