"""Persistent, text-safe protocol for the isolated EmoAtlas runtime."""

import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import Protocol, Self, TextIO, cast


class EmotionBackend(Protocol):
    """A backend that processes text in memory and returns only numeric features."""

    def __enter__(self) -> Self: ...

    def analyze(self, rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]: ...

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...


class EmotionWorkerError(RuntimeError):
    """Safe worker error whose message never contains response text."""


class EmoAtlasWorkerBackend:
    """Keep both language models loaded in one isolated Python 3.11 worker."""

    def __init__(
        self,
        *,
        python_candidates: Sequence[str],
        module: str,
        portuguese_model: str,
        english_model: str,
        max_distance: int,
    ) -> None:
        self.python_candidates = tuple(python_candidates)
        self.module = module
        self.portuguese_model = portuguese_model
        self.english_model = english_model
        self.max_distance = max_distance
        self._process: subprocess.Popen[str] | None = None

    def __enter__(self) -> Self:
        executable = next((Path(item) for item in self.python_candidates if Path(item).is_file()), None)
        if executable is None:
            raise EmotionWorkerError("No configured EmoAtlas Python runtime is available")
        command = [
            str(executable),
            "-m",
            self.module,
            "--portuguese-model",
            self.portuguese_model,
            "--english-model",
            self.english_model,
            "--max-distance",
            str(self.max_distance),
        ]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        return self

    def _streams(self) -> tuple[TextIO, TextIO]:
        if self._process is None or self._process.stdin is None or self._process.stdout is None:
            raise EmotionWorkerError("EmoAtlas worker is not running")
        return cast(TextIO, self._process.stdin), cast(TextIO, self._process.stdout)

    def analyze(self, rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
        """Send one in-memory shard and accept only a text-free response."""
        stdin, stdout = self._streams()
        request = {
            "command": "analyze",
            "rows": [
                {
                    "pair_id": str(row["pair_id"]),
                    "source_pt": str(row["source_pt"]),
                    "translation_en": str(row["translation_en"]),
                }
                for row in rows
            ],
        }
        try:
            stdin.write(json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n")
            stdin.flush()
            response_line = stdout.readline()
        except (BrokenPipeError, OSError) as error:
            raise EmotionWorkerError("EmoAtlas worker communication failed") from error
        if not response_line:
            raise EmotionWorkerError("EmoAtlas worker terminated unexpectedly")
        try:
            decoded: object = json.loads(response_line)
        except json.JSONDecodeError as error:
            raise EmotionWorkerError("EmoAtlas worker returned an invalid control response") from error
        if not isinstance(decoded, dict):
            raise EmotionWorkerError("EmoAtlas worker returned an invalid control response")
        response = cast(dict[str, object], decoded)
        if response.get("status") != "ok":
            raise EmotionWorkerError("EmoAtlas worker reported a processing failure")
        records_value = response.get("records")
        if not isinstance(records_value, list):
            raise EmotionWorkerError("EmoAtlas worker returned an invalid record count")
        records = cast(list[object], records_value)
        if len(records) != len(rows):
            raise EmotionWorkerError("EmoAtlas worker returned an invalid record count")
        if not all(isinstance(record, dict) for record in records):
            raise EmotionWorkerError("EmoAtlas worker returned an invalid record schema")
        return [cast(dict[str, object], record) for record in records]

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exception, traceback
        process = self._process
        self._process = None
        if process is None:
            return
        if process.poll() is None and process.stdin is not None:
            try:
                process.stdin.write('{"command":"shutdown"}\n')
                process.stdin.flush()
                process.wait(timeout=10)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        if exception_type is None and process.returncode not in (0, None):
            raise EmotionWorkerError("EmoAtlas worker exited unsuccessfully")
