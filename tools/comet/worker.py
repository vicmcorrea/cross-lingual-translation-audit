"""Line-oriented worker for pinned, reference-free COMETKiwi inference."""

import argparse
import contextlib
import json
import re
import sys
from pathlib import Path
from typing import Any

_FULL_COMMIT = re.compile(r"[0-9a-f]{40}")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--device", required=True, choices=("cpu", "cuda"))
    parser.add_argument("--precision", required=True, choices=("float16", "bfloat16", "float32"))
    arguments = parser.parse_args()
    if _FULL_COMMIT.fullmatch(arguments.revision) is None:
        parser.error("--revision must be a full 40-character commit")
    if arguments.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if arguments.device != "cuda" and arguments.precision != "float32":
        parser.error("Non-CUDA COMET execution requires float32")
    return arguments


def _load_model(repository: str, revision: str, precision: str) -> Any:
    from comet import load_from_checkpoint
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(repo_id=repository, revision=revision))
    preferred = snapshot / "checkpoints" / "model.ckpt"
    if preferred.is_file():
        checkpoint = preferred
    else:
        candidates = sorted(snapshot.rglob("*.ckpt"))
        if len(candidates) != 1:
            raise RuntimeError("Pinned COMET snapshot does not contain one unambiguous checkpoint")
        checkpoint = candidates[0]
    model = load_from_checkpoint(str(checkpoint))
    if precision == "float16":
        model.half()
    elif precision == "bfloat16":
        model.bfloat16()
    else:
        model.float()
    return model


def _score(model: Any, samples: list[dict[str, str]], batch_size: int, device: str) -> list[float]:
    prediction = model.predict(
        samples,
        batch_size=batch_size,
        gpus=1 if device == "cuda" else 0,
        accelerator="gpu" if device == "cuda" else device,
        progress_bar=False,
        length_batching=True,
    )
    return [float(score) for score in prediction.scores]


def main() -> None:
    """Load the pinned model once, then score JSON requests from standard input."""
    arguments = _arguments()
    protocol_output = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        model = _load_model(arguments.repository, arguments.revision, arguments.precision)
    protocol_output.write(json.dumps({"status": "ready"}, separators=(",", ":")) + "\n")
    protocol_output.flush()
    for line in sys.stdin:
        request = json.loads(line)
        request_id = request.get("request_id")
        raw_samples = request.get("samples")
        if not isinstance(request_id, int) or not isinstance(raw_samples, list):
            raise TypeError("Invalid COMET worker request")
        samples: list[dict[str, str]] = []
        for raw_sample in raw_samples:
            if not isinstance(raw_sample, dict):
                raise TypeError("Invalid COMET sample")
            source = raw_sample.get("src")
            translation = raw_sample.get("mt")
            if not isinstance(source, str) or not isinstance(translation, str):
                raise TypeError("COMET samples require string src and mt values")
            samples.append({"src": source, "mt": translation})
        with contextlib.redirect_stdout(sys.stderr):
            scores = _score(model, samples, arguments.batch_size, arguments.device)
        protocol_output.write(
            json.dumps({"request_id": request_id, "scores": scores}, separators=(",", ":")) + "\n"
        )
        protocol_output.flush()


if __name__ == "__main__":
    main()
