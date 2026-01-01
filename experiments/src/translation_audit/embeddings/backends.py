"""Frozen embedding backends with a download-free synthetic test implementation."""

import hashlib
import importlib
import re
from collections.abc import Sequence
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt
from omegaconf import DictConfig

FloatMatrix = npt.NDArray[np.float32]
_PINNED_QWEN_REVISIONS = {
    "Qwen/Qwen3-Embedding-4B": "5cf2132abc99cad020ac570b19d031efec650f2b",
    "Qwen/Qwen3-Embedding-8B": "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af",
}
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")


class EmbeddingBackend(Protocol):
    """Minimal interface required by the embedding stage."""

    @property
    def dimension(self) -> int | None: ...

    def encode_queries(self, texts: Sequence[str], instruction: str) -> FloatMatrix: ...

    def encode_documents(self, texts: Sequence[str]) -> FloatMatrix: ...


class _SentenceTransformerModel(Protocol):
    """Typed boundary around the optional Sentence Transformers dependency."""

    max_seq_length: int

    def __init__(self, model_name_or_path: str, **kwargs: object) -> None: ...

    def get_sentence_embedding_dimension(self) -> int | None: ...

    def encode(self, sentences: list[str], **kwargs: object) -> object: ...


def validate_frozen_qwen_config(encoder_cfg: DictConfig) -> None:
    """Reject unpinned, mutable, or repository-code-enabled encoders."""
    repository = str(encoder_cfg.repository)
    revision = str(encoder_cfg.revision)
    if repository not in _PINNED_QWEN_REVISIONS:
        raise ValueError("The active embedding stage accepts only the approved Qwen3 repositories")
    if _COMMIT_SHA.fullmatch(revision) is None:
        raise ValueError("encoder.revision must be an immutable 40-character commit SHA")
    if revision != _PINNED_QWEN_REVISIONS[repository]:
        raise ValueError("encoder.revision does not match the approved Qwen3 checkpoint")
    if not bool(encoder_cfg.frozen):
        raise ValueError("The embedding encoder must remain frozen")
    if bool(encoder_cfg.trust_remote_code):
        raise ValueError("Remote repository code is not permitted")
    if not bool(encoder_cfg.normalize_embeddings):
        raise ValueError("Cosine-comparable embeddings must be L2-normalized")


def _normalize(matrix: FloatMatrix) -> FloatMatrix:
    """Return finite, unit-normalized float32 rows."""
    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] <= 0:
        raise ValueError("Embedding backend returned an invalid matrix shape")
    if not np.isfinite(array).all():
        raise ValueError("Embedding backend returned non-finite values")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embedding backend returned a zero vector")
    return np.asarray(array / norms, dtype=np.float32)


class SyntheticEmbeddingBackend:
    """Deterministic local backend for contract tests without model downloads."""

    def __init__(self, dimension: int, seed: int) -> None:
        if dimension <= 0:
            raise ValueError("Synthetic embedding dimension must be positive")
        self._dimension = dimension
        self._seed = seed

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode_queries(self, texts: Sequence[str], instruction: str) -> FloatMatrix:
        return self._encode(texts, domain=f"query\0{instruction}")

    def encode_documents(self, texts: Sequence[str]) -> FloatMatrix:
        return self._encode(texts, domain="document")

    def _encode(self, texts: Sequence[str], domain: str) -> FloatMatrix:
        rows = np.empty((len(texts), self._dimension), dtype=np.float32)
        for row_index, value in enumerate(texts):
            seed_material = f"{self._seed}\0{domain}\0{value}".encode()
            values: list[float] = []
            counter = 0
            while len(values) < self._dimension:
                digest = hashlib.sha256(seed_material + counter.to_bytes(4, "big")).digest()
                values.extend((byte - 127.5) / 127.5 for byte in digest)
                counter += 1
            rows[row_index] = values[: self._dimension]
        return _normalize(rows)


class SentenceTransformerBackend:
    """Lazy Sentence Transformers adapter for revision-pinned Qwen3 encoders."""

    def __init__(self, encoder_cfg: DictConfig, device: str) -> None:
        validate_frozen_qwen_config(encoder_cfg)
        try:
            torch_module = importlib.import_module("torch")
            sentence_transformers_module = importlib.import_module("sentence_transformers")
        except ImportError as error:
            raise RuntimeError(
                "Install the locked gpu dependency group before executing Qwen3 embeddings"
            ) from error

        precision = str(encoder_cfg.precision)
        torch_dtype = {
            "bfloat16": torch_module.bfloat16,
            "float16": torch_module.float16,
            "float32": torch_module.float32,
        }.get(precision)
        if torch_dtype is None:
            raise ValueError(f"Unsupported embedding precision {precision}")

        self._batch_size = int(encoder_cfg.batch_size)
        self._max_length = int(encoder_cfg.max_length)
        self._normalize_embeddings = bool(encoder_cfg.normalize_embeddings)
        constructor = cast(
            "type[_SentenceTransformerModel]",
            sentence_transformers_module.SentenceTransformer,
        )
        self._model = constructor(
            str(encoder_cfg.repository),
            revision=str(encoder_cfg.revision),
            device=device,
            trust_remote_code=False,
            model_kwargs={"torch_dtype": torch_dtype},
            tokenizer_kwargs={"padding_side": "left"},
        )
        self._model.max_seq_length = self._max_length

    @property
    def dimension(self) -> int:
        dimension = self._model.get_sentence_embedding_dimension()
        if dimension is None:
            raise ValueError("Sentence Transformer did not report an embedding dimension")
        return dimension

    def encode_queries(self, texts: Sequence[str], instruction: str) -> FloatMatrix:
        prompt = f"Instruct: {instruction}\nQuery: "
        return self._encode(texts, prompt=prompt)

    def encode_documents(self, texts: Sequence[str]) -> FloatMatrix:
        return self._encode(texts, prompt=None)

    def _encode(self, texts: Sequence[str], prompt: str | None) -> FloatMatrix:
        encoded = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize_embeddings,
            prompt=prompt,
        )
        return _normalize(np.asarray(encoded, dtype=np.float32))


def create_embedding_backend(cfg: DictConfig) -> EmbeddingBackend:
    """Construct the configured backend after enforcing the frozen encoder contract."""
    validate_frozen_qwen_config(cfg.encoder)
    backend_name = str(cfg.stage.backend)
    if backend_name == "synthetic":
        return SyntheticEmbeddingBackend(
            dimension=int(cfg.stage.synthetic_dimension),
            seed=int(cfg.stage.synthetic_seed),
        )
    if backend_name == "sentence_transformers":
        return SentenceTransformerBackend(cfg.encoder, device=str(cfg.runtime.device))
    raise ValueError(f"Unknown embedding backend {backend_name}")
