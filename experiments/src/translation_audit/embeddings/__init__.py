"""Embedding backends for frozen multilingual encoders."""

from translation_audit.embeddings.backends import (
    EmbeddingBackend,
    SentenceTransformerBackend,
    SyntheticEmbeddingBackend,
    create_embedding_backend,
)

__all__ = [
    "EmbeddingBackend",
    "SentenceTransformerBackend",
    "SyntheticEmbeddingBackend",
    "create_embedding_backend",
]
