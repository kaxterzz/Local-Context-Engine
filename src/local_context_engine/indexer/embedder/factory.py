"""Embedder factory — creates the appropriate embedder from configuration."""

from __future__ import annotations

from local_context_engine.indexer.embedder.base_embedder import BaseEmbedder
from local_context_engine.indexer.embedder.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
)


class EmbedderFactory:
    """Creates :class:`BaseEmbedder` instances from configuration."""

    @staticmethod
    def create(config: "EmbeddingConfig") -> BaseEmbedder:  # type: ignore[name-defined]
        """
        Create an embedder from the embedding configuration.

        Currently all supported models use the sentence-transformers backend.
        This factory method is the extension point for future backends
        (e.g. llama.cpp, Ollama local, etc.).

        Args:
            config: :class:`~local_context_engine.core.config.EmbeddingConfig` instance.

        Returns:
            A configured :class:`BaseEmbedder`.
        """
        from local_context_engine.core.config import EmbeddingConfig  # noqa: PLC0415

        assert isinstance(config, EmbeddingConfig)
        return SentenceTransformerEmbedder.from_config(config)
