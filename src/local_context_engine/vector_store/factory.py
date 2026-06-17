"""Vector store factory."""

from __future__ import annotations

from local_context_engine.vector_store.base_store import BaseVectorStore
from local_context_engine.vector_store.faiss_store import FAISSVectorStore


class VectorStoreFactory:
    """Creates :class:`BaseVectorStore` instances from configuration."""

    @staticmethod
    def create(config: "VectorStoreConfig") -> BaseVectorStore:  # type: ignore[name-defined]
        """
        Instantiate the configured vector store backend.

        Currently supports: ``faiss``.
        Future backends: ``chroma``, ``qdrant``.

        Args:
            config: :class:`~local_context_engine.core.config.VectorStoreConfig`.

        Returns:
            Configured :class:`BaseVectorStore` instance.
        """
        from local_context_engine.core.config import VectorStoreConfig  # noqa: PLC0415

        assert isinstance(config, VectorStoreConfig)

        backend = config.backend.lower()
        if backend == "faiss":
            return FAISSVectorStore.from_config(config)
        else:
            raise ValueError(
                f"Unknown vector store backend: {config.backend!r}. "
                "Supported: 'faiss'"
            )
