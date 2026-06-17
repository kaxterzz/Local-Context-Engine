"""Abstract base class for local embedding models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseEmbedder(ABC):
    """
    Abstract local embedding model interface.

    All embedders must work offline after the initial model download.
    No API keys, no cloud calls, no telemetry.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """HuggingFace model identifier or local path."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Output embedding dimension."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of document texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            Float32 numpy array of shape ``(len(texts), dimension)``.
        """

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a single query string.

        Query embedding may use a different prefix than document embedding
        (required by asymmetric models like BGE and Nomic).

        Args:
            text: Query string.

        Returns:
            Float32 numpy array of shape ``(dimension,)``.
        """

    def embed_documents_batched(
        self, texts: list[str], batch_size: int = 32
    ) -> np.ndarray:
        """
        Embed documents in batches to manage memory.

        Default implementation calls :meth:`embed_documents` in chunks.
        Subclasses may override for model-specific batch optimisation.

        Args:
            texts:      All texts to embed.
            batch_size: Number of texts per batch.

        Returns:
            Float32 numpy array of shape ``(len(texts), dimension)``.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        all_embeddings: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_emb = self.embed_documents(batch)
            all_embeddings.append(batch_emb)

        return np.vstack(all_embeddings).astype(np.float32)
