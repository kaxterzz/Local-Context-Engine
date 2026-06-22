"""Abstract vector store interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class VectorSearchResult:
    """A single result from a vector similarity search."""

    chunk_id: str
    score: float       # Cosine similarity or inner product score (0–1)
    metadata: dict     # Any additional metadata stored with the vector


class BaseVectorStore(ABC):
    """
    Abstract interface for local vector databases.

    All implementations must persist data to local storage only.
    No cloud connectivity is permitted.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Expected embedding vector dimension."""

    @property
    @abstractmethod
    def total_vectors(self) -> int:
        """Number of vectors currently stored."""

    @property
    @abstractmethod
    def stored_chunk_ids(self) -> set[str]:
        """Chunk IDs that currently have a stored vector."""

    @abstractmethod
    def insert_vectors(
        self,
        chunk_ids: list[str],
        embeddings: np.ndarray,
        metadata: list[dict] | None = None,
    ) -> None:
        """
        Insert embedding vectors.

        Args:
            chunk_ids:  Unique IDs for each vector (parallel to ``embeddings``).
            embeddings: Float32 array of shape ``(n, dimension)``.
            metadata:   Optional list of metadata dicts, one per vector.
        """

    @abstractmethod
    def search_vectors(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        filter_metadata: dict | None = None,
    ) -> list[VectorSearchResult]:
        """
        Find the ``k`` nearest neighbours to ``query_embedding``.

        Args:
            query_embedding: Float32 array of shape ``(dimension,)``.
            k:               Number of results to return.
            filter_metadata: Optional metadata key/value filters.

        Returns:
            List of :class:`VectorSearchResult`, sorted by descending score.
        """

    @abstractmethod
    def delete_vectors(self, chunk_ids: list[str]) -> int:
        """
        Remove vectors by their chunk IDs.

        Returns:
            Number of vectors actually deleted.
        """

    @abstractmethod
    def get_vector(self, chunk_id: str) -> np.ndarray | None:
        """Return the stored vector for ``chunk_id``, or ``None``."""

    @abstractmethod
    def save(self) -> None:
        """Persist the index to disk."""

    @abstractmethod
    def load(self) -> bool:
        """
        Load the index from disk.

        Returns:
            ``True`` if the index was loaded successfully, ``False`` if no
            persisted index was found (first run).
        """

    @abstractmethod
    def clear(self) -> None:
        """Delete all vectors and reset the index."""
