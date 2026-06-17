"""
BM25 keyword retrieval.

Uses the ``rank-bm25`` library (BM25Okapi implementation).
The corpus is built from chunk content and rebuilt lazily when the
chunk set changes.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BM25Result:
    chunk_id: str
    score: float  # Normalised to [0, 1]


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer: lowercase, split on non-alphanumeric."""
    return re.split(r"[^a-zA-Z0-9_$]+", text.lower())


class BM25Retriever:
    """
    BM25 full-text search over chunk content.

    The index is built lazily on first search and invalidated when
    new chunks are added via :meth:`add_documents`.
    """

    def __init__(self) -> None:
        self._corpus: list[str] = []
        self._chunk_ids: list[str] = []
        self._tokenized: list[list[str]] = []
        self._bm25 = None
        self._dirty = True

    def add_documents(self, chunk_ids: list[str], texts: list[str]) -> None:
        """
        Add or replace the full document corpus.

        For large repositories, call this once with all chunks.
        Calling it multiple times rebuilds the index.

        Args:
            chunk_ids: Parallel list of chunk IDs.
            texts:     Parallel list of chunk text content.
        """
        self._chunk_ids = list(chunk_ids)
        self._corpus = list(texts)
        self._tokenized = [_tokenize(t) for t in texts]
        self._bm25 = None
        self._dirty = True
        logger.debug("BM25 corpus updated: %d documents.", len(self._corpus))

    def _build_index(self) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:
            raise ImportError("rank-bm25 is required. Run: pip install rank-bm25") from e

        if self._tokenized:
            self._bm25 = BM25Okapi(self._tokenized)
        self._dirty = False
        logger.debug("BM25 index built over %d documents.", len(self._tokenized))

    def search(self, query: str, k: int = 50) -> list[BM25Result]:
        """
        Search for the ``k`` best-matching chunks.

        Args:
            query: Raw query string (will be tokenized).
            k:     Maximum results to return.

        Returns:
            List of :class:`BM25Result`, sorted by descending score.
        """
        if not self._chunk_ids:
            return []

        if self._dirty or self._bm25 is None:
            self._build_index()

        if self._bm25 is None:
            return []

        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)

        # Normalise scores to [0, 1]
        max_score = float(scores.max()) if scores.size > 0 else 1.0
        if max_score == 0:
            return []

        # Get top-k indices
        top_k = min(k, len(scores))
        top_indices = scores.argsort()[-top_k:][::-1]

        results: list[BM25Result] = []
        for idx in top_indices:
            raw_score = float(scores[idx])
            if raw_score <= 0:
                continue
            results.append(
                BM25Result(
                    chunk_id=self._chunk_ids[idx],
                    score=raw_score / max_score,
                )
            )
        return results

    @property
    def corpus_size(self) -> int:
        return len(self._chunk_ids)
