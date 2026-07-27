"""
BM25 keyword retrieval.

Implements BM25-Okapi scoring over a compact, array-backed inverted index.

Memory model
------------
The previous implementation (``rank_bm25.BM25Okapi``) kept a Python dict of
term frequencies per document (~100 bytes per posting) plus the raw corpus
and token lists — several GB for a large legacy repository. This version:

  - Never retains raw texts or token lists.
  - Stores the index as three int32 numpy arrays (term id, doc id, freq —
    12 bytes per posting) plus one shared vocabulary dict.
  - Caps the corpus at ``max_docs`` documents and ``max_tokens_per_doc``
    tokens per document.
  - Supports streaming construction via :meth:`add_batch` + :meth:`finalize`
    so the full corpus text is never materialised at once.

Scoring matches BM25-Okapi (k1=1.5, b=0.75, negative-IDF floor at
``epsilon × average_idf``, like ``rank_bm25``).
"""

from __future__ import annotations

import logging
import re
from array import array
from collections import Counter
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

_K1 = 1.5
_B = 0.75
_IDF_EPSILON = 0.25


@dataclass
class BM25Result:
    chunk_id: str
    score: float  # Normalised to [0, 1]


def _tokenize(text: str, max_tokens: int | None = None) -> list[str]:
    """Simple tokenizer: lowercase, split on non-alphanumeric."""
    tokens = re.split(r"[^a-zA-Z0-9_$]+", text.lower())
    if max_tokens is not None and len(tokens) > max_tokens:
        return tokens[:max_tokens]
    return tokens


class BM25Retriever:
    """
    BM25 full-text search over chunk content.

    Streaming usage (bounded memory)::

        bm25 = BM25Retriever(max_docs=100_000)
        for chunk_ids, texts in batches:
            bm25.add_batch(chunk_ids, texts)
        bm25.finalize()

    Or the legacy one-shot form::

        bm25.add_documents(all_chunk_ids, all_texts)
    """

    def __init__(
        self,
        max_docs: int = 100_000,
        max_tokens_per_doc: int = 512,
    ) -> None:
        self._max_docs = max_docs
        self._max_tokens_per_doc = max_tokens_per_doc
        self._chunk_ids: list[str] = []
        self._truncated = False
        self.reset()

    def reset(self) -> None:
        """Drop the corpus and index, releasing all memory."""
        self._chunk_ids = []
        self._truncated = False
        self._vocab: dict[str, int] = {}
        # Posting triples accumulated during build (4 bytes per value).
        self._acc_terms: array | None = array("i")
        self._acc_docs: array | None = array("i")
        self._acc_freqs: array | None = array("i")
        self._acc_doc_lens: array | None = array("i")
        # Finalized index arrays.
        self._doc_ids: np.ndarray | None = None
        self._freqs: np.ndarray | None = None
        self._term_starts: np.ndarray | None = None
        self._idf: np.ndarray | None = None
        self._doc_lens: np.ndarray | None = None
        self._avgdl: float = 0.0
        self._built = False

    def add_batch(self, chunk_ids: list[str], texts: list[str]) -> int:
        """
        Append a batch of documents to the corpus being built.

        Tokenizes immediately and discards the raw text. Returns how many
        documents were actually added (0 once ``max_docs`` is reached).
        """
        if self._acc_terms is None:
            # Index was already finalized; start a fresh corpus.
            self.reset()

        added = 0
        for chunk_id, text in zip(chunk_ids, texts, strict=False):
            if self._max_docs and len(self._chunk_ids) >= self._max_docs:
                if not self._truncated:
                    self._truncated = True
                    logger.warning(
                        "BM25 corpus reached max_docs=%d; further chunks are "
                        "excluded from keyword search (semantic search still "
                        "covers them). Raise CONTEXT_BM25_MAX_DOCS to include "
                        "more at the cost of RAM.",
                        self._max_docs,
                    )
                break
            doc_idx = len(self._chunk_ids)
            self._chunk_ids.append(chunk_id)
            tokens = _tokenize(text, self._max_tokens_per_doc)
            counts = Counter(t for t in tokens if t)
            self._acc_doc_lens.append(sum(counts.values()))
            for token, freq in counts.items():
                term_id = self._vocab.setdefault(token, len(self._vocab))
                self._acc_terms.append(term_id)
                self._acc_docs.append(doc_idx)
                self._acc_freqs.append(freq)
            added += 1
        self._built = False
        return added

    def add_documents(self, chunk_ids: list[str], texts: list[str]) -> None:
        """
        Replace the full document corpus in one call.

        Prefer :meth:`add_batch` + :meth:`finalize` for large corpora.
        """
        self.reset()
        self.add_batch(chunk_ids, texts)
        logger.debug("BM25 corpus updated: %d documents.", len(self._chunk_ids))

    def finalize(self) -> None:
        """Build the inverted index and free the accumulation buffers."""
        self._build_index()

    def _build_index(self) -> None:
        if self._acc_terms is None:
            # Already finalized (or reset with no new docs).
            self._built = True
            return

        n_docs = len(self._chunk_ids)
        n_terms = len(self._vocab)
        if n_docs and n_terms:
            term_ids = np.frombuffer(self._acc_terms, dtype=np.int32)
            doc_ids = np.frombuffer(self._acc_docs, dtype=np.int32)
            freqs = np.frombuffer(self._acc_freqs, dtype=np.int32)
            self._doc_lens = np.frombuffer(
                self._acc_doc_lens, dtype=np.int32
            ).astype(np.float32)
            self._avgdl = float(self._doc_lens.mean()) or 1.0

            # Group postings by term (CSR layout).
            order = np.argsort(term_ids, kind="stable")
            sorted_terms = term_ids[order]
            self._doc_ids = np.ascontiguousarray(doc_ids[order])
            self._freqs = np.ascontiguousarray(freqs[order]).astype(np.float32)
            df = np.bincount(sorted_terms, minlength=n_terms).astype(np.int64)
            self._term_starts = np.concatenate(
                ([0], np.cumsum(df))
            ).astype(np.int64)

            # BM25-Okapi IDF with the standard negative-IDF floor.
            idf = np.log(n_docs - df + 0.5) - np.log(df + 0.5)
            average_idf = float(idf.mean()) if idf.size else 0.0
            idf[idf < 0] = _IDF_EPSILON * average_idf
            self._idf = idf.astype(np.float32)

        # Free the accumulation buffers — index arrays are all that remain.
        self._acc_terms = None
        self._acc_docs = None
        self._acc_freqs = None
        self._acc_doc_lens = None
        self._built = True
        logger.debug("BM25 index built over %d documents.", n_docs)

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

        if not self._built:
            self._build_index()

        if self._doc_ids is None or self._idf is None:
            return []

        scores = np.zeros(len(self._chunk_ids), dtype=np.float32)
        norm = _K1 * (1.0 - _B + _B * self._doc_lens / self._avgdl)
        for token in set(_tokenize(query)):
            term_id = self._vocab.get(token)
            if term_id is None:
                continue
            start = self._term_starts[term_id]
            end = self._term_starts[term_id + 1]
            docs = self._doc_ids[start:end]
            f = self._freqs[start:end]
            scores[docs] += self._idf[term_id] * f * (_K1 + 1.0) / (f + norm[docs])

        max_score = float(scores.max()) if scores.size else 0.0
        if max_score <= 0:
            return []

        top_k = min(k, scores.size)
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

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

    @property
    def truncated(self) -> bool:
        """True when the corpus hit ``max_docs`` and dropped documents."""
        return self._truncated
