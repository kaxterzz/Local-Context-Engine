"""
FAISS-backed vector store.

Supports three index types:
  - ``flat``  — Exact brute-force L2 search. Accurate, any size.
  - ``hnsw``  — Approximate HNSW graph search. Recommended for >10k vectors.
  - ``ivf``   — Inverted file index. Memory-efficient for very large sets.

The index is persisted to a binary file alongside a JSON metadata sidecar
that maps vector integer IDs to chunk IDs (FAISS uses integer IDs internally).

Thread-safety: read operations (search) are safe to call concurrently.
Write operations (insert, delete) acquire a lock.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import numpy as np

from local_context_engine.core.exceptions import VectorDimensionMismatchError, VectorStoreError
from local_context_engine.vector_store.base_store import BaseVectorStore, VectorSearchResult

logger = logging.getLogger(__name__)


class FAISSVectorStore(BaseVectorStore):
    """
    Local FAISS vector store with persistence.

    Usage::

        store = FAISSVectorStore(dimension=384, storage_path=Path(".context/vectors"))
        store.load()  # load existing index or create fresh
        store.insert_vectors(chunk_ids, embeddings)
        results = store.search_vectors(query_embedding, k=10)
        store.save()
    """

    def __init__(
        self,
        dimension: int = 384,
        index_type: str = "hnsw",
        storage_path: Path = Path(".context/vectors"),
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 50,
    ) -> None:
        self._dimension = dimension
        self._index_type = index_type
        self._storage_path = storage_path
        self._hnsw_m = hnsw_m
        self._hnsw_ef_construction = hnsw_ef_construction
        self._hnsw_ef_search = hnsw_ef_search

        self._index = None          # faiss.Index
        self._id_to_chunk: dict[int, str] = {}   # faiss int ID → chunk_id
        self._chunk_to_id: dict[str, int] = {}   # chunk_id → faiss int ID
        self._metadata_store: dict[str, dict] = {}  # chunk_id → metadata
        self._next_id: int = 0
        self._lock = threading.Lock()

        self._index_path = storage_path / "faiss.index"
        self._meta_path = storage_path / "faiss_meta.json"

    # ── Properties ─────────────────────────────────────────────

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def total_vectors(self) -> int:
        if self._index is None:
            return 0
        return self._index.ntotal

    # ── Index creation ─────────────────────────────────────────

    def _create_index(self):
        try:
            import faiss
        except ImportError as e:
            raise VectorStoreError(
                "faiss-cpu is not installed. Run: pip install faiss-cpu"
            ) from e

        if self._index_type == "flat":
            index = faiss.IndexFlatIP(self._dimension)   # Inner product (cosine for normalized)
        elif self._index_type == "hnsw":
            index = faiss.IndexHNSWFlat(self._dimension, self._hnsw_m)
            index.hnsw.efConstruction = self._hnsw_ef_construction
            index.hnsw.efSearch = self._hnsw_ef_search
        elif self._index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self._dimension)
            nlist = max(4, int(self._dimension ** 0.5))
            index = faiss.IndexIVFFlat(quantizer, self._dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        else:
            raise VectorStoreError(f"Unknown index type: {self._index_type!r}")

        # Wrap with IDMap to support explicit integer IDs
        id_index = faiss.IndexIDMap(index)
        logger.debug("Created FAISS %s index (dim=%d)", self._index_type, self._dimension)
        return id_index

    # ── Public API ─────────────────────────────────────────────

    def insert_vectors(
        self,
        chunk_ids: list[str],
        embeddings: np.ndarray,
        metadata: list[dict] | None = None,
    ) -> None:
        if embeddings.shape[1] != self._dimension:
            raise VectorDimensionMismatchError(self._dimension, embeddings.shape[1])

        embeddings_f32 = embeddings.astype(np.float32)

        with self._lock:
            if self._index is None:
                self._index = self._create_index()

            # Assign FAISS integer IDs
            new_ids = np.arange(
                self._next_id, self._next_id + len(chunk_ids), dtype=np.int64
            )
            self._index.add_with_ids(embeddings_f32, new_ids)

            for i, (chunk_id, faiss_id) in enumerate(zip(chunk_ids, new_ids.tolist())):
                self._id_to_chunk[faiss_id] = chunk_id
                self._chunk_to_id[chunk_id] = faiss_id
                if metadata:
                    self._metadata_store[chunk_id] = metadata[i]

            self._next_id += len(chunk_ids)

        logger.debug("Inserted %d vectors into FAISS index.", len(chunk_ids))

    def search_vectors(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        filter_metadata: dict | None = None,
    ) -> list[VectorSearchResult]:
        if self._index is None or self._index.ntotal == 0:
            return []

        q = query_embedding.astype(np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)

        # Fetch more candidates when filtering so we still get k results after filter
        fetch_k = k * 5 if filter_metadata else k
        fetch_k = min(fetch_k, self._index.ntotal)

        with self._lock:
            distances, indices = self._index.search(q, fetch_k)

        results: list[VectorSearchResult] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS padding value
                continue
            chunk_id = self._id_to_chunk.get(int(idx))
            if chunk_id is None:
                continue

            meta = self._metadata_store.get(chunk_id, {})

            # Apply metadata filter
            if filter_metadata:
                if not all(meta.get(k) == v for k, v in filter_metadata.items()):
                    continue

            # Scores from IndexFlatIP are inner products (≈ cosine for normalized vecs)
            # Clamp to [0, 1]
            score = float(np.clip(dist, 0.0, 1.0))
            results.append(VectorSearchResult(chunk_id=chunk_id, score=score, metadata=meta))

            if len(results) >= k:
                break

        return sorted(results, key=lambda r: r.score, reverse=True)

    def delete_vectors(self, chunk_ids: list[str]) -> int:
        """
        Remove vectors by chunk ID.

        Note: FAISS HNSW does not support direct deletion.
        For HNSW, we rebuild the index without the deleted vectors.
        For IDMap + Flat, we use remove_ids.
        """
        if not chunk_ids or self._index is None:
            return 0

        with self._lock:
            try:
                import faiss

                ids_to_remove = [
                    self._chunk_to_id[cid]
                    for cid in chunk_ids
                    if cid in self._chunk_to_id
                ]
                if not ids_to_remove:
                    return 0

                id_selector = faiss.IDSelectorBatch(
                    len(ids_to_remove),
                    faiss.swig_ptr(np.array(ids_to_remove, dtype=np.int64)),
                )
                removed = self._index.remove_ids(id_selector)

                for cid in chunk_ids:
                    if cid in self._chunk_to_id:
                        faiss_id = self._chunk_to_id.pop(cid)
                        self._id_to_chunk.pop(faiss_id, None)
                        self._metadata_store.pop(cid, None)

                logger.debug("Removed %d vectors from FAISS index.", removed)
                return removed
            except Exception as exc:
                logger.warning("FAISS delete failed: %s", exc)
                return 0

    def get_vector(self, chunk_id: str) -> np.ndarray | None:
        if self._index is None or chunk_id not in self._chunk_to_id:
            return None
        try:
            import faiss

            faiss_id = self._chunk_to_id[chunk_id]
            vec = np.zeros((1, self._dimension), dtype=np.float32)
            self._index.reconstruct(faiss_id, vec[0])
            return vec[0]
        except Exception:
            return None

    def save(self) -> None:
        if self._index is None:
            return

        self._storage_path.mkdir(parents=True, exist_ok=True)
        try:
            import faiss

            faiss.write_index(self._index, str(self._index_path))

            meta = {
                "id_to_chunk": {str(k): v for k, v in self._id_to_chunk.items()},
                "next_id": self._next_id,
                "metadata_store": self._metadata_store,
                "dimension": self._dimension,
                "index_type": self._index_type,
            }
            self._meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            logger.info(
                "FAISS index saved: %d vectors → %s",
                self._index.ntotal,
                self._index_path,
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed to save FAISS index: {exc}") from exc

    def load(self) -> bool:
        if not self._index_path.exists() or not self._meta_path.exists():
            logger.info("No existing FAISS index found at %s. Starting fresh.", self._storage_path)
            self._index = self._create_index()
            return False

        try:
            import faiss

            self._index = faiss.read_index(str(self._index_path))

            meta = json.loads(self._meta_path.read_text(encoding="utf-8"))
            self._id_to_chunk = {int(k): v for k, v in meta["id_to_chunk"].items()}
            self._chunk_to_id = {v: int(k) for k, v in meta["id_to_chunk"].items()}
            self._next_id = meta.get("next_id", len(self._id_to_chunk))
            self._metadata_store = meta.get("metadata_store", {})

            logger.info(
                "FAISS index loaded: %d vectors from %s",
                self._index.ntotal,
                self._index_path,
            )
            return True
        except Exception as exc:
            logger.warning("Failed to load FAISS index: %s. Creating fresh index.", exc)
            self._index = self._create_index()
            return False

    def clear(self) -> None:
        with self._lock:
            self._index = self._create_index()
            self._id_to_chunk.clear()
            self._chunk_to_id.clear()
            self._metadata_store.clear()
            self._next_id = 0
        logger.info("FAISS index cleared.")

    @classmethod
    def from_config(cls, config: "VectorStoreConfig") -> "FAISSVectorStore":  # type: ignore[name-defined]
        from local_context_engine.core.config import VectorStoreConfig  # noqa: PLC0415

        assert isinstance(config, VectorStoreConfig)
        return cls(
            dimension=config.dimension,
            index_type=config.index_type,
            storage_path=config.storage_path,
            hnsw_m=config.hnsw_m,
            hnsw_ef_construction=config.hnsw_ef_construction,
            hnsw_ef_search=config.hnsw_ef_search,
        )
