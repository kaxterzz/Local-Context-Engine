"""
Tests for bounded-batch streaming in the indexing pipeline:
queue bounds, incremental re-index, graceful cancellation, and
memory-pressure handling.
"""

from __future__ import annotations

from datetime import UTC
from pathlib import Path

import numpy as np

from local_context_engine.core.config import (
    ChunkingConfig,
    EngineConfig,
    LimitsConfig,
    MetadataStoreConfig,
    SecurityConfig,
    SymbolGraphConfig,
    VectorStoreConfig,
)
from local_context_engine.core.instance import MemoryMonitor, MemoryPressure
from local_context_engine.indexer.chunkers.symbol_chunker import SymbolChunker
from local_context_engine.indexer.embedder.base_embedder import BaseEmbedder
from local_context_engine.indexer.parsers.registry import ParserRegistry
from local_context_engine.indexer.pipeline import IndexingPipeline, LazyFileSources
from local_context_engine.indexer.scanner.file_scanner import FileScanner
from local_context_engine.metadata_store.database import Database
from local_context_engine.vector_store.factory import VectorStoreFactory


class RecordingEmbedder(BaseEmbedder):
    """Fake embedder that records every batch size it receives."""

    def __init__(self, dimension: int = 64) -> None:
        self._dim = dimension
        self.batch_sizes: list[int] = []

    @property
    def model_name(self) -> str:
        return "recording-fake"

    @property
    def dimension(self) -> int:
        return self._dim

    def _vec(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.standard_normal(self._dim).astype(np.float32)
        return v / np.linalg.norm(v)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)

    def embed_documents_batched(
        self, texts: list[str], batch_size: int | None = None
    ) -> np.ndarray:
        self.batch_sizes.append(len(texts))
        return self.embed_documents(texts)


def _make_repo(root: Path, n_files: int) -> None:
    (root / "src").mkdir(parents=True, exist_ok=True)
    for i in range(n_files):
        (root / "src" / f"module_{i:03d}.py").write_text(
            f'"""Module {i}."""\n\n\n'
            f"def handler_{i}(payload):\n"
            f'    """Process payload {i}."""\n'
            f"    return payload + {i}\n\n\n"
            f"class Service{i}:\n"
            f'    """Service class {i}."""\n\n'
            f"    def run(self):\n"
            f"        return handler_{i}(41)\n"
        )


def _make_pipeline(
    tmp_path: Path,
    limits: LimitsConfig,
    embedder: RecordingEmbedder,
    memory_monitor: MemoryMonitor | None = None,
) -> IndexingPipeline:
    config = EngineConfig(
        vector_store=VectorStoreConfig(
            dimension=64, index_type="flat", storage_path=tmp_path / "vectors"
        ),
        metadata_store=MetadataStoreConfig(db_path=tmp_path / "meta.db"),
        chunking=ChunkingConfig(min_tokens=10, target_tokens=100, max_tokens=300),
        security=SecurityConfig(never_index_patterns=[]),
        symbol_graph=SymbolGraphConfig(
            persist=True, storage_path=tmp_path / "graph.pkl"
        ),
        limits=limits,
    )
    from local_context_engine.symbol_graph.graph import SymbolGraph

    return IndexingPipeline(
        config=config,
        scanner=FileScanner.from_config(config),
        parser_registry=ParserRegistry.default(),
        chunker=SymbolChunker.from_config(config),
        database=Database.from_config(config.metadata_store),
        vector_store=VectorStoreFactory.create(config.vector_store),
        embedder=embedder,
        symbol_graph=SymbolGraph(),
        memory_monitor=memory_monitor,
    )


class TestBoundedBatches:
    async def test_embeddings_flow_in_bounded_batches(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        _make_repo(repo, 30)
        embedder = RecordingEmbedder()
        limits = LimitsConfig(index_batch_size=5, max_queue_size=25)
        pipeline = _make_pipeline(tmp_path / "work", limits, embedder)

        await pipeline.initialize()
        stats = await pipeline.index_repository(repo, incremental=True)
        await pipeline.shutdown()

        assert stats.files_indexed == 30
        assert stats.total_chunks > 0
        # Multiple flushes happened — never one repo-sized embed call.
        assert len(embedder.batch_sizes) >= 30 // 5
        # Each flush is bounded by the chunk queue plus one file's chunks.
        assert max(embedder.batch_sizes) <= limits.max_queue_size + 10
        assert sum(embedder.batch_sizes) == stats.total_chunks

    async def test_incremental_reindex_skips_unchanged(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        _make_repo(repo, 10)
        embedder = RecordingEmbedder()
        limits = LimitsConfig(index_batch_size=5)
        work = tmp_path / "work"

        pipeline = _make_pipeline(work, limits, embedder)
        await pipeline.initialize()
        first = await pipeline.index_repository(repo, incremental=True)
        await pipeline.shutdown()
        assert first.files_indexed == 10

        embedder2 = RecordingEmbedder()
        pipeline2 = _make_pipeline(work, limits, embedder2)
        await pipeline2.initialize()
        second = await pipeline2.index_repository(repo, incremental=True)
        await pipeline2.shutdown()

        assert second.files_indexed == 0
        assert second.files_skipped == 10
        assert embedder2.batch_sizes == []  # nothing re-embedded

        # Touch one file → only that file is re-indexed.
        (repo / "src" / "module_003.py").write_text("def changed(): return 1\n")
        embedder3 = RecordingEmbedder()
        pipeline3 = _make_pipeline(work, limits, embedder3)
        await pipeline3.initialize()
        third = await pipeline3.index_repository(repo, incremental=True)
        await pipeline3.shutdown()
        assert third.files_indexed == 1
        assert third.files_skipped == 9


class TestCancellation:
    async def test_request_stop_halts_gracefully(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        _make_repo(repo, 30)
        embedder = RecordingEmbedder()
        limits = LimitsConfig(index_batch_size=5)
        pipeline = _make_pipeline(tmp_path / "work", limits, embedder)

        # Stop after the first embed flush.
        original = embedder.embed_documents_batched

        def stopping(texts, batch_size=None):
            result = original(texts, batch_size)
            pipeline.request_stop("test cancellation")
            return result

        embedder.embed_documents_batched = stopping  # type: ignore[method-assign]

        await pipeline.initialize()
        stats = await pipeline.index_repository(repo, incremental=True)
        await pipeline.shutdown()

        assert 0 < stats.files_indexed < 30
        assert any("stopped early" in e for e in stats.errors)

        # A follow-up run picks up the remaining files incrementally.
        embedder2 = RecordingEmbedder()
        pipeline2 = _make_pipeline(tmp_path / "work", limits, embedder2)
        await pipeline2.initialize()
        resumed = await pipeline2.index_repository(repo, incremental=True)
        await pipeline2.shutdown()
        assert resumed.files_indexed + stats.files_indexed >= 30
        assert not resumed.errors


class _StubMonitor(MemoryMonitor):
    """Memory monitor with a scripted pressure sequence."""

    def __init__(self, sequence: list[MemoryPressure]) -> None:
        super().__init__(soft_limit_mb=1, hard_limit_mb=2)
        self._sequence = list(sequence)

    def check(self) -> MemoryPressure:
        if self._sequence:
            return self._sequence.pop(0)
        return MemoryPressure.NORMAL


class TestMemoryPressure:
    async def test_hard_limit_stops_indexing(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        _make_repo(repo, 30)
        embedder = RecordingEmbedder()
        limits = LimitsConfig(index_batch_size=5)
        monitor = _StubMonitor([MemoryPressure.HARD])
        pipeline = _make_pipeline(tmp_path / "work", limits, embedder, monitor)

        await pipeline.initialize()
        stats = await pipeline.index_repository(repo, incremental=True)
        await pipeline.shutdown()

        assert stats.files_indexed < 30
        assert any("hard memory limit" in e for e in stats.errors)

    async def test_soft_limit_shrinks_batches_but_completes(
        self, tmp_path: Path
    ) -> None:
        repo = tmp_path / "repo"
        _make_repo(repo, 30)
        embedder = RecordingEmbedder()
        limits = LimitsConfig(index_batch_size=8)
        monitor = _StubMonitor([MemoryPressure.SOFT, MemoryPressure.SOFT])
        pipeline = _make_pipeline(tmp_path / "work", limits, embedder, monitor)

        await pipeline.initialize()
        stats = await pipeline.index_repository(repo, incremental=True)
        await pipeline.shutdown()

        # Soft pressure degrades throughput but never aborts the run.
        assert stats.files_indexed == 30
        assert not stats.errors


class TestLazyFileSources:
    def test_bounded_cache_and_full_iteration(self, tmp_path: Path) -> None:
        from datetime import datetime

        from local_context_engine.core.types import FileRecord, IndexingStatus, Language

        records = []
        for i in range(20):
            p = tmp_path / f"f{i}.py"
            p.write_text(f"content {i}")
            records.append(
                FileRecord(
                    id=str(i),
                    path=f"f{i}.py",
                    absolute_path=str(p),
                    language=Language.PYTHON,
                    size_bytes=10,
                    hash="x",
                    modified_at=datetime.now(tz=UTC),
                    status=IndexingStatus.NEW,
                )
            )

        sources = LazyFileSources(records, max_cached=4)
        # Reads succeed and iteration covers every file …
        seen = dict(sources.items())
        assert len(seen) == 20
        assert seen["f7.py"] == "content 7"
        # … while the cache never grows past its bound.
        assert len(sources._cache) <= 4
        # Missing files return the default instead of raising.
        assert sources.get("nope.py") == ""
