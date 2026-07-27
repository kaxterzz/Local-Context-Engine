"""
Large-repository memory stability test.

Indexes a few hundred generated files and asserts that Python heap usage
(tracked with tracemalloc, which is deterministic unlike RSS) stays bounded —
i.e. memory reaches a plateau instead of growing with repository size.
"""

from __future__ import annotations

import gc
import tracemalloc
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
from local_context_engine.indexer.chunkers.symbol_chunker import SymbolChunker
from local_context_engine.indexer.embedder.base_embedder import BaseEmbedder
from local_context_engine.indexer.parsers.registry import ParserRegistry
from local_context_engine.indexer.pipeline import IndexingPipeline
from local_context_engine.indexer.scanner.file_scanner import FileScanner
from local_context_engine.metadata_store.database import Database
from local_context_engine.retrieval.bm25_retriever import BM25Retriever
from local_context_engine.retrieval.hybrid_retriever import HybridRetriever
from local_context_engine.symbol_graph.graph import SymbolGraph
from local_context_engine.vector_store.factory import VectorStoreFactory


class _FakeEmbedder(BaseEmbedder):
    def __init__(self, dimension: int = 64) -> None:
        self._dim = dimension

    @property
    def model_name(self) -> str:
        return "fake"

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

    def embed_documents_batched(self, texts, batch_size=None) -> np.ndarray:
        return self.embed_documents(texts)


def _generate_repo(root: Path, n_files: int) -> int:
    (root / "src").mkdir(parents=True)
    body = "\n".join(
        f"    field_{j} = 'value {j} with searchable tokens alpha beta'"
        for j in range(60)
    )
    total = 0
    for i in range(n_files):
        content = (
            f'"""Big module {i}."""\n\n\n'
            f"class BigService{i}:\n{body}\n\n"
            f"    def process_{i}(self, data):\n"
            f'        """Process incoming data batch number {i}."""\n'
            f"        return [x * {i} for x in data]\n"
        )
        (root / "src" / f"big_module_{i:04d}.py").write_text(content)
        total += len(content)
    return total


def _make_config(work: Path) -> EngineConfig:
    return EngineConfig(
        vector_store=VectorStoreConfig(
            dimension=64, index_type="flat", storage_path=work / "vectors"
        ),
        metadata_store=MetadataStoreConfig(db_path=work / "meta.db"),
        chunking=ChunkingConfig(min_tokens=10, target_tokens=150, max_tokens=400),
        security=SecurityConfig(never_index_patterns=[]),
        symbol_graph=SymbolGraphConfig(persist=True, storage_path=work / "graph.pkl"),
        limits=LimitsConfig(index_batch_size=10, max_queue_size=50),
    )


class TestLargeRepositoryMemoryStability:
    async def test_indexing_heap_stays_bounded(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        work = tmp_path / "work"
        work.mkdir()
        repo_bytes = _generate_repo(repo, 300)
        assert repo_bytes > 1_000_000  # a few MB of source

        config = _make_config(work)
        pipeline = IndexingPipeline(
            config=config,
            scanner=FileScanner.from_config(config),
            parser_registry=ParserRegistry.default(),
            chunker=SymbolChunker.from_config(config),
            database=Database.from_config(config.metadata_store),
            vector_store=VectorStoreFactory.create(config.vector_store),
            embedder=_FakeEmbedder(),
            symbol_graph=SymbolGraph(),
        )

        gc.collect()
        tracemalloc.start()
        baseline, _ = tracemalloc.get_traced_memory()

        await pipeline.initialize()
        stats = await pipeline.index_repository(repo, incremental=True)
        _, peak = tracemalloc.get_traced_memory()
        await pipeline.shutdown()
        tracemalloc.stop()

        assert stats.files_indexed == 300
        assert not stats.errors

        peak_growth_mb = (peak - baseline) / 1e6
        # The whole repo is several MB of source that previously ALL sat in
        # RAM at once (chunks ×2 copies + file_sources dict + full embedding
        # matrix). With streaming batches the peak heap must stay far below
        # repository size × copies. Generous CI-safe bound:
        assert peak_growth_mb < max(60.0, repo_bytes / 1e6 * 8), (
            f"peak tracemalloc growth {peak_growth_mb:.1f} MB suggests the "
            "pipeline is accumulating repository-sized state again"
        )

    async def test_bm25_warmup_heap_stays_bounded(self, tmp_path: Path) -> None:
        """MCP warm-up (BM25 build) must not materialise the whole corpus."""
        repo = tmp_path / "repo"
        work = tmp_path / "work"
        work.mkdir()
        _generate_repo(repo, 150)

        config = _make_config(work)
        pipeline = IndexingPipeline(
            config=config,
            scanner=FileScanner.from_config(config),
            parser_registry=ParserRegistry.default(),
            chunker=SymbolChunker.from_config(config),
            database=Database.from_config(config.metadata_store),
            vector_store=VectorStoreFactory.create(config.vector_store),
            embedder=_FakeEmbedder(),
            symbol_graph=SymbolGraph(),
        )
        await pipeline.initialize()
        await pipeline.index_repository(repo, incremental=True)
        await pipeline.shutdown()

        db = Database.from_config(config.metadata_store)
        await db.init()
        vs = VectorStoreFactory.create(config.vector_store)
        vs.load()
        retriever = HybridRetriever(
            embedder=_FakeEmbedder(),
            vector_store=vs,
            bm25=BM25Retriever(max_docs=100_000, max_tokens_per_doc=256),
            database=db,
            config=config.retrieval,
        )

        gc.collect()
        tracemalloc.start()
        baseline, _ = tracemalloc.get_traced_memory()
        await retriever.initialize()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_growth_mb = (peak - baseline) / 1e6
        assert peak_growth_mb < 60.0, (
            f"BM25 warm-up allocated {peak_growth_mb:.1f} MB peak — corpus "
            "streaming/token freeing has regressed"
        )

        from local_context_engine.core.types import SearchQuery

        results = await retriever.search(
            SearchQuery(text="searchable tokens alpha", limit=5)
        )
        assert results
        await db.close()
