"""
Integration tests for the full indexing pipeline.

These tests exercise the complete flow:
Scanner → Parser → Chunker → Embedder → Vector Store + Metadata Store
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_context_engine.core.config import (
    ChunkingConfig,
    EmbeddingConfig,
    EngineConfig,
    MetadataStoreConfig,
    PerformanceConfig,
    PIIMaskingConfig,
    RetrievalConfig,
    SecurityConfig,
    SymbolGraphConfig,
    VectorStoreConfig,
)
from local_context_engine.indexer.pipeline import IndexingPipeline


@pytest.fixture
def small_repo(tmp_path: Path) -> Path:
    """Create a tiny sample repository for pipeline testing."""
    # PHP
    php_dir = tmp_path / "app" / "Http" / "Controllers"
    php_dir.mkdir(parents=True)
    (php_dir / "UserController.php").write_text(
        """\
<?php
namespace App\\Http\\Controllers;

use App\\Services\\UserService;

class UserController extends Controller
{
    public function __construct(private UserService $userService) {}

    public function index()
    {
        return response()->json($this->userService->getAll());
    }
}
"""
    )

    # TypeScript
    ts_dir = tmp_path / "src" / "hooks"
    ts_dir.mkdir(parents=True)
    (ts_dir / "useUsers.ts").write_text(
        """\
import { useQuery } from "@tanstack/react-query";

export function useUsers() {
    return useQuery({
        queryKey: ["users"],
        queryFn: () => fetch("/api/users").then(r => r.json()),
    });
}
"""
    )

    # Python
    (tmp_path / "service.py").write_text(
        """\
class UserService:
    def get_all(self):
        return []

    def create(self, data):
        return data
"""
    )

    return tmp_path


@pytest.fixture
def pipeline_config(tmp_path: Path) -> EngineConfig:
    return EngineConfig(
        embedding=EmbeddingConfig(
            model="BAAI/bge-small-en-v1.5",
            device="cpu",
            batch_size=2,
            max_seq_length=128,
        ),
        vector_store=VectorStoreConfig(
            backend="faiss",
            dimension=384,
            index_type="flat",
            storage_path=tmp_path / "vectors",
        ),
        metadata_store=MetadataStoreConfig(
            db_path=tmp_path / "test.db",
        ),
        chunking=ChunkingConfig(
            min_tokens=10,
            target_tokens=100,
            max_tokens=300,
            overlap_tokens=10,
        ),
        security=SecurityConfig(enable_pii_masking=True, never_index_patterns=[]),
        pii_masking=PIIMaskingConfig(),
        performance=PerformanceConfig(incremental=True, scan_workers=1, parse_workers=1),
        symbol_graph=SymbolGraphConfig(persist=False),
    )


@pytest.mark.asyncio
@pytest.mark.slow
async def test_full_pipeline_runs(small_repo: Path, pipeline_config: EngineConfig) -> None:
    """Full pipeline integration test: index a small repo and verify stats."""
    pipeline = IndexingPipeline.from_config(pipeline_config)
    await pipeline.initialize()

    stats = await pipeline.index_repository(small_repo, incremental=False)
    await pipeline.shutdown()

    assert stats.total_files_scanned >= 3
    assert stats.files_indexed >= 1
    assert stats.files_failed == 0
    assert stats.total_chunks > 0


@pytest.mark.asyncio
@pytest.mark.slow
async def test_incremental_index_skips_unchanged(
    small_repo: Path, pipeline_config: EngineConfig
) -> None:
    """Second incremental run should skip unchanged files."""
    pipeline = IndexingPipeline.from_config(pipeline_config)
    await pipeline.initialize()

    # First run
    stats1 = await pipeline.index_repository(small_repo, incremental=True)
    # Second run (nothing changed)
    stats2 = await pipeline.index_repository(small_repo, incremental=True)
    await pipeline.shutdown()

    # On second run, all previously indexed files should be skipped
    assert stats2.files_skipped >= stats1.files_indexed
    assert stats2.files_indexed == 0 or stats2.files_failed == 0
