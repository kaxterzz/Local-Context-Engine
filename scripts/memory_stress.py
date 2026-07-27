"""
Reproducible memory stress test for the Local Context Engine.

Generates a large fake repository, runs the indexing pipeline with a
deterministic fake embedder (no model download, no GPU), and samples
process RSS throughout. Also exercises the MCP warm-up path
(BM25 corpus build) which previously loaded every chunk into RAM.

Usage:
    .venv/bin/python scripts/memory_stress.py --files 600
    .venv/bin/python scripts/memory_stress.py --files 600 --tracemalloc

Exit code is non-zero if RSS growth exceeds --max-growth-mb, so this
can be wired into CI as a regression gate.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import random
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np
import psutil

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from local_context_engine.core.config import (  # noqa: E402
    ChunkingConfig,
    EngineConfig,
    MetadataStoreConfig,
    SymbolGraphConfig,
    VectorStoreConfig,
)
from local_context_engine.indexer.embedder.base_embedder import BaseEmbedder  # noqa: E402

_PHP_TEMPLATE = """<?php

namespace App\\Http\\Controllers;

use App\\Services\\{name}Service;
use App\\Models\\{name};

class {name}Controller extends Controller
{{
    public function __construct(private readonly {name}Service $service) {{}}

{methods}
}}
"""

_PHP_METHOD = """    /**
     * {doc}
     */
    public function {method}(Request $request)
    {{
        $items = $this->service->query()->where('status', '{status}');
        // {filler}
        return response()->json($items->paginate({page}));
    }}
"""

_TS_TEMPLATE = """import {{ useQuery }} from "@tanstack/react-query";
import {{ fetch{name} }} from "../api/{lower}";

export function use{name}() {{
    return useQuery({{ queryKey: ["{lower}"], queryFn: fetch{name} }});
}}

export function {name}List() {{
    const {{ data }} = use{name}();
    // {filler}
    return <ul>{{data?.map((x) => <li key={{x.id}}>{{x.name}}</li>)}}</ul>;
}}
"""


class FakeEmbedder(BaseEmbedder):
    """Deterministic, model-free embedder for stress testing."""

    def __init__(self, dimension: int = 384) -> None:
        self._dim = dimension

    @property
    def model_name(self) -> str:
        return "fake-stress-embedder"

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
        return self.embed_documents(texts)


def generate_fake_repo(root: Path, n_files: int, seed: int = 42, scale: int = 1) -> None:
    """Generate a fake repo. ``scale`` multiplies per-file content size."""
    random.seed(seed)
    (root / "app" / "Http" / "Controllers").mkdir(parents=True)
    (root / "src" / "components").mkdir(parents=True)
    for i in range(n_files):
        name = f"Entity{i:04d}"
        if i % 3 == 0:
            methods = "\n".join(
                _PHP_METHOD.format(
                    doc=f"Handle {name} action {j} with filters and pagination",
                    method=f"action{j}",
                    status=random.choice(["active", "pending", "archived"]),
                    filler=" ".join(
                        f"token{k}" for k in range(random.randint(30, 120) * scale)
                    ),
                    page=random.randint(10, 100),
                )
                for j in range(random.randint(4, 12) * scale)
            )
            path = root / "app" / "Http" / "Controllers" / f"{name}Controller.php"
            path.write_text(_PHP_TEMPLATE.format(name=name, methods=methods))
        else:
            filler = " ".join(
                f"word{k}" for k in range(random.randint(50, 300) * scale)
            )
            path = root / "src" / "components" / f"{name}.tsx"
            path.write_text(
                _TS_TEMPLATE.format(name=name, lower=name.lower(), filler=filler)
            )


def make_config(work_dir: Path) -> EngineConfig:
    return EngineConfig(
        vector_store=VectorStoreConfig(
            dimension=384, index_type="flat", storage_path=work_dir / "vectors"
        ),
        metadata_store=MetadataStoreConfig(db_path=work_dir / "metadata.db"),
        chunking=ChunkingConfig(min_tokens=20, target_tokens=200, max_tokens=500),
        symbol_graph=SymbolGraphConfig(persist=True, storage_path=work_dir / "graph.pkl"),
    )


class RSSSampler:
    def __init__(self) -> None:
        self._proc = psutil.Process()
        self.samples: list[tuple[float, float]] = []
        self._t0 = time.monotonic()
        self._stop = False

    def sample(self, label: str = "") -> float:
        rss = self._proc.memory_info().rss / 1e6
        self.samples.append((time.monotonic() - self._t0, rss))
        if label:
            print(f"    [rss] {rss:8.1f} MB  {label}")
        return rss

    async def run(self, interval: float = 0.25) -> None:
        while not self._stop:
            self.sample()
            await asyncio.sleep(interval)

    def stop(self) -> None:
        self._stop = True

    @property
    def peak(self) -> float:
        return max(r for _, r in self.samples) if self.samples else 0.0


async def run_stress(repo: Path, work_dir: Path, use_tracemalloc: bool) -> dict:
    from local_context_engine.indexer.chunkers.symbol_chunker import SymbolChunker
    from local_context_engine.indexer.parsers.registry import ParserRegistry
    from local_context_engine.indexer.pipeline import IndexingPipeline
    from local_context_engine.indexer.scanner.file_scanner import FileScanner
    from local_context_engine.metadata_store.database import Database
    from local_context_engine.retrieval.bm25_retriever import BM25Retriever
    from local_context_engine.retrieval.hybrid_retriever import HybridRetriever
    from local_context_engine.symbol_graph.graph import SymbolGraph
    from local_context_engine.vector_store.factory import VectorStoreFactory

    config = make_config(work_dir)
    embedder = FakeEmbedder()

    pipeline = IndexingPipeline(
        config=config,
        scanner=FileScanner.from_config(config),
        parser_registry=ParserRegistry.default(),
        chunker=SymbolChunker.from_config(config),
        database=Database.from_config(config.metadata_store),
        vector_store=VectorStoreFactory.create(config.vector_store),
        embedder=embedder,
        symbol_graph=SymbolGraph(),
    )

    sampler = RSSSampler()
    if use_tracemalloc:
        tracemalloc.start(10)

    gc.collect()
    baseline = sampler.sample("baseline before indexing")

    sampler_task = asyncio.create_task(sampler.run())
    t0 = time.monotonic()
    await pipeline.initialize()
    stats = await pipeline.index_repository(repo, incremental=True)
    await pipeline.shutdown()
    index_time = time.monotonic() - t0
    sampler.sample("after indexing")

    if use_tracemalloc:
        snap = tracemalloc.take_snapshot()
        print("\n  Top allocations after indexing:")
        for stat in snap.statistics("lineno")[:12]:
            print(f"    {stat.size / 1e6:8.1f} MB  {stat.traceback.format()[-1].strip()}")

    # ── MCP warm-up path: BM25 corpus build ────────────────────
    db = Database.from_config(config.metadata_store)
    await db.init()
    vs = VectorStoreFactory.create(config.vector_store)
    vs.load()
    retriever = HybridRetriever(
        embedder=embedder,
        vector_store=vs,
        bm25=BM25Retriever(),
        database=db,
        config=config.retrieval,
    )
    await retriever.initialize()
    sampler.sample("after MCP warm-up (BM25 build)")

    from local_context_engine.core.types import SearchQuery

    await retriever.search(SearchQuery(text="user authentication pagination", limit=10))
    sampler.sample("after search")
    await db.close()

    sampler.stop()
    await sampler_task
    gc.collect()
    final = sampler.sample("final after gc")

    return {
        "baseline_mb": baseline,
        "peak_mb": sampler.peak,
        "final_mb": final,
        "growth_mb": sampler.peak - baseline,
        "index_time_s": index_time,
        "files_indexed": stats.files_indexed,
        "chunks": stats.total_chunks,
        "vectors": stats.total_vectors,
        "errors": len(stats.errors),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=int, default=600)
    ap.add_argument("--scale", type=int, default=1,
                    help="Multiply per-file content size (simulate large legacy files).")
    ap.add_argument("--tracemalloc", action="store_true")
    ap.add_argument("--max-growth-mb", type=float, default=None,
                    help="Fail if peak RSS growth exceeds this many MB.")
    ap.add_argument("--keep", action="store_true", help="Keep the temp dirs.")
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="lce-stress-")
    repo = Path(tmp) / "repo"
    work = Path(tmp) / "work"
    repo.mkdir()
    work.mkdir()

    print(f"Generating fake repo with {args.files} files in {repo} …")
    generate_fake_repo(repo, args.files, scale=args.scale)
    total_bytes = sum(f.stat().st_size for f in repo.rglob("*") if f.is_file())
    print(f"  repo size: {total_bytes / 1e6:.1f} MB on disk")

    results = asyncio.run(run_stress(repo, work, args.tracemalloc))

    print("\n── Results ──────────────────────────────")
    for k, v in results.items():
        print(f"  {k:>16}: {v:.1f}" if isinstance(v, float) else f"  {k:>16}: {v}")

    if not args.keep:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)

    if args.max_growth_mb is not None and results["growth_mb"] > args.max_growth_mb:
        print(f"\nFAIL: RSS growth {results['growth_mb']:.1f} MB "
              f"> limit {args.max_growth_mb:.1f} MB")
        sys.exit(1)


if __name__ == "__main__":
    main()
