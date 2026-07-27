"""
Indexing pipeline orchestrator.

Wires together: Scanner → Parser → Chunker → Embedder → Vector Store + Metadata Store.

Supports:
  - Full indexing (first run)
  - Incremental indexing (changed files only)
  - Background async operation
  - Progress callbacks
  - Graceful error handling per file (one bad file never stops the whole run)

Memory model
------------
The pipeline never holds the whole repository in RAM. Files are parsed,
chunked, embedded, and persisted in bounded batches
(``limits.index_batch_size`` files / ``limits.max_queue_size`` chunks at a
time); all references to a batch are dropped before the next one starts.
Between batches the RSS is checked against ``limits.memory_soft_limit_mb``
(shrink batches, collect garbage, warn) and ``limits.memory_hard_limit_mb``
(stop gracefully and persist what was completed).
"""

from __future__ import annotations

import asyncio
import gc
import logging
import time
import uuid
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from local_context_engine.core.config import EngineConfig
from local_context_engine.core.instance import (
    MemoryMonitor,
    MemoryPressure,
    log_process_status,
)
from local_context_engine.core.types import (
    FileRecord,
    IndexingProgress,
    IndexingStats,
    IndexingStatus,
)
from local_context_engine.indexer.chunkers.symbol_chunker import SymbolChunker
from local_context_engine.indexer.embedder.factory import EmbedderFactory
from local_context_engine.indexer.parsers.registry import ParserRegistry
from local_context_engine.indexer.scanner.file_scanner import FileScanner
from local_context_engine.metadata_store.database import Database
from local_context_engine.metadata_store.repositories import (
    ChunkRepository,
    FileRepository,
    RelationshipRepository,
    SymbolRepository,
)
from local_context_engine.symbol_graph.graph import SymbolGraph
from local_context_engine.symbol_graph.analyzers.laravel_analyzer import LaravelAnalyzer
from local_context_engine.symbol_graph.analyzers.react_analyzer import ReactAnalyzer
from local_context_engine.vector_store.factory import VectorStoreFactory

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[IndexingProgress], None]


class LazyFileSources:
    """
    Read-through, LRU-bounded ``{relative_path: source}`` mapping.

    The symbol-graph analyzers only ever look at one file at a time; loading
    the whole repository into a dict (the previous behaviour) made graph
    building consume as much RAM as the repository itself. This map keeps at
    most ``max_cached`` sources in memory.
    """

    def __init__(self, files: list[FileRecord], max_cached: int = 64) -> None:
        self._paths: dict[str, str] = {
            f.path: f.absolute_path for f in files
        }
        self._cache: dict[str, str] = {}
        self._order: list[str] = []
        self._max_cached = max(1, max_cached)

    def get(self, path: str, default: str = "") -> str:
        if path in self._cache:
            return self._cache[path]
        abs_path = self._paths.get(path)
        if abs_path is None:
            return default
        try:
            source = Path(abs_path).read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.debug("Cannot read %s for graph analysis: %s", path, exc)
            return default
        self._cache[path] = source
        self._order.append(path)
        while len(self._order) > self._max_cached:
            evicted = self._order.pop(0)
            self._cache.pop(evicted, None)
        return source

    def items(self) -> Iterator[tuple[str, str]]:
        """Yield every (path, source) pair without caching the full set."""
        for path in self._paths:
            yield path, self.get(path)

    def __contains__(self, path: str) -> bool:
        return path in self._paths

    def __len__(self) -> int:
        return len(self._paths)


class IndexingPipeline:
    """
    Orchestrates the full indexing workflow.

    Usage::

        pipeline = IndexingPipeline.from_config(config)
        await pipeline.initialize()
        stats = await pipeline.index_repository(Path("/path/to/repo"))
        await pipeline.shutdown()
    """

    def __init__(
        self,
        config: EngineConfig,
        scanner: FileScanner,
        parser_registry: ParserRegistry,
        chunker: SymbolChunker,
        database: Database,
        vector_store: "BaseVectorStore",  # type: ignore[name-defined]
        embedder: "BaseEmbedder",  # type: ignore[name-defined]
        symbol_graph: SymbolGraph | None = None,
        memory_monitor: MemoryMonitor | None = None,
    ) -> None:
        self._config = config
        self._scanner = scanner
        self._parser_registry = parser_registry
        self._chunker = chunker
        self._database = database
        self._vector_store = vector_store
        self._embedder = embedder
        self._symbol_graph = symbol_graph
        self._memory_monitor = memory_monitor or MemoryMonitor(
            soft_limit_mb=config.limits.memory_soft_limit_mb,
            hard_limit_mb=config.limits.memory_hard_limit_mb,
        )
        self._stop_requested = False
        self._stopped_reason: str | None = None

    def request_stop(self, reason: str = "stop requested") -> None:
        """Ask the pipeline to stop gracefully at the next batch boundary."""
        self._stop_requested = True
        self._stopped_reason = reason
        logger.warning("Indexing stop requested: %s", reason)

    async def initialize(self) -> None:
        """Initialise the database and load the vector index."""
        await self._database.init()
        self._vector_store.load()
        # Recover metadata after an interrupted run where vectors were saved
        # successfully but the final SQLite status update did not commit.
        stored_ids = self._vector_store.stored_chunk_ids
        if stored_ids:
            async with self._database.session() as session:
                chunk_repo = ChunkRepository(session)
                await chunk_repo.mark_embedded(list(stored_ids))
                await session.commit()
        logger.info("IndexingPipeline initialised.")

    async def shutdown(self) -> None:
        """Save the vector index, symbol graph, and close the database."""
        self._vector_store.save()
        if self._symbol_graph is not None and self._config.symbol_graph.persist:
            self._symbol_graph.save(self._config.symbol_graph.storage_path)
        await self._database.close()
        logger.info("IndexingPipeline shut down.")

    # ── Main entry points ─────────────────────────────────────

    async def index_repository(
        self,
        repo_root: Path,
        incremental: bool = True,
        progress_callback: ProgressCallback | None = None,
    ) -> IndexingStats:
        """
        Index (or re-index) an entire repository.

        Args:
            repo_root:          Root directory of the repository.
            incremental:        Skip files whose hash has not changed.
            progress_callback:  Optional callable invoked after each phase step.

        Returns:
            :class:`IndexingStats` summarising the completed run.
        """
        stats = IndexingStats()
        run_id = str(uuid.uuid4())
        total_start = time.monotonic()

        logger.info("Starting indexing run %s for %s", run_id, repo_root)

        try:
            # ── Phase 1: Scan ──────────────────────────────────────
            t0 = time.monotonic()
            if progress_callback:
                progress_callback(IndexingProgress(phase="scanning"))

            async with self._database.session() as session:
                file_repo = FileRepository(session)
                all_known = await file_repo.get_all()
                existing_hashes = {f.path: f.hash for f in all_known}
                known_paths = {f.path for f in all_known}
                del all_known

            file_records = await self._scanner.scan_repository(
                repo_root, existing_hashes=existing_hashes, incremental=incremental
            )
            del existing_hashes
            stats.total_files_scanned = len(file_records)
            stats.scan_time_seconds = time.monotonic() - t0

            # Detect deletions
            deleted_paths = await self._scanner.detect_deleted(repo_root, known_paths)
            if deleted_paths:
                await self._handle_deletions(deleted_paths)
                logger.info("Marked %d files as deleted.", len(deleted_paths))

            # Filter to only new / modified files
            files_to_index = [
                f for f in file_records
                if f.status in (IndexingStatus.NEW, IndexingStatus.MODIFIED)
            ]
            stats.files_skipped = sum(
                1 for f in file_records if f.status == IndexingStatus.UNCHANGED
            )
            del file_records
            logger.info(
                "%d files to index (%d unchanged, %d total).",
                len(files_to_index),
                stats.files_skipped,
                stats.total_files_scanned,
            )

            # ── Phase 2–4: Parse + Chunk + Embed in bounded batches ──
            t0 = time.monotonic()
            files_completed = await self._index_files_streaming(
                files_to_index, repo_root, stats, progress_callback
            )
            stats.parse_time_seconds = time.monotonic() - t0

            # ── Phase 5: Build symbol graph ───────────────────────
            if self._symbol_graph is not None and not self._stop_requested:
                if progress_callback:
                    progress_callback(IndexingProgress(phase="graph"))
                await self._build_symbol_graph()

            stats.total_time_seconds = time.monotonic() - total_start
            stats.total_vectors = self._vector_store.total_vectors
            stats.files_indexed = files_completed - stats.files_failed

            if self._stop_requested:
                stats.errors.append(
                    f"Indexing stopped early: {self._stopped_reason}. "
                    f"Completed {files_completed}/{len(files_to_index)} files; "
                    "re-run 'context index' to continue incrementally."
                )

            logger.info(
                "Indexing complete: %d indexed | %d skipped | %d failed | %.1fs",
                stats.files_indexed,
                stats.files_skipped,
                stats.files_failed,
                stats.total_time_seconds,
            )

        except Exception as exc:
            logger.exception("Indexing pipeline failed: %s", exc)
            stats.errors.append(str(exc))
            stats.total_time_seconds = time.monotonic() - total_start
            stats.total_vectors = self._vector_store.total_vectors

        return stats

    # ── Phase implementations ──────────────────────────────────

    async def _index_files_streaming(
        self,
        files: list[FileRecord],
        repo_root: Path,
        stats: IndexingStats,
        progress_callback: ProgressCallback | None,
    ) -> int:
        """
        Parse, chunk, persist, and embed files in bounded batches.

        Returns the number of files processed (may be less than ``len(files)``
        when a graceful stop was requested).
        """
        limits = self._config.limits
        batch_files = limits.index_batch_size
        max_chunk_buffer = limits.max_queue_size
        total = len(files)
        processed = 0
        pending_chunks: list[Any] = []
        batch_started = time.monotonic()
        files_in_batch = 0

        async def flush(reason: str) -> None:
            nonlocal pending_chunks, batch_started, files_in_batch
            if pending_chunks:
                embed_t0 = time.monotonic()
                await self._embed_chunks(pending_chunks, stats)
                stats.embedding_time_seconds += time.monotonic() - embed_t0
            log_process_status(
                logger,
                project=repo_root,
                phase="batch-complete",
                reason=reason,
                files_done=processed,
                files_total=total,
                batch_files=files_in_batch,
                batch_chunks=len(pending_chunks),
                queue_depth=0,
                batch_size=batch_files,
                duration_s=round(time.monotonic() - batch_started, 2),
            )
            pending_chunks = []
            files_in_batch = 0
            batch_started = time.monotonic()

        for file_record in files:
            # ── Stop / memory checks at batch boundaries ───────────
            if self._stop_requested:
                break
            if files_in_batch >= batch_files or len(pending_chunks) >= max_chunk_buffer:
                await flush("batch-full")
                pressure = self._memory_monitor.check()
                if pressure is MemoryPressure.HARD:
                    self.request_stop(
                        f"hard memory limit reached "
                        f"({self._memory_monitor.rss_mb:.0f} MB >= "
                        f"{self._memory_monitor.hard_limit_mb} MB)"
                    )
                    break
                if pressure is MemoryPressure.SOFT:
                    batch_files = max(1, batch_files // 2)
                    gc.collect()
                    logger.warning(
                        "Soft memory limit reached (%.0f MB >= %d MB): "
                        "batch size reduced to %d, caches flushed.",
                        self._memory_monitor.rss_mb,
                        self._memory_monitor.soft_limit_mb,
                        batch_files,
                    )
                elif batch_files < limits.index_batch_size:
                    batch_files = min(limits.index_batch_size, batch_files * 2)

            if progress_callback and processed % 10 == 0:
                progress_callback(
                    IndexingProgress(
                        phase="parsing",
                        total=total,
                        processed=processed,
                        current_file=file_record.path,
                    )
                )

            chunks = await self._parse_and_persist_file(file_record, stats)
            if chunks:
                pending_chunks.extend(chunks)
            processed += 1
            files_in_batch += 1

        if pending_chunks and not self._stop_requested:
            await flush("final")
        elif pending_chunks:
            # Graceful stop: still persist what was already parsed.
            await flush("stopped")

        return processed

    async def _parse_and_persist_file(
        self, file_record: FileRecord, stats: IndexingStats
    ) -> list[Any]:
        """Parse one file, persist symbols/chunks/relationships, return chunks."""
        try:
            source = Path(file_record.absolute_path).read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError as exc:
            logger.warning("Cannot read %s: %s", file_record.path, exc)
            stats.files_failed += 1
            stats.errors.append(f"Read error: {file_record.path}: {exc}")
            return []

        try:
            # Parse
            parse_result = self._parser_registry.parse(
                source, file_record.language, file_record.id, file_record.path
            )

            # Chunk — deduplicate by ID upfront
            raw_chunks = self._chunker.chunk(source, file_record, parse_result)
            chunks = list({c.id: c for c in raw_chunks}.values())
            del raw_chunks

            # Persist to metadata store
            async with self._database.session() as session:
                file_repo = FileRepository(session)
                sym_repo = SymbolRepository(session)
                chunk_repo = ChunkRepository(session)
                rel_repo = RelationshipRepository(session)

                # Always clean up stale data before inserting.
                # Handles both re-indexed (MODIFIED) files and NEW files
                # that have leftover rows from a previously interrupted run.
                await sym_repo.delete_by_file(file_record.id)
                await chunk_repo.delete_by_file(file_record.id)
                await rel_repo.delete_by_source_file(file_record.path)

                # Upsert file record
                file_record.indexed_at = datetime.now(tz=timezone.utc)
                file_record.line_count = source.count("\n") + 1
                await file_repo.upsert(file_record)

                # Insert symbols and chunks
                # Deduplicate by ID — minified files can yield duplicate
                # symbols when multiple one-letter names share line 1.
                inserted_symbol_ids: set[str] = set()
                if parse_result.symbols:
                    unique_symbols = list(
                        {s.id: s for s in parse_result.symbols}.values()
                    )
                    await sym_repo.insert_many(unique_symbols)
                    stats.total_symbols += len(unique_symbols)
                    inserted_symbol_ids = {s.id for s in unique_symbols}

                if chunks:
                    await chunk_repo.insert_many(chunks)

                if parse_result.relationships:
                    # Deduplicate by ID — PHP parser emits the same
                    # belongsTo/hasMany relationship multiple times when
                    # a model has several methods pointing to the same target.
                    valid_rels = list({
                        r.id: r for r in parse_result.relationships
                        if r.source_symbol_id in inserted_symbol_ids
                    }.values())
                    if valid_rels:
                        await rel_repo.insert_many(valid_rels)

                await session.commit()

            stats.total_chunks += len(chunks)
            return chunks

        except Exception as exc:
            logger.warning("Failed to index %s: %s", file_record.path, exc)
            stats.files_failed += 1
            stats.errors.append(f"Index error: {file_record.path}: {exc}")
            return []

    async def _build_symbol_graph(self) -> None:
        """Rebuild the symbol graph from the current metadata store contents."""
        if self._symbol_graph is None:
            return

        logger.info("Building symbol graph from indexed symbols…")
        async with self._database.session() as session:
            file_repo = FileRepository(session)
            sym_repo = SymbolRepository(session)
            rel_repo = RelationshipRepository(session)
            files = await file_repo.get_all()
            symbols = await sym_repo.get_all()
            relationships = await rel_repo.get_all()

        # Lazy, LRU-bounded source access — analyzers read one file at a
        # time; the whole repository is never resident in memory.
        file_sources = LazyFileSources(
            files, max_cached=self._config.limits.analyzer_source_cache_files
        )
        del files
        inferred_relationships = [
            *ReactAnalyzer().analyze(symbols, file_sources),
            *LaravelAnalyzer().analyze(symbols, file_sources),
        ]
        del file_sources
        relationships = list({
            rel.id: rel for rel in [*relationships, *inferred_relationships]
        }.values())
        del inferred_relationships

        self._symbol_graph.clear()
        self._symbol_graph.add_symbols(symbols)
        self._symbol_graph.add_relationships(relationships)
        resolved = self._symbol_graph.resolve_dangling_edges()

        logger.info(
            "Symbol graph built: %d nodes, %d edges (%d resolved).",
            self._symbol_graph.node_count,
            self._symbol_graph.edge_count,
            resolved,
        )

    async def _embed_chunks(self, chunks: list[Any], stats: IndexingStats) -> None:
        """Generate embeddings for a bounded batch of chunks and store them."""
        batch_size = self._config.performance.embedding_batch_size
        chunk_ids = [c.id for c in chunks]
        texts = [c.content_masked for c in chunks]  # Use PII-masked content
        metadata = [
            {
                "language": c.language.value,
                "symbol_name": c.symbol_name or "",
                "file_path": c.file_path,
            }
            for c in chunks
        ]

        logger.debug("Generating embeddings for %d chunks…", len(chunks))

        # Run embedding in thread pool (CPU-bound)
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self._embedder.embed_documents_batched(texts, batch_size),
        )
        del texts

        # Insert into vector store
        self._vector_store.insert_vectors(chunk_ids, embeddings, metadata)
        del embeddings, metadata
        stats.total_vectors = self._vector_store.total_vectors

        # Mark chunks as embedded in metadata store
        async with self._database.session() as session:
            chunk_repo = ChunkRepository(session)
            await chunk_repo.mark_embedded(chunk_ids)
            await session.commit()

        logger.debug("Embedded and stored %d vectors.", len(chunk_ids))

    async def _handle_deletions(self, deleted_paths: list[str]) -> None:
        """Remove deleted files from the metadata store."""
        async with self._database.session() as session:
            file_repo = FileRepository(session)
            for path in deleted_paths:
                await file_repo.mark_deleted(path)
            await session.commit()

    # ── Factory ────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config: EngineConfig) -> "IndexingPipeline":
        """Construct a fully-wired pipeline from the engine config."""
        scanner = FileScanner.from_config(config)
        parser_registry = ParserRegistry.default()
        chunker = SymbolChunker.from_config(config)
        database = Database.from_config(config.metadata_store)
        vector_store = VectorStoreFactory.create(config.vector_store)
        embedder = EmbedderFactory.create(config.embedding)
        symbol_graph = SymbolGraph() if config.symbol_graph.persist else None

        return cls(
            config=config,
            scanner=scanner,
            parser_registry=parser_registry,
            chunker=chunker,
            database=database,
            vector_store=vector_store,
            embedder=embedder,
            symbol_graph=symbol_graph,
        )
