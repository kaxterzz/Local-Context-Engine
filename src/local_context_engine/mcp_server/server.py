"""
FastMCP-based MCP server for the Local Context Engine.

Exposes the following tools to AI coding assistants:

  search_codebase        — Hybrid semantic + keyword search
  search_symbol          — Find symbols by name/type
  read_chunk             — Fetch a specific chunk by ID
  find_references        — Who uses this symbol?
  find_dependencies      — What does this symbol depend on?
  trace_flow             — Trace execution path between symbols
  explain_architecture   — High-level codebase structure summary
  find_similar_code      — Find code patterns similar to a snippet
  recent_changes         — List recently modified files
  save_memory            — Persist agent knowledge
  retrieve_memory        — Recall agent knowledge
  get_stats              — Repository indexing statistics

All tools are read-only. The MCP server never modifies source files.
No data leaves the local machine.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from local_context_engine.core.config import EngineConfig, load_config
from local_context_engine.core.types import SearchQuery
from local_context_engine.indexer.embedder.factory import EmbedderFactory
from local_context_engine.memory.agent_memory import AgentMemory, MemoryCategory
from local_context_engine.metadata_store.database import Database
from local_context_engine.metadata_store.repositories import (
    ChunkRepository,
    FileRepository,
    SymbolRepository,
)
from local_context_engine.retrieval.bm25_retriever import BM25Retriever
from local_context_engine.retrieval.hybrid_retriever import HybridRetriever
from local_context_engine.security.pii_masker import PIIMasker
from local_context_engine.security.redactor import ContentRedactor
from local_context_engine.symbol_graph.graph import SymbolGraph
from local_context_engine.vector_store.factory import VectorStoreFactory

logger = logging.getLogger(__name__)


def create_mcp_server(
    repo_root: Path | None = None,
    config_path: Path | None = None,
) -> Any:
    """
    Create and configure a FastMCP server instance.

    Args:
        repo_root:   Root of the indexed repository. Defaults to CWD.
        config_path: Optional path to a YAML config override.

    Returns:
        A configured FastMCP server object ready to be run.
    """
    try:
        from fastmcp import FastMCP
    except ImportError as e:
        raise ImportError(
            "fastmcp is required for the MCP server. Run: pip install fastmcp"
        ) from e

    repo_root = repo_root or Path.cwd()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    config = load_config(project_root=repo_root, config_override=config_path)

    # ── Lightweight setup (no model loading yet) ──────────────
    database = Database.from_config(config.metadata_store)
    vector_store = VectorStoreFactory.create(config.vector_store)
    bm25 = BM25Retriever()
    masker = PIIMasker.from_config(config.pii_masking)
    redactor = ContentRedactor.from_masker(masker)
    symbol_graph = SymbolGraph()
    memory = AgentMemory.from_config(config.memory)

    # Keep initialization granular. Loading embeddings and BM25 can take
    # seconds, but many tools only need SQLite, memory, or the symbol graph.
    _embedder: list[Any] = [None]
    _retriever: list[Any] = [None]
    _metadata_initialized = [False]
    _vector_initialized = [False]
    _graph_initialized = [False]
    _memory_initialized = [False]
    _retrieval_initialized = [False]
    _metadata_lock = asyncio.Lock()
    _vector_lock = asyncio.Lock()
    _graph_lock = asyncio.Lock()
    _memory_lock = asyncio.Lock()
    _retrieval_lock = asyncio.Lock()

    async def _ensure_metadata_initialized() -> None:
        if _metadata_initialized[0]:
            return
        async with _metadata_lock:
            if _metadata_initialized[0]:
                return
            await database.init()
            _metadata_initialized[0] = True

    async def _ensure_vector_loaded() -> None:
        if _vector_initialized[0]:
            return
        async with _vector_lock:
            if _vector_initialized[0]:
                return
            await asyncio.to_thread(vector_store.load)
            _vector_initialized[0] = True

    async def _ensure_graph_loaded() -> None:
        if _graph_initialized[0]:
            return
        async with _graph_lock:
            if _graph_initialized[0]:
                return
            await asyncio.to_thread(symbol_graph.load, config.symbol_graph.storage_path)
            _graph_initialized[0] = True

    async def _ensure_memory_initialized() -> None:
        if _memory_initialized[0]:
            return
        async with _memory_lock:
            if _memory_initialized[0]:
                return
            await asyncio.to_thread(memory.init)
            _memory_initialized[0] = True

    async def _ensure_retrieval_initialized() -> None:
        if _retrieval_initialized[0]:
            return

        await _ensure_metadata_initialized()
        await _ensure_vector_loaded()
        await _ensure_graph_loaded()

        async with _retrieval_lock:
            if _retrieval_initialized[0]:
                return
            if _embedder[0] is None:
                _embedder[0] = await asyncio.to_thread(
                    EmbedderFactory.create,
                    config.embedding,
                )
            if _retriever[0] is None:
                _retriever[0] = HybridRetriever(
                    embedder=_embedder[0],
                    vector_store=vector_store,
                    bm25=bm25,
                    database=database,
                    config=config.retrieval,
                    redactor=redactor,
                    symbol_graph=symbol_graph,
                )
            await _retriever[0].initialize()
            _retrieval_initialized[0] = True

    async def _warm_up() -> None:
        await _ensure_retrieval_initialized()
        await _ensure_memory_initialized()

    # ── Background warmup via FastMCP lifespan ─────────────────
    # Start initialization as soon as the event loop is running so the
    # model is warm before the first tool call arrives (Claude Code sends
    # tool calls seconds after the MCP handshake, which would otherwise
    # block for 5–15 s and risk hitting the client timeout).

    @asynccontextmanager
    async def lifespan(_server: Any):  # type: ignore[return]
        warmup = asyncio.create_task(_warm_up())

        def _on_done(task: asyncio.Task) -> None:  # type: ignore[type-arg]
            if task.cancelled():
                return
            exc = task.exception()
            if exc is not None:
                logger.warning("MCP warm-up failed: %s", exc)
            else:
                logger.info("MCP server warm-up complete.")

        warmup.add_done_callback(_on_done)
        try:
            yield
        finally:
            warmup.cancel()
            try:
                await warmup
            except (asyncio.CancelledError, Exception):
                pass

    # ── Build FastMCP server ───────────────────────────────────
    mcp = FastMCP(
        name=config.mcp_server.name,
        version=config.mcp_server.version,
        lifespan=lifespan,
    )

    # ─────────────────────────────────────────────────────────
    # Tool: search_codebase
    # ─────────────────────────────────────────────────────────

    @mcp.tool()
    async def search_codebase(
        query: str,
        limit: int = 10,
        language: str | None = None,
        path_filter: str | None = None,
    ) -> list[dict]:
        """
        Search the codebase using hybrid semantic + keyword retrieval.

        Returns the most relevant code chunks with file paths, line numbers,
        symbol names, and relevance scores.

        Args:
            query:       Natural language or code query.
            limit:       Maximum number of results (1–50).
            language:    Filter by language: 'php', 'typescript', 'javascript', 'python'.
            path_filter: Glob pattern to restrict results (e.g. 'app/Http/**').
        """
        await _ensure_retrieval_initialized()

        from local_context_engine.core.types import Language  # noqa: PLC0415

        lang_filter = None
        if language:
            try:
                lang_filter = Language(language.lower())
            except ValueError:
                pass

        search_query = SearchQuery(
            text=query,
            limit=min(limit, config.retrieval.max_limit),
            language_filter=lang_filter,
            path_filter=path_filter,
        )

        results = await _retriever[0].search(search_query)
        return [
            {
                "chunk_id": r.chunk_id,
                "file_path": r.file_path,
                "symbol_name": r.symbol_name,
                "symbol_type": r.symbol_type,
                "language": r.language,
                "line_start": r.line_start,
                "line_end": r.line_end,
                "score": round(r.score, 4),
                "score_breakdown": {k: round(v, 4) for k, v in r.score_breakdown.items()},
                "snippet": r.snippet[:2000] if r.snippet else "",  # Truncate for safety
            }
            for r in results
        ]

    # ─────────────────────────────────────────────────────────
    # Tool: search_symbol
    # ─────────────────────────────────────────────────────────

    @mcp.tool()
    async def search_symbol(
        name: str,
        symbol_type: str | None = None,
        language: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """
        Find symbols (classes, functions, components) by name.

        Args:
            name:        Symbol name to search for (partial match supported).
            symbol_type: Filter by type: 'class', 'function', 'method',
                         'react_component', 'hook', 'controller', 'model', etc.
            language:    Filter by language.
            limit:       Maximum results.
        """
        await _ensure_metadata_initialized()

        async with database.session() as session:
            sym_repo = SymbolRepository(session)
            symbols = await sym_repo.find_by_name(name, language=language)

            if symbol_type:
                symbols = [s for s in symbols if s.symbol_type.value == symbol_type]

            return [
                {
                    "id": s.id,
                    "name": s.name,
                    "qualified_name": s.qualified_name,
                    "symbol_type": s.symbol_type.value,
                    "language": s.language.value,
                    "file_path": s.file_path,
                    "line_start": s.line_start,
                    "line_end": s.line_end,
                    "visibility": s.visibility,
                    "docstring": s.docstring,
                }
                for s in symbols[:limit]
            ]

    # ─────────────────────────────────────────────────────────
    # Tool: read_chunk
    # ─────────────────────────────────────────────────────────

    @mcp.tool()
    async def read_chunk(chunk_id: str) -> dict:
        """
        Fetch the full content of a specific code chunk by its ID.

        Use this after search_codebase to retrieve the full text of
        an interesting result.

        Args:
            chunk_id: The chunk_id returned by search_codebase.
        """
        await _ensure_metadata_initialized()

        async with database.session() as session:
            chunk_repo = ChunkRepository(session)
            chunk = await chunk_repo.get_by_id(chunk_id)

        if chunk is None:
            return {"error": f"Chunk '{chunk_id}' not found."}

        safe_content = redactor.redact(chunk.content)

        return {
            "chunk_id": chunk_id,
            "file_path": chunk.file_path,
            "language": chunk.language.value,
            "symbol_name": chunk.symbol_name,
            "symbol_type": chunk.symbol_type.value if chunk.symbol_type else None,
            "line_start": chunk.line_start,
            "line_end": chunk.line_end,
            "token_count": chunk.token_count,
            "content": safe_content,
        }

    # ─────────────────────────────────────────────────────────
    # Tool: find_references
    # ─────────────────────────────────────────────────────────

    @mcp.tool()
    async def find_references(symbol_name: str, depth: int = 2) -> list[dict]:
        """
        Find all symbols that reference (use, call, or import) the given symbol.

        Args:
            symbol_name: The symbol to look up (e.g. 'UserService', 'useAuth').
            depth:       How many graph hops to traverse (1–5).
        """
        await _ensure_graph_loaded()

        symbols = symbol_graph.lookup_by_name(symbol_name)
        if not symbols:
            return [{"message": f"Symbol '{symbol_name}' not found in graph."}]

        results = []
        for target in symbols[:3]:  # Up to 3 matching symbols
            refs = symbol_graph.find_references(target.id, depth=min(depth, 5))
            for ref_sym, rel_type, confidence in refs:
                results.append({
                    "name": ref_sym.name,
                    "qualified_name": ref_sym.qualified_name,
                    "symbol_type": ref_sym.symbol_type.value,
                    "file_path": ref_sym.file_path,
                    "line_start": ref_sym.line_start,
                    "relationship": rel_type,
                    "confidence": round(confidence, 2),
                })

        return results

    # ─────────────────────────────────────────────────────────
    # Tool: find_dependencies
    # ─────────────────────────────────────────────────────────

    @mcp.tool()
    async def find_dependencies(symbol_name: str, depth: int = 2) -> list[dict]:
        """
        Find all symbols that the given symbol depends on.

        Args:
            symbol_name: The symbol to analyse (e.g. 'OrderController').
            depth:       Traversal depth (1–5).
        """
        await _ensure_graph_loaded()

        symbols = symbol_graph.lookup_by_name(symbol_name)
        if not symbols:
            return [{"message": f"Symbol '{symbol_name}' not found in graph."}]

        results = []
        for source in symbols[:3]:
            deps = symbol_graph.find_dependencies(source.id, depth=min(depth, 5))
            for dep_sym, rel_type, confidence in deps:
                results.append({
                    "name": dep_sym.name,
                    "qualified_name": dep_sym.qualified_name,
                    "symbol_type": dep_sym.symbol_type.value,
                    "file_path": dep_sym.file_path,
                    "line_start": dep_sym.line_start,
                    "relationship": rel_type,
                    "confidence": round(confidence, 2),
                })

        return results

    # ─────────────────────────────────────────────────────────
    # Tool: trace_flow
    # ─────────────────────────────────────────────────────────

    @mcp.tool()
    async def trace_flow(from_symbol: str, to_symbol: str) -> dict:
        """
        Trace the execution or dependency path between two symbols.

        Useful for understanding how a user action flows through the system
        (e.g. from a Controller action to a database Model).

        Args:
            from_symbol: Starting symbol name.
            to_symbol:   Target symbol name.
        """
        await _ensure_graph_loaded()

        from_syms = symbol_graph.lookup_by_name(from_symbol)
        to_syms = symbol_graph.lookup_by_name(to_symbol)

        if not from_syms:
            return {"error": f"Symbol '{from_symbol}' not found."}
        if not to_syms:
            return {"error": f"Symbol '{to_symbol}' not found."}

        path = symbol_graph.trace_execution_path(from_syms[0].id, to_syms[0].id)
        if not path:
            return {
                "message": f"No path found from '{from_symbol}' to '{to_symbol}'.",
                "suggestion": "The symbols may be in different modules with no direct dependency.",
            }

        return {
            "from": from_symbol,
            "to": to_symbol,
            "path_length": len(path),
            "path": [
                {
                    "name": s.name,
                    "symbol_type": s.symbol_type.value,
                    "file_path": s.file_path,
                    "line_start": s.line_start,
                }
                for s in path
            ],
        }

    # ─────────────────────────────────────────────────────────
    # Tool: explain_architecture
    # ─────────────────────────────────────────────────────────

    @mcp.tool()
    async def explain_architecture() -> dict:
        """
        Return a high-level summary of the indexed codebase architecture.

        Includes file counts by language, top symbol types, and framework detection.
        """
        await _ensure_metadata_initialized()
        await _ensure_vector_loaded()
        await _ensure_graph_loaded()

        async with database.session() as session:
            file_repo = FileRepository(session)
            sym_repo = SymbolRepository(session)
            chunk_repo = ChunkRepository(session)

            total_files = await file_repo.count()
            lang_counts = await file_repo.language_counts()
            total_symbols = await sym_repo.count()
            total_chunks = await chunk_repo.count()

        return {
            "repository": str(repo_root),
            "total_files": total_files,
            "total_symbols": total_symbols,
            "total_chunks": total_chunks,
            "total_vectors": vector_store.total_vectors,
            "languages": lang_counts,
            "embedding_model": _embedder[0].model_name if _embedder[0] else "unknown",
            "symbol_graph": {
                "nodes": symbol_graph.node_count,
                "edges": symbol_graph.edge_count,
            },
        }

    # ─────────────────────────────────────────────────────────
    # Tool: find_similar_code
    # ─────────────────────────────────────────────────────────

    @mcp.tool()
    async def find_similar_code(code_snippet: str, limit: int = 5) -> list[dict]:
        """
        Find code that is semantically similar to the provided snippet.

        Useful for finding existing implementations of a pattern, detecting
        duplicated logic, or locating where a pattern is used.

        Args:
            code_snippet: A code snippet to search for similar patterns.
            limit:        Maximum results (1–20).
        """
        await _ensure_retrieval_initialized()

        search_query = SearchQuery(
            text=code_snippet,
            limit=min(limit, 20),
        )
        results = await _retriever[0].search(search_query)
        return [
            {
                "file_path": r.file_path,
                "symbol_name": r.symbol_name,
                "line_start": r.line_start,
                "line_end": r.line_end,
                "score": round(r.score, 4),
                "snippet": r.snippet[:1000] if r.snippet else "",
            }
            for r in results
        ]

    # ─────────────────────────────────────────────────────────
    # Tool: recent_changes
    # ─────────────────────────────────────────────────────────

    @mcp.tool()
    async def recent_changes(limit: int = 20) -> list[dict]:
        """
        List files most recently modified or indexed.

        Useful for understanding what has recently changed in the repository.

        Args:
            limit: Maximum number of files to return (1–100).
        """
        await _ensure_metadata_initialized()

        async with database.session() as session:
            file_repo = FileRepository(session)
            all_files = await file_repo.get_all()

        # Sort by indexed_at descending
        recent = sorted(
            [f for f in all_files if f.indexed_at],
            key=lambda f: f.indexed_at,  # type: ignore[return-value]
            reverse=True,
        )[: min(limit, 100)]

        return [
            {
                "path": f.path,
                "language": f.language.value,
                "line_count": f.line_count,
                "size_bytes": f.size_bytes,
                "indexed_at": f.indexed_at.isoformat() if f.indexed_at else None,
                "status": f.status.value,
            }
            for f in recent
        ]

    # ─────────────────────────────────────────────────────────
    # Tool: save_memory
    # ─────────────────────────────────────────────────────────

    @mcp.tool()
    async def save_memory(
        content: str,
        category: str = "general",
        tags: list[str] | None = None,
    ) -> dict:
        """
        Save a piece of architectural knowledge or coding convention.

        Use this to remember important decisions, patterns, or context
        across coding sessions.

        Args:
            content:  The knowledge to remember.
            category: One of: architecture, convention, decision, bug, api,
                      domain, tooling, general.
            tags:     Optional tags for later filtering.
        """
        await _ensure_memory_initialized()

        try:
            cat = MemoryCategory(category.lower())
        except ValueError:
            cat = MemoryCategory.GENERAL

        memory_id = memory.save_memory(category=cat, content=content, tags=tags)
        return {"memory_id": memory_id, "status": "saved", "category": cat.value}

    # ─────────────────────────────────────────────────────────
    # Tool: retrieve_memory
    # ─────────────────────────────────────────────────────────

    @mcp.tool()
    async def retrieve_memory(
        query: str,
        category: str | None = None,
        limit: int = 5,
    ) -> list[dict]:
        """
        Recall previously saved architectural knowledge.

        Args:
            query:    What you want to remember (keyword search).
            category: Optional category filter.
            limit:    Maximum results.
        """
        await _ensure_memory_initialized()

        cat = None
        if category:
            try:
                cat = MemoryCategory(category.lower())
            except ValueError:
                pass

        entries = memory.retrieve_memory(query=query, category=cat, limit=limit)
        return [
            {
                "id": e.id,
                "category": e.category.value,
                "content": e.content,
                "tags": e.tags,
                "created_at": e.created_at.isoformat(),
                "updated_at": e.updated_at.isoformat(),
            }
            for e in entries
        ]

    # ─────────────────────────────────────────────────────────
    # Tool: get_stats
    # ─────────────────────────────────────────────────────────

    @mcp.tool()
    async def get_stats() -> dict:
        """
        Return indexing statistics for the current repository.

        Useful for understanding what has been indexed and the health
        of the context engine.
        """
        await _ensure_metadata_initialized()
        await _ensure_vector_loaded()
        await _ensure_graph_loaded()
        return await explain_architecture()

    return mcp
