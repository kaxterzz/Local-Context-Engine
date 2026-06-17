"""
Hybrid retrieval engine.

Combines four signals into a single ranked result list:

  final_score = semantic  × 0.50
              + bm25      × 0.20
              + symbol    × 0.20
              + graph     × 0.10

Each component is normalised to [0, 1] before weighting.

The number of candidate results fetched from each component is
``limit × candidate_multiplier`` (default 5×) to give the reranker
enough material to work with.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from local_context_engine.core.config import RetrievalConfig
from local_context_engine.core.types import SearchQuery, SearchResult
from local_context_engine.indexer.embedder.base_embedder import BaseEmbedder
from local_context_engine.metadata_store.database import Database
from local_context_engine.metadata_store.repositories import ChunkRepository
from local_context_engine.retrieval.bm25_retriever import BM25Retriever
from local_context_engine.security.redactor import ContentRedactor
from local_context_engine.vector_store.base_store import BaseVectorStore

logger = logging.getLogger(__name__)


@dataclass
class _ScoredCandidate:
    chunk_id: str
    semantic_score: float = 0.0
    bm25_score: float = 0.0
    symbol_score: float = 0.0
    graph_score: float = 0.0

    @property
    def final_score(
        self,
        w_sem: float = 0.50,
        w_bm25: float = 0.20,
        w_sym: float = 0.20,
        w_graph: float = 0.10,
    ) -> float:
        return (
            self.semantic_score * w_sem
            + self.bm25_score * w_bm25
            + self.symbol_score * w_sym
            + self.graph_score * w_graph
        )

    def compute_final(self, cfg: RetrievalConfig) -> float:
        return (
            self.semantic_score * cfg.semantic_weight
            + self.bm25_score * cfg.bm25_weight
            + self.symbol_score * cfg.symbol_weight
            + self.graph_score * cfg.graph_weight
        )


class HybridRetriever:
    """
    Combines semantic, BM25, symbol, and graph retrieval.

    Usage::

        retriever = HybridRetriever(embedder, vector_store, bm25, db, config)
        await retriever.initialize()
        results = await retriever.search(SearchQuery(text="journal entry approval"))
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        bm25: BM25Retriever,
        database: Database,
        config: RetrievalConfig,
        redactor: ContentRedactor | None = None,
        symbol_graph: "SymbolGraph | None" = None,  # type: ignore[name-defined]
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._bm25 = bm25
        self._database = database
        self._config = config
        self._redactor = redactor
        self._graph = symbol_graph

    async def initialize(self) -> None:
        """Build the BM25 corpus from the metadata store."""
        async with self._database.session() as session:
            chunk_repo = ChunkRepository(session)
            all_contents = await chunk_repo.get_all_contents()

        if all_contents:
            chunk_ids, texts = zip(*all_contents)
            self._bm25.add_documents(list(chunk_ids), list(texts))
            logger.info("BM25 corpus built: %d documents.", len(all_contents))
        else:
            logger.warning("No chunks found for BM25 index. Run 'context index' first.")

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        """
        Execute a hybrid search.

        Args:
            query: Structured :class:`SearchQuery` object.

        Returns:
            Ranked list of :class:`SearchResult` objects.
        """
        if self._vector_store.total_vectors == 0:
            logger.warning("Vector store is empty. Run 'context index' first.")
            return []

        candidates_count = min(
            query.limit * self._config.candidate_multiplier,
            self._config.max_limit * self._config.candidate_multiplier,
        )

        # ── 1. Semantic search ─────────────────────────────────
        loop = asyncio.get_event_loop()
        query_embedding = await loop.run_in_executor(
            None, self._embedder.embed_query, query.text
        )

        # Build metadata filter
        meta_filter: dict | None = None
        if query.language_filter:
            meta_filter = {"language": query.language_filter.value}

        semantic_results = self._vector_store.search_vectors(
            query_embedding, k=candidates_count, filter_metadata=meta_filter
        )

        # ── 2. BM25 keyword search ─────────────────────────────
        bm25_results = self._bm25.search(query.text, k=candidates_count)

        # ── 3. Symbol name search ──────────────────────────────
        symbol_scores: dict[str, float] = {}
        async with self._database.session() as session:
            from local_context_engine.metadata_store.repositories import (
                SymbolRepository,
            )

            sym_repo = SymbolRepository(session)
            matching_symbols = await sym_repo.find_by_name(query.text)
            if matching_symbols:
                # Boost only chunks from files that contain a name-matching symbol
                chunk_repo = ChunkRepository(session)
                for sym in matching_symbols[:10]:
                    if sym.file_path:
                        file_chunk_ids = await chunk_repo.get_ids_by_file_path(sym.file_path)
                        for chunk_id in file_chunk_ids:
                            symbol_scores[chunk_id] = min(
                                symbol_scores.get(chunk_id, 0.0) + 0.5, 1.0
                            )

        # ── 4. Graph proximity ─────────────────────────────────
        graph_scores: dict[str, float] = {}
        # (Placeholder: in a full implementation, this traverses the
        #  symbol graph from query-matching symbols to score related chunks)

        # ── Merge and rerank ───────────────────────────────────
        all_candidates: dict[str, _ScoredCandidate] = {}

        for sr in semantic_results:
            all_candidates.setdefault(sr.chunk_id, _ScoredCandidate(sr.chunk_id))
            all_candidates[sr.chunk_id].semantic_score = sr.score

        for br in bm25_results:
            all_candidates.setdefault(br.chunk_id, _ScoredCandidate(br.chunk_id))
            all_candidates[br.chunk_id].bm25_score = br.score

        for chunk_id, score in symbol_scores.items():
            all_candidates.setdefault(chunk_id, _ScoredCandidate(chunk_id))
            all_candidates[chunk_id].symbol_score = min(score, 1.0)

        # Score and rank
        scored = [
            (cid, cand.compute_final(self._config))
            for cid, cand in all_candidates.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Filter by minimum score
        scored = [
            (cid, score) for cid, score in scored
            if score >= self._config.min_score
        ]

        # Fetch top-k chunk details from metadata store
        top_chunk_ids = [cid for cid, _ in scored[: query.limit]]
        if not top_chunk_ids:
            return []

        async with self._database.session() as session:
            chunk_repo = ChunkRepository(session)
            results: list[SearchResult] = []

            for rank, (chunk_id, final_score) in enumerate(scored[: query.limit]):
                chunk = await chunk_repo.get_by_id(chunk_id)
                if chunk is None:
                    continue

                candidate = all_candidates[chunk_id]
                snippet = chunk.content if query.include_snippets else ""

                # Apply output redaction
                if self._redactor and snippet:
                    snippet = self._redactor.redact(snippet)

                # Apply path filter if set
                if query.path_filter:
                    import pathspec

                    spec = pathspec.PathSpec.from_lines("gitwildmatch", [query.path_filter])
                    if not spec.match_file(chunk.file_path):
                        continue

                results.append(
                    SearchResult(
                        chunk_id=chunk_id,
                        file_path=chunk.file_path,
                        symbol_name=chunk.symbol_name,
                        symbol_type=chunk.symbol_type.value if chunk.symbol_type else None,
                        language=chunk.language.value,
                        snippet=snippet,
                        line_start=chunk.line_start,
                        line_end=chunk.line_end,
                        score=final_score,
                        score_breakdown={
                            "semantic": candidate.semantic_score,
                            "bm25": candidate.bm25_score,
                            "symbol": candidate.symbol_score,
                            "graph": candidate.graph_score,
                        },
                    )
                )

        return results

    @classmethod
    def from_config(
        cls,
        config: "EngineConfig",  # type: ignore[name-defined]
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        database: Database,
        redactor: ContentRedactor | None = None,
        symbol_graph: "SymbolGraph | None" = None,  # type: ignore[name-defined]
    ) -> "HybridRetriever":
        bm25 = BM25Retriever()
        return cls(
            embedder=embedder,
            vector_store=vector_store,
            bm25=bm25,
            database=database,
            config=config.retrieval,
            redactor=redactor,
            symbol_graph=symbol_graph,
        )
