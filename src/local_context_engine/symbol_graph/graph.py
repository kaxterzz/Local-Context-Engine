"""
Symbol relationship graph backed by NetworkX.

Models the codebase as a directed graph where:
  - Nodes are symbols (classes, functions, components, etc.)
  - Edges are typed relationships (extends, calls, imports, has_many, etc.)

Supports:
  - find_dependencies()    — what does symbol X depend on?
  - find_references()      — what uses symbol X?
  - find_callers()         — who calls symbol X?
  - find_callees()         — what does X call?
  - trace_execution_path() — traverse from entry point to leaf
  - Graph persistence to disk (pickle)
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

from local_context_engine.core.types import Relationship, RelationshipType, Symbol

logger = logging.getLogger(__name__)


class SymbolGraph:
    """
    Directed relationship graph for all indexed symbols.

    Uses ``networkx.DiGraph`` internally. Nodes are symbol IDs (strings);
    edges carry relationship type and confidence as attributes.
    """

    def __init__(self) -> None:
        try:
            import networkx as nx

            self._graph: Any = nx.DiGraph()
            self._nx = nx
        except ImportError as e:
            raise ImportError(
                "networkx is required for the symbol graph. Run: pip install networkx"
            ) from e

        # Symbol metadata store: symbol_id → Symbol
        self._symbols: dict[str, Symbol] = {}
        # Name lookup: qualified_name → symbol_id (may have duplicates)
        self._name_to_id: dict[str, list[str]] = {}

    # ── Graph building ─────────────────────────────────────────

    def add_symbol(self, symbol: Symbol) -> None:
        """Register a symbol as a graph node."""
        self._graph.add_node(
            symbol.id,
            name=symbol.name,
            qualified_name=symbol.qualified_name,
            symbol_type=symbol.symbol_type.value,
            language=symbol.language.value,
            file_path=symbol.file_path,
            line_start=symbol.line_start,
            line_end=symbol.line_end,
        )
        self._symbols[symbol.id] = symbol
        self._name_to_id.setdefault(symbol.name, []).append(symbol.id)
        self._name_to_id.setdefault(symbol.qualified_name, []).append(symbol.id)

    def add_symbols(self, symbols: list[Symbol]) -> None:
        for s in symbols:
            self.add_symbol(s)

    def add_relationship(self, rel: Relationship) -> None:
        """Add a directed edge for a relationship."""
        # Resolve target by name if not yet resolved
        target_id = rel.target_symbol_id
        if target_id is None:
            candidates = self._name_to_id.get(rel.target_name, [])
            if candidates:
                target_id = candidates[0]  # Take first match

        if target_id and target_id in self._graph:
            self._graph.add_edge(
                rel.source_symbol_id,
                target_id,
                relationship_type=rel.relationship_type.value,
                confidence=rel.confidence,
                metadata=rel.metadata,
            )
        else:
            # Store as a "dangling" edge with target name only
            # These get resolved in a second pass after full indexing
            self._graph.add_node(f"unresolved::{rel.target_name}", name=rel.target_name)
            self._graph.add_edge(
                rel.source_symbol_id,
                f"unresolved::{rel.target_name}",
                relationship_type=rel.relationship_type.value,
                confidence=rel.confidence,
                unresolved=True,
            )

    def add_relationships(self, relationships: list[Relationship]) -> None:
        for r in relationships:
            self.add_relationship(r)

    def resolve_dangling_edges(self) -> int:
        """
        Second-pass resolution of edges that could not be resolved during initial build.

        Returns the number of edges resolved.
        """
        resolved = 0
        unresolved_nodes = [
            n for n in self._graph.nodes if str(n).startswith("unresolved::")
        ]

        for unresolved in unresolved_nodes:
            target_name = str(unresolved)[len("unresolved::"):]
            candidates = self._name_to_id.get(target_name, [])
            if not candidates:
                continue

            target_id = candidates[0]
            if target_id not in self._graph:
                continue

            # Rewire edges
            for pred in list(self._graph.predecessors(unresolved)):
                edge_data = self._graph[pred][unresolved]
                self._graph.add_edge(pred, target_id, **edge_data)
                resolved += 1

            self._graph.remove_node(unresolved)

        logger.debug("Resolved %d dangling graph edges.", resolved)
        return resolved

    # ── Query API ──────────────────────────────────────────────

    def find_dependencies(
        self, symbol_id: str, depth: int = 3
    ) -> list[tuple[Symbol, str, float]]:
        """
        Return symbols that ``symbol_id`` depends on.

        Args:
            symbol_id: The source symbol.
            depth:     Maximum traversal depth.

        Returns:
            List of ``(symbol, relationship_type, confidence)`` tuples.
        """
        results: list[tuple[Symbol, str, float]] = []
        if symbol_id not in self._graph:
            return results

        visited = {symbol_id}
        queue = [(symbol_id, 0)]

        while queue:
            current_id, current_depth = queue.pop(0)
            if current_depth >= depth:
                continue

            for successor in self._graph.successors(current_id):
                if successor in visited or str(successor).startswith("unresolved::"):
                    continue
                visited.add(successor)
                edge = self._graph[current_id][successor]
                symbol = self._symbols.get(successor)
                if symbol:
                    results.append((
                        symbol,
                        edge.get("relationship_type", "unknown"),
                        edge.get("confidence", 1.0),
                    ))
                queue.append((successor, current_depth + 1))

        return results

    def find_references(
        self, symbol_id: str, depth: int = 3
    ) -> list[tuple[Symbol, str, float]]:
        """
        Return symbols that reference (depend on) ``symbol_id``.

        Args:
            symbol_id: The target symbol.
            depth:     Maximum traversal depth.

        Returns:
            List of ``(symbol, relationship_type, confidence)`` tuples.
        """
        results: list[tuple[Symbol, str, float]] = []
        if symbol_id not in self._graph:
            return results

        visited = {symbol_id}
        queue = [(symbol_id, 0)]

        while queue:
            current_id, current_depth = queue.pop(0)
            if current_depth >= depth:
                continue

            for predecessor in self._graph.predecessors(current_id):
                if predecessor in visited or str(predecessor).startswith("unresolved::"):
                    continue
                visited.add(predecessor)
                edge = self._graph[predecessor][current_id]
                symbol = self._symbols.get(predecessor)
                if symbol:
                    results.append((
                        symbol,
                        edge.get("relationship_type", "unknown"),
                        edge.get("confidence", 1.0),
                    ))
                queue.append((predecessor, current_depth + 1))

        return results

    def find_callers(self, symbol_id: str) -> list[Symbol]:
        """Symbols that call (CALLS edge) ``symbol_id``."""
        return [
            sym
            for sym, rel_type, _ in self.find_references(symbol_id)
            if rel_type == RelationshipType.CALLS.value
        ]

    def find_callees(self, symbol_id: str) -> list[Symbol]:
        """Symbols called by ``symbol_id``."""
        return [
            sym
            for sym, rel_type, _ in self.find_dependencies(symbol_id)
            if rel_type == RelationshipType.CALLS.value
        ]

    def trace_execution_path(
        self, from_symbol_id: str, to_symbol_id: str
    ) -> list[Symbol]:
        """
        Find the shortest dependency path between two symbols.

        Returns:
            Ordered list of symbols on the path, or empty list if no path exists.
        """
        try:
            path_ids = self._nx.shortest_path(
                self._graph, from_symbol_id, to_symbol_id
            )
            return [self._symbols[sid] for sid in path_ids if sid in self._symbols]
        except (self._nx.NetworkXNoPath, self._nx.NodeNotFound):
            return []

    def get_symbol(self, symbol_id: str) -> Symbol | None:
        return self._symbols.get(symbol_id)

    def lookup_by_name(self, name: str) -> list[Symbol]:
        ids = self._name_to_id.get(name, [])
        return [self._symbols[i] for i in ids if i in self._symbols]

    @property
    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    # ── Persistence ────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Persist the graph to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(
                {
                    "graph": self._graph,
                    "symbols": self._symbols,
                    "name_to_id": self._name_to_id,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        logger.info("Symbol graph saved: %d nodes, %d edges → %s", self.node_count, self.edge_count, path)

    def load(self, path: Path) -> bool:
        """Load the graph from disk. Returns ``True`` if successful."""
        if not path.exists():
            return False
        try:
            with path.open("rb") as f:
                data = pickle.load(f)
            self._graph = data["graph"]
            self._symbols = data["symbols"]
            self._name_to_id = data["name_to_id"]
            logger.info(
                "Symbol graph loaded: %d nodes, %d edges from %s",
                self.node_count,
                self.edge_count,
                path,
            )
            return True
        except Exception as exc:
            logger.warning("Failed to load symbol graph: %s", exc)
            return False

    def clear(self) -> None:
        self._graph.clear()
        self._symbols.clear()
        self._name_to_id.clear()
