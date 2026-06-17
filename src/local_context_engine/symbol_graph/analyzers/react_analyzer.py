"""
React / TanStack ecosystem relationship inference.

Infers relationships that cannot be extracted from AST alone:
  - Component → Hook usage
  - Hook → useQuery/useMutation API endpoint
  - Context Provider → Consumer components
  - Page → Component composition
  - TanStack Router route → Page component
"""

from __future__ import annotations

import logging
import re

from local_context_engine.core.types import Relationship, RelationshipType, Symbol, SymbolType
from local_context_engine.indexer.parsers.base_parser import BaseParser

logger = logging.getLogger(__name__)

# Patterns for detecting JSX component usage: <ComponentName
_JSX_USAGE_PATTERN = re.compile(r"<([A-Z]\w+)[\s/>]")

# Context.Provider usage: <SomeContext.Provider
_CONTEXT_PROVIDER_PATTERN = re.compile(r"<(\w+Context)\.Provider")

# useContext(SomeContext) → consuming a context
_USE_CONTEXT_PATTERN = re.compile(r"useContext\s*\(\s*(\w+Context)\s*\)")

# API endpoint patterns in hooks
_API_ENDPOINT_PATTERN = re.compile(r"['\"`]/api/[a-z0-9/_-]+['\"`]", re.IGNORECASE)

# Direct function/hook calls: useLogin(), loginApi(credentials), etc.
_CALL_PATTERN = re.compile(r"(?<![\w$.])([A-Za-z_]\w*)\s*\(")

# TanStack Query/Mutation references: mutationFn: loginApi, queryFn: fetchUsers.
_QUERY_MUTATION_FN_PATTERN = re.compile(
    r"\b(?:queryFn|mutationFn)\s*:\s*([A-Za-z_]\w*)"
)

# TanStack Router: createRoute, createFileRoute
_TANSTACK_ROUTE_PATTERN = re.compile(
    r"createRoute\s*\(\s*\{[^}]*component\s*:\s*(\w+)"
)

_IGNORED_CALLS = {
    "Array",
    "Boolean",
    "Date",
    "Error",
    "Number",
    "Object",
    "Promise",
    "String",
    "fetch",
    "if",
    "map",
    "parseInt",
    "setTimeout",
    "switch",
}


class ReactAnalyzer:
    """
    Infers React/TanStack-specific relationships between indexed symbols.

    Call :meth:`analyze` after all files have been parsed.
    """

    def analyze(
        self,
        symbols: list[Symbol],
        file_sources: dict[str, str],
    ) -> list[Relationship]:
        """
        Generate React-specific relationships.

        Args:
            symbols:      All symbols in the repository.
            file_sources: Map of ``file_path → source_code``.

        Returns:
            List of inferred :class:`Relationship` objects.
        """
        relationships: list[Relationship] = []

        sym_by_name: dict[str, Symbol] = {s.name: s for s in symbols}

        # Component → Hook and Component → Component
        components = [
            s for s in symbols
            if s.symbol_type in (
                SymbolType.REACT_COMPONENT,
                SymbolType.REACT_HOOK,
                SymbolType.FUNCTION,
                SymbolType.ARROW_FUNCTION,
            )
        ]
        for component in components:
            source = _symbol_source(file_sources.get(component.file_path, ""), component)
            if not source:
                continue

            relationships.extend(
                self._analyze_calls(component, source, sym_by_name)
            )

            # JSX usage within the component
            for match in _JSX_USAGE_PATTERN.finditer(source):
                used_name = match.group(1)
                if used_name == component.name:
                    continue
                if used_name in sym_by_name:
                    target = sym_by_name[used_name]
                    relationships.append(
                        Relationship(
                            id=BaseParser.make_rel_id(component.id, used_name, "renders"),
                            source_symbol_id=component.id,
                            source_file_path=component.file_path,
                            target_symbol_id=target.id,
                            target_name=used_name,
                            relationship_type=RelationshipType.RENDERS,
                            confidence=0.9,
                            metadata={"inferred_by": "jsx_usage"},
                        )
                    )

            # Context consumption
            for match in _USE_CONTEXT_PATTERN.finditer(source):
                context_name = match.group(1)
                if context_name in sym_by_name:
                    target = sym_by_name[context_name]
                    relationships.append(
                        Relationship(
                            id=BaseParser.make_rel_id(component.id, context_name, "consumes"),
                            source_symbol_id=component.id,
                            source_file_path=component.file_path,
                            target_symbol_id=target.id,
                            target_name=context_name,
                            relationship_type=RelationshipType.CONSUMES,
                            confidence=1.0,
                            metadata={"inferred_by": "useContext"},
                        )
                    )

            # API endpoint usage in hooks
            if component.symbol_type == SymbolType.REACT_HOOK:
                for match in _API_ENDPOINT_PATTERN.finditer(source):
                    endpoint = match.group(0).strip("'\"`")
                    relationships.append(
                        Relationship(
                            id=BaseParser.make_rel_id(component.id, endpoint, "queries"),
                            source_symbol_id=component.id,
                            source_file_path=component.file_path,
                            target_symbol_id=None,
                            target_name=endpoint,
                            relationship_type=RelationshipType.QUERIES,
                            confidence=0.8,
                            metadata={"inferred_by": "api_endpoint", "endpoint": endpoint},
                        )
                    )

        # Context Provider → Providing
        for file_path, source in file_sources.items():
            for match in _CONTEXT_PROVIDER_PATTERN.finditer(source):
                context_name = match.group(1)
                if context_name in sym_by_name:
                    import uuid
                    provider_sym_id = str(
                        uuid.uuid5(uuid.NAMESPACE_URL, f"{file_path}::provider::{context_name}")
                    )
                    relationships.append(
                        Relationship(
                            id=BaseParser.make_rel_id(provider_sym_id, context_name, "provides"),
                            source_symbol_id=provider_sym_id,
                            source_file_path=file_path,
                            target_symbol_id=sym_by_name[context_name].id,
                            target_name=context_name,
                            relationship_type=RelationshipType.PROVIDES,
                            confidence=0.95,
                            metadata={"inferred_by": "context_provider_usage"},
                        )
                    )

        return relationships

    def _analyze_calls(
        self,
        symbol: Symbol,
        source: str,
        sym_by_name: dict[str, Symbol],
    ) -> list[Relationship]:
        """Infer direct calls/references between indexed frontend symbols."""
        relationships: list[Relationship] = []
        seen_targets: set[str] = set()

        target_names = [match.group(1) for match in _CALL_PATTERN.finditer(source)]
        target_names.extend(
            match.group(1)
            for match in _QUERY_MUTATION_FN_PATTERN.finditer(source)
        )

        for target_name in target_names:
            if (
                target_name == symbol.name
                or target_name in seen_targets
                or target_name in _IGNORED_CALLS
                or target_name not in sym_by_name
            ):
                continue

            target = sym_by_name[target_name]
            seen_targets.add(target_name)
            relationships.append(
                Relationship(
                    id=BaseParser.make_rel_id(symbol.id, target_name, "calls"),
                    source_symbol_id=symbol.id,
                    source_file_path=symbol.file_path,
                    target_symbol_id=target.id,
                    target_name=target_name,
                    relationship_type=RelationshipType.CALLS,
                    confidence=0.9,
                    metadata={"inferred_by": "direct_call"},
                )
            )

        return relationships


def _symbol_source(source: str, symbol: Symbol) -> str:
    """Return only the indexed symbol body when line bounds are available."""
    if not source or symbol.line_start <= 0 or symbol.line_end < symbol.line_start:
        return source

    lines = source.splitlines()
    start = max(symbol.line_start - 1, 0)
    end = min(symbol.line_end, len(lines))
    return "\n".join(lines[start:end])
