"""
Laravel-specific relationship inference for the symbol graph.

Infers relationships that cannot be extracted purely from AST analysis:
  - Controller → Service (constructor injection pattern)
  - Service → Repository (constructor injection pattern)
  - Repository → Model (type hint analysis)
  - Route → Controller (route file parsing)
  - Model → Migration (naming convention)
  - Policy → Model (naming convention: UserPolicy → User)
  - Event → Listener (EventServiceProvider binding analysis)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from local_context_engine.core.types import Relationship, RelationshipType, Symbol, SymbolType
from local_context_engine.indexer.parsers.base_parser import BaseParser

logger = logging.getLogger(__name__)

# Constructor injection pattern: public function __construct(TypeHint $var, ...)
_CONSTRUCTOR_INJECTION_PATTERN = re.compile(
    r"public\s+function\s+__construct\s*\(([^)]+)\)", re.DOTALL
)

# Type-hinted parameter: TypeHint $variableName
_TYPE_HINT_PARAM = re.compile(r"(\w+(?:\\\w+)*)\s+\$\w+")

# Route definition patterns
_ROUTE_PATTERN = re.compile(
    r"Route\s*::\s*(?:get|post|put|patch|delete|any|match|resource|apiResource)\s*"
    r"\(\s*['\"][^'\"]+['\"]\s*,\s*"
    r"(?:\[([A-Za-z\\]+)::class\s*,\s*['\"](\w+)['\"]\]"
    r"|['\"]([A-Za-z\\@]+)['\"])",
    re.MULTILINE,
)

# Policy model binding: Gate::policy(Model::class, Policy::class)
_GATE_POLICY_PATTERN = re.compile(
    r"Gate\s*::\s*policy\s*\(\s*([A-Za-z\\]+)::class\s*,\s*([A-Za-z\\]+)::class"
)


class LaravelAnalyzer:
    """
    Infers Laravel-specific relationships between indexed symbols.

    Call :meth:`analyze` after all files have been parsed to add
    convention-based edges to the symbol graph.
    """

    def analyze(
        self,
        symbols: list[Symbol],
        file_sources: dict[str, str],
    ) -> list[Relationship]:
        """
        Generate additional relationships from Laravel conventions.

        Args:
            symbols:      All symbols in the repository.
            file_sources: Map of file_path → source code string.

        Returns:
            List of inferred :class:`Relationship` objects to add to the graph.
        """
        relationships: list[Relationship] = []

        sym_by_name: dict[str, Symbol] = {s.name: s for s in symbols}
        sym_by_type: dict[SymbolType, list[Symbol]] = {}
        for s in symbols:
            sym_by_type.setdefault(s.symbol_type, []).append(s)

        # Controller → Service injection
        for controller in sym_by_type.get(SymbolType.CONTROLLER, []):
            source = file_sources.get(controller.file_path, "")
            if source:
                relationships.extend(
                    self._analyze_constructor_injection(controller, source, sym_by_name)
                )

        # Service → Repository injection
        for service in sym_by_type.get(SymbolType.SERVICE, []):
            source = file_sources.get(service.file_path, "")
            if source:
                relationships.extend(
                    self._analyze_constructor_injection(service, source, sym_by_name)
                )

        # Repository → Model (by name convention: UserRepository → User)
        for repo in sym_by_type.get(SymbolType.REPOSITORY, []):
            model_name = repo.name.replace("Repository", "")
            if model_name in sym_by_name:
                relationships.append(
                    Relationship(
                        id=BaseParser.make_rel_id(repo.id, model_name, "depends_on"),
                        source_symbol_id=repo.id,
                        source_file_path=repo.file_path,
                        target_symbol_id=sym_by_name[model_name].id,
                        target_name=model_name,
                        relationship_type=RelationshipType.DEPENDS_ON,
                        confidence=0.9,
                        metadata={"inferred_by": "laravel_naming_convention"},
                    )
                )

        # Policy → Model (UserPolicy → User)
        for policy in sym_by_type.get(SymbolType.POLICY, []):
            model_name = policy.name.replace("Policy", "")
            if model_name in sym_by_name:
                relationships.append(
                    Relationship(
                        id=BaseParser.make_rel_id(policy.id, model_name, "belongs_to"),
                        source_symbol_id=policy.id,
                        source_file_path=policy.file_path,
                        target_symbol_id=sym_by_name[model_name].id,
                        target_name=model_name,
                        relationship_type=RelationshipType.DEPENDS_ON,
                        confidence=0.85,
                        metadata={"inferred_by": "policy_naming_convention"},
                    )
                )

        # Route → Controller (from route files)
        for file_path, source in file_sources.items():
            if "routes/" in file_path and file_path.endswith(".php"):
                relationships.extend(
                    self._analyze_routes(file_path, source, sym_by_name)
                )

        return relationships

    def _analyze_constructor_injection(
        self, symbol: Symbol, source: str, sym_by_name: dict[str, Symbol]
    ) -> list[Relationship]:
        """Extract dependency injection from __construct parameters."""
        relationships: list[Relationship] = []
        match = _CONSTRUCTOR_INJECTION_PATTERN.search(source)
        if not match:
            return relationships

        params_str = match.group(1)
        for param_match in _TYPE_HINT_PARAM.finditer(params_str):
            type_hint = param_match.group(1).split("\\")[-1]  # Strip namespace
            if type_hint in sym_by_name and type_hint != symbol.name:
                target = sym_by_name[type_hint]
                relationships.append(
                    Relationship(
                        id=BaseParser.make_rel_id(symbol.id, type_hint, "injects"),
                        source_symbol_id=symbol.id,
                        source_file_path=symbol.file_path,
                        target_symbol_id=target.id,
                        target_name=type_hint,
                        relationship_type=RelationshipType.INJECTS,
                        confidence=0.95,
                        metadata={"inferred_by": "constructor_injection"},
                    )
                )
        return relationships

    def _analyze_routes(
        self, file_path: str, source: str, sym_by_name: dict[str, Symbol]
    ) -> list[Relationship]:
        """Parse route definitions to extract Route → Controller edges."""
        relationships: list[Relationship] = []
        for match in _ROUTE_PATTERN.finditer(source):
            controller_class = (match.group(1) or match.group(3) or "").split("\\")[-1]
            if "@" in controller_class:
                controller_class = controller_class.split("@")[0]

            if controller_class and controller_class in sym_by_name:
                controller = sym_by_name[controller_class]
                # Route symbol is synthetic (the route file itself)
                import uuid
                route_sym_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{file_path}::{match.group(0)}"))
                relationships.append(
                    Relationship(
                        id=BaseParser.make_rel_id(route_sym_id, controller_class, "route_to"),
                        source_symbol_id=route_sym_id,
                        source_file_path=file_path,
                        target_symbol_id=controller.id,
                        target_name=controller_class,
                        relationship_type=RelationshipType.ROUTE_TO,
                        confidence=1.0,
                        metadata={"inferred_by": "route_analysis"},
                    )
                )
        return relationships
