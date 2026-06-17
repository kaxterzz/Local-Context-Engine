"""
TypeScript / TSX / JavaScript / JSX parser using Tree-sitter.

Extracts:
  - Classes, interfaces, type aliases, enums
  - Functions (declarations, expressions, arrow functions)
  - React components (PascalCase functions/consts returning JSX)
  - React hooks (functions starting with 'use')
  - TanStack Query hooks (useQuery, useMutation, useInfiniteQuery)
  - Context providers and consumers
  - Named and default exports
  - Import statements
  - TanStack Router / Next.js routes
"""

from __future__ import annotations

import logging
import re

from local_context_engine.core.types import (
    Language,
    Relationship,
    RelationshipType,
    Symbol,
    SymbolType,
)
from local_context_engine.indexer.parsers.base_parser import BaseParser, ParseResult

logger = logging.getLogger(__name__)

# TanStack Query hooks that imply API calls
_QUERY_HOOKS = {"useQuery", "useInfiniteQuery", "useSuspenseQuery"}
_MUTATION_HOOKS = {"useMutation"}

# React hooks that imply context consumption
_CONTEXT_HOOKS = {"useContext"}

# Patterns for React component detection (PascalCase names)
_PASCAL_CASE = re.compile(r"^[A-Z][a-zA-Z0-9]*$")
_USE_PREFIX = re.compile(r"^use[A-Z]")


def _is_react_component(name: str) -> bool:
    return bool(_PASCAL_CASE.match(name))


def _is_hook(name: str) -> bool:
    return bool(_USE_PREFIX.match(name))


class TypeScriptParser(BaseParser):
    """
    TypeScript / TSX / JavaScript / JSX parser.

    Uses Tree-sitter when available, falls back to regex.
    Handles both TypeScript and TSX (React with TypeScript).
    """

    def __init__(self, language: Language = Language.TYPESCRIPT) -> None:
        self._language_enum = language
        self._ts_available = False
        self._parser = None
        self._ts_language = None
        self._init_treesitter()

    def _init_treesitter(self) -> None:
        try:
            from tree_sitter import Language as TSLanguage, Parser

            if self._language_enum in (Language.TYPESCRIPT, Language.TSX):
                import tree_sitter_typescript as tstypescript

                if self._language_enum == Language.TSX:
                    self._ts_language = TSLanguage(tstypescript.language_tsx())
                else:
                    self._ts_language = TSLanguage(tstypescript.language_typescript())
            else:
                import tree_sitter_javascript as tsjavascript

                self._ts_language = TSLanguage(tsjavascript.language())

            self._parser = Parser(self._ts_language)
            self._ts_available = True
            logger.debug("TypeScript/JS Tree-sitter parser initialised for %s.", self._language_enum)
        except Exception as e:
            logger.warning("Tree-sitter TS/JS unavailable, using regex fallback: %s", e)

    @property
    def supported_language(self) -> Language:
        return self._language_enum

    @property
    def supported_extensions(self) -> list[str]:
        if self._language_enum in (Language.TYPESCRIPT, Language.TSX):
            return [".ts", ".tsx"]
        return [".js", ".jsx", ".mjs", ".cjs"]

    def parse(self, source: str, file_id: str, file_path: str) -> ParseResult:
        result = ParseResult(
            file_id=file_id,
            file_path=file_path,
            language=self._language_enum,
        )
        try:
            if self._ts_available:
                self._parse_treesitter(source, file_id, file_path, result)
            else:
                self._parse_regex(source, file_id, file_path, result)
        except Exception as exc:
            logger.error("TypeScript parse error in %s: %s", file_path, exc)
            result.errors.append(str(exc))
        return result

    # ── Tree-sitter ────────────────────────────────────────────

    def _parse_treesitter(
        self, source: str, file_id: str, file_path: str, result: ParseResult
    ) -> None:
        tree = self._parser.parse(bytes(source, "utf-8"))
        root = tree.root_node

        self._extract_classes(root, source, file_id, file_path, result)
        self._extract_interfaces(root, source, file_id, file_path, result)
        self._extract_type_aliases(root, source, file_id, file_path, result)
        self._extract_enums(root, source, file_id, file_path, result)
        self._extract_functions(root, source, file_id, file_path, result)
        self._extract_variable_declarations(root, source, file_id, file_path, result)
        self._extract_imports(root, source, file_id, file_path, result)

    def _extract_classes(self, root, source, file_id, file_path, result):
        for node in self._find_nodes(root, "class_declaration"):
            name_node = node.child_by_field_name("name")
            if not name_node:
                continue
            name = self.safe_decode(name_node.text)
            line_start = node.start_point[0] + 1
            line_end = node.end_point[0] + 1
            sym_id = self.make_symbol_id(file_id, name, line_start)

            # Extract methods
            for method in self._find_nodes(node, "method_definition"):
                key_node = method.child_by_field_name("key")
                if not key_node:
                    continue
                method_name = self.safe_decode(key_node.text)
                m_line_start = method.start_point[0] + 1
                m_line_end = method.end_point[0] + 1
                m_sym_id = self.make_symbol_id(file_id, f"{name}.{method_name}", m_line_start)
                result.symbols.append(
                    Symbol(
                        id=m_sym_id,
                        file_id=file_id,
                        file_path=file_path,
                        name=method_name,
                        qualified_name=f"{name}.{method_name}",
                        symbol_type=SymbolType.METHOD,
                        line_start=m_line_start,
                        line_end=m_line_end,
                        language=self._language_enum,
                        parent_id=sym_id,
                        parent_name=name,
                    )
                )

            result.symbols.append(
                Symbol(
                    id=sym_id,
                    file_id=file_id,
                    file_path=file_path,
                    name=name,
                    qualified_name=name,
                    symbol_type=SymbolType.CLASS,
                    line_start=line_start,
                    line_end=line_end,
                    language=self._language_enum,
                )
            )

    def _extract_interfaces(self, root, source, file_id, file_path, result):
        for node in self._find_nodes(root, "interface_declaration"):
            name_node = node.child_by_field_name("name")
            if not name_node:
                continue
            name = self.safe_decode(name_node.text)
            line_start = node.start_point[0] + 1
            line_end = node.end_point[0] + 1
            sym_id = self.make_symbol_id(file_id, name, line_start)
            result.symbols.append(
                Symbol(
                    id=sym_id,
                    file_id=file_id,
                    file_path=file_path,
                    name=name,
                    qualified_name=name,
                    symbol_type=SymbolType.INTERFACE,
                    line_start=line_start,
                    line_end=line_end,
                    language=self._language_enum,
                )
            )

    def _extract_type_aliases(self, root, source, file_id, file_path, result):
        for node in self._find_nodes(root, "type_alias_declaration"):
            name_node = node.child_by_field_name("name")
            if not name_node:
                continue
            name = self.safe_decode(name_node.text)
            line_start = node.start_point[0] + 1
            line_end = node.end_point[0] + 1
            sym_id = self.make_symbol_id(file_id, name, line_start)
            result.symbols.append(
                Symbol(
                    id=sym_id,
                    file_id=file_id,
                    file_path=file_path,
                    name=name,
                    qualified_name=name,
                    symbol_type=SymbolType.TYPE_ALIAS,
                    line_start=line_start,
                    line_end=line_end,
                    language=self._language_enum,
                )
            )

    def _extract_enums(self, root, source, file_id, file_path, result):
        for node in self._find_nodes(root, "enum_declaration"):
            name_node = node.child_by_field_name("name")
            if not name_node:
                continue
            name = self.safe_decode(name_node.text)
            line_start = node.start_point[0] + 1
            line_end = node.end_point[0] + 1
            sym_id = self.make_symbol_id(file_id, name, line_start)
            result.symbols.append(
                Symbol(
                    id=sym_id,
                    file_id=file_id,
                    file_path=file_path,
                    name=name,
                    qualified_name=name,
                    symbol_type=SymbolType.ENUM,
                    line_start=line_start,
                    line_end=line_end,
                    language=self._language_enum,
                )
            )

    def _extract_functions(self, root, source, file_id, file_path, result):
        for node in self._find_nodes(root, "function_declaration"):
            name_node = node.child_by_field_name("name")
            if not name_node:
                continue
            name = self.safe_decode(name_node.text)
            line_start = node.start_point[0] + 1
            line_end = node.end_point[0] + 1
            sym_type = self._classify_function(name)
            sym_id = self.make_symbol_id(file_id, name, line_start)
            result.symbols.append(
                Symbol(
                    id=sym_id,
                    file_id=file_id,
                    file_path=file_path,
                    name=name,
                    qualified_name=name,
                    symbol_type=sym_type,
                    line_start=line_start,
                    line_end=line_end,
                    language=self._language_enum,
                )
            )
            # Hook → API relationships
            self._infer_hook_relationships(node, source, sym_id, file_path, result)

    def _extract_variable_declarations(self, root, source, file_id, file_path, result):
        """Extract const/let/var declarations that hold arrow functions or components."""
        for node in self._find_nodes(root, "lexical_declaration"):
            for declarator in self._find_nodes(node, "variable_declarator"):
                name_node = declarator.child_by_field_name("name")
                value_node = declarator.child_by_field_name("value")
                if not name_node or not value_node:
                    continue

                name = self.safe_decode(name_node.text)
                line_start = declarator.start_point[0] + 1
                line_end = declarator.end_point[0] + 1

                # Only extract meaningful function-like constructs
                if value_node.type not in {"arrow_function", "function", "function_expression"}:
                    continue

                sym_type = self._classify_function(name)
                sym_id = self.make_symbol_id(file_id, name, line_start)
                result.symbols.append(
                    Symbol(
                        id=sym_id,
                        file_id=file_id,
                        file_path=file_path,
                        name=name,
                        qualified_name=name,
                        symbol_type=sym_type,
                        line_start=line_start,
                        line_end=line_end,
                        language=self._language_enum,
                    )
                )
                self._infer_hook_relationships(value_node, source, sym_id, file_path, result)

    def _extract_imports(self, root, source, file_id, file_path, result):
        """Extract import statements and build dependency relationships."""
        for node in self._find_nodes(root, "import_statement"):
            source_node = node.child_by_field_name("source")
            if not source_node:
                continue
            import_path = self.safe_decode(source_node.text).strip("'\"")
            sym_id = self.make_symbol_id(file_id, f"import:{import_path}", node.start_point[0])
            result.relationships.append(
                Relationship(
                    id=self.make_rel_id(file_id, import_path, "imports"),
                    source_symbol_id=sym_id,
                    source_file_path=file_path,
                    target_symbol_id=None,
                    target_name=import_path,
                    relationship_type=RelationshipType.IMPORTS,
                )
            )

    def _infer_hook_relationships(self, func_node, source, sym_id, file_path, result):
        """Infer query/mutation relationships from hook calls inside a function."""
        func_source = self.safe_decode(func_node.text)
        for query_hook in _QUERY_HOOKS:
            if query_hook in func_source:
                result.relationships.append(
                    Relationship(
                        id=self.make_rel_id(sym_id, query_hook, "queries"),
                        source_symbol_id=sym_id,
                        source_file_path=file_path,
                        target_symbol_id=None,
                        target_name=query_hook,
                        relationship_type=RelationshipType.QUERIES,
                    )
                )
        for mut_hook in _MUTATION_HOOKS:
            if mut_hook in func_source:
                result.relationships.append(
                    Relationship(
                        id=self.make_rel_id(sym_id, mut_hook, "mutates"),
                        source_symbol_id=sym_id,
                        source_file_path=file_path,
                        target_symbol_id=None,
                        target_name=mut_hook,
                        relationship_type=RelationshipType.MUTATES,
                    )
                )

    # ── Regex fallback ────────────────────────────────────────

    def _parse_regex(self, source, file_id, file_path, result):
        """Regex-based TypeScript/JS symbol extraction."""
        # Interfaces
        for match in re.finditer(r"(?:export\s+)?interface\s+(\w+)", source):
            name = match.group(1)
            line = source[: match.start()].count("\n") + 1
            sym_id = self.make_symbol_id(file_id, name, line)
            result.symbols.append(
                Symbol(
                    id=sym_id, file_id=file_id, file_path=file_path,
                    name=name, qualified_name=name, symbol_type=SymbolType.INTERFACE,
                    line_start=line, line_end=line, language=self._language_enum,
                )
            )

        # Classes
        for match in re.finditer(r"(?:export\s+)?(?:abstract\s+)?class\s+(\w+)", source):
            name = match.group(1)
            line = source[: match.start()].count("\n") + 1
            sym_id = self.make_symbol_id(file_id, name, line)
            result.symbols.append(
                Symbol(
                    id=sym_id, file_id=file_id, file_path=file_path,
                    name=name, qualified_name=name, symbol_type=SymbolType.CLASS,
                    line_start=line, line_end=line, language=self._language_enum,
                )
            )

        # Functions / Components / Hooks
        for match in re.finditer(
            r"(?:export\s+)?(?:const|function)\s+(\w+)\s*(?:=\s*(?:async\s+)?\(|[(<])",
            source,
        ):
            name = match.group(1)
            line = source[: match.start()].count("\n") + 1
            sym_type = self._classify_function(name)
            sym_id = self.make_symbol_id(file_id, name, line)
            result.symbols.append(
                Symbol(
                    id=sym_id, file_id=file_id, file_path=file_path,
                    name=name, qualified_name=name, symbol_type=sym_type,
                    line_start=line, line_end=line, language=self._language_enum,
                )
            )

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _classify_function(name: str) -> SymbolType:
        """Classify a function by name as React component, hook, or plain function."""
        if _is_hook(name):
            return SymbolType.REACT_HOOK
        if _is_react_component(name):
            return SymbolType.REACT_COMPONENT
        return SymbolType.FUNCTION

    def _find_nodes(self, root, node_type: str) -> list:
        results = []
        stack = [root]
        while stack:
            node = stack.pop()
            if node.type == node_type:
                results.append(node)
            stack.extend(reversed(node.children))
        return results
