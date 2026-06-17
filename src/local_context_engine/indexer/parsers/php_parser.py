"""
PHP parser using Tree-sitter.

Extracts:
  - Classes (standard, abstract, final, readonly)
  - Interfaces / Traits / Enums
  - Functions and methods
  - Constants and class constants
  - Namespace declarations and use statements
  - Laravel-specific patterns: Controllers, Models, Services,
    Repositories, Policies, Middleware, Events, Listeners, Jobs,
    Commands, Requests, Resources, Routes, Migrations
  - Eloquent relationships (hasMany, hasOne, belongsTo, etc.)
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from local_context_engine.core.types import (
    Language,
    Relationship,
    RelationshipType,
    Symbol,
    SymbolType,
)
from local_context_engine.indexer.parsers.base_parser import BaseParser, ParseResult

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Laravel relationship method names → RelationshipType
_LARAVEL_REL_MAP: dict[str, RelationshipType] = {
    "hasOne": RelationshipType.HAS_ONE,
    "hasMany": RelationshipType.HAS_MANY,
    "belongsTo": RelationshipType.BELONGS_TO,
    "belongsToMany": RelationshipType.BELONGS_TO_MANY,
    "hasOneThrough": RelationshipType.HAS_ONE_THROUGH,
    "hasManyThrough": RelationshipType.HAS_MANY_THROUGH,
    "morphOne": RelationshipType.MORPH_ONE,
    "morphMany": RelationshipType.MORPH_MANY,
}

# Patterns to detect Laravel class types from class names / extends / implements
_LARAVEL_TYPE_HINTS: dict[str, SymbolType] = {
    "Controller": SymbolType.CONTROLLER,
    "Request": SymbolType.REQUEST,
    "Resource": SymbolType.RESOURCE,
    "Policy": SymbolType.POLICY,
    "Middleware": SymbolType.MIDDLEWARE,
    "Event": SymbolType.EVENT,
    "Listener": SymbolType.LISTENER,
    "Job": SymbolType.JOB,
    "Command": SymbolType.COMMAND,
    "Notification": SymbolType.NOTIFICATION,
    "Observer": SymbolType.OBSERVER,
    "Rule": SymbolType.RULE,
    "ServiceProvider": SymbolType.PROVIDER,
    "Seeder": SymbolType.SEEDER,
    "Factory": SymbolType.FACTORY,
}

# Migration: if filename starts with timestamp and contains _create_ or _add_
_MIGRATION_PATTERN = re.compile(r"\d{4}_\d{2}_\d{2}_\d{6}_", re.IGNORECASE)
_PHP_IMPORT_PATTERN = re.compile(
    r"^\s*use\s+(?!function\b|const\b)([^;]+);",
    re.MULTILINE,
)
_PHP_CLASS_HEADER_PATTERN = re.compile(
    r"(?m)^(?P<indent>\s*)"
    r"(?:(?:abstract|final|readonly)\s+)?"
    r"(?P<kind>class|interface|trait|enum)\s+"
    r"(?P<name>\w+)"
    r"(?:\s+extends\s+(?P<extends>[^\{\n]+?))?"
    r"(?:\s+implements\s+(?P<implements>[^\{\n]+?))?"
    r"\s*\{"
)
_PHP_CLASS_CONST_PATTERN = re.compile(
    r"^\s*(?:public|protected|private)?\s*const\s+([A-Z_][A-Z0-9_]*)\b",
    re.MULTILINE,
)
_PHP_PROPERTY_PATTERN = re.compile(
    r"^\s*(?:public|protected|private)\s+"
    r"(?:static\s+)?(?:readonly\s+)?"
    r"(?:[\w\\|?]+\s+)?\$(\w+)\b",
    re.MULTILINE,
)
_PHP_TRAIT_USE_PATTERN = re.compile(
    r"^\s*use\s+([A-Za-z_\\][\w\\]*)\s*;",
    re.MULTILINE,
)
_PHP_GLOBAL_CONST_PATTERN = re.compile(
    r"^\s*const\s+([A-Z_][A-Z0-9_]*)\b",
    re.MULTILINE,
)


def _infer_laravel_type(
    class_name: str,
    parent_class: str | None,
    file_path: str,
) -> SymbolType:
    """Heuristically infer the Laravel class type."""
    # Check filename for migration
    import os

    filename = os.path.basename(file_path)
    if _MIGRATION_PATTERN.match(filename):
        return SymbolType.MIGRATION

    # Check parent class
    if parent_class:
        for hint, sym_type in _LARAVEL_TYPE_HINTS.items():
            if hint in parent_class:
                return sym_type
        if "Model" in parent_class or parent_class.endswith("Model"):
            return SymbolType.MODEL

    # Check class name suffix
    for hint, sym_type in _LARAVEL_TYPE_HINTS.items():
        if class_name.endswith(hint):
            return sym_type

    # Path-based hints
    path_lower = file_path.lower()
    if "/models/" in path_lower:
        return SymbolType.MODEL
    if "/services/" in path_lower:
        return SymbolType.SERVICE
    if "/repositories/" in path_lower:
        return SymbolType.REPOSITORY
    if "/controllers/" in path_lower:
        return SymbolType.CONTROLLER

    return SymbolType.CLASS


class PHPParser(BaseParser):
    """
    PHP code parser using Tree-sitter.

    Falls back to regex-based extraction if Tree-sitter is unavailable.
    """

    def __init__(self) -> None:
        self._ts_available = False
        self._parser = None
        self._language = None
        self._init_treesitter()

    def _init_treesitter(self) -> None:
        try:
            import tree_sitter_php as tsphp
            from tree_sitter import Language, Parser

            self._language = Language(tsphp.language_php())
            self._parser = Parser(self._language)
            self._ts_available = True
            logger.debug("PHP Tree-sitter parser initialised.")
        except Exception as e:
            logger.warning("Tree-sitter PHP unavailable, using regex fallback: %s", e)

    @property
    def supported_language(self) -> Language:
        return Language.PHP

    @property
    def supported_extensions(self) -> list[str]:
        return [".php"]

    def parse(self, source: str, file_id: str, file_path: str) -> ParseResult:
        result = ParseResult(
            file_id=file_id,
            file_path=file_path,
            language=Language.PHP,
        )
        try:
            if self._ts_available:
                self._parse_treesitter(source, file_id, file_path, result)
            else:
                self._parse_regex(source, file_id, file_path, result)
            self._extract_supplemental_symbols(source, file_id, file_path, result)
        except Exception as exc:
            logger.error("PHP parse error in %s: %s", file_path, exc)
            result.errors.append(str(exc))
        return result

    # ── Tree-sitter parsing ────────────────────────────────────

    def _parse_treesitter(
        self, source: str, file_id: str, file_path: str, result: ParseResult
    ) -> None:
        tree = self._parser.parse(bytes(source, "utf-8"))
        root = tree.root_node

        namespace = self._extract_namespace(root)
        self._extract_classes(root, source, file_id, file_path, namespace, result)
        self._extract_functions(root, source, file_id, file_path, namespace, result)

    def _extract_namespace(self, root: "Node") -> str:  # type: ignore[name-defined]
        """Extract the namespace declaration from the parse tree."""
        query_str = "(namespace_definition name: (_) @ns)"
        try:
            query = self._language.query(query_str)
            captures = query.captures(root)
            if "ns" in captures and captures["ns"]:
                return self.safe_decode(captures["ns"][0].text)
        except Exception:
            pass
        return ""

    def _extract_classes(
        self,
        root: "Node",  # type: ignore[name-defined]
        source: str,
        file_id: str,
        file_path: str,
        namespace: str,
        result: ParseResult,
    ) -> None:
        """Extract class, interface, trait, and enum declarations."""
        node_types = [
            ("class_declaration", SymbolType.CLASS),
            ("interface_declaration", SymbolType.INTERFACE),
            ("trait_declaration", SymbolType.TRAIT),
            ("enum_declaration", SymbolType.ENUM),
        ]

        for node_type, base_sym_type in node_types:
            for node in self._find_nodes(root, node_type):
                class_name = self._get_child_text(node, "name", source)
                if not class_name:
                    continue

                qualified = f"{namespace}\\{class_name}" if namespace else class_name
                line_start = node.start_point[0] + 1
                line_end = node.end_point[0] + 1

                # Find parent class (extends)
                parent_name = None
                base_clause = self._get_first_child(node, "base_clause")
                if base_clause:
                    parent_name = self._get_node_text(base_clause, source).strip()
                    # Clean "extends ClassName" → "ClassName"
                    if parent_name.lower().startswith("extends"):
                        parent_name = parent_name[7:].strip()

                # Determine actual symbol type (Laravel-aware)
                sym_type = (
                    _infer_laravel_type(class_name, parent_name, file_path)
                    if base_sym_type == SymbolType.CLASS
                    else base_sym_type
                )

                # Docstring
                docstring = self._extract_docstring(node, source)

                sym_id = self.make_symbol_id(file_id, qualified, line_start)
                symbol = Symbol(
                    id=sym_id,
                    file_id=file_id,
                    file_path=file_path,
                    name=class_name,
                    qualified_name=qualified,
                    symbol_type=sym_type,
                    line_start=line_start,
                    line_end=line_end,
                    language=Language.PHP,
                    docstring=docstring,
                    metadata={"namespace": namespace, "parent": parent_name},
                )
                result.symbols.append(symbol)

                # Extract methods within this class
                self._extract_methods(node, source, file_id, file_path, sym_id, qualified, result)

                # Extract Eloquent relationships from methods
                if sym_type == SymbolType.MODEL:
                    self._extract_eloquent_relationships(
                        node, source, file_id, file_path, sym_id, qualified, result
                    )

                # Parent class relationship
                if parent_name:
                    result.relationships.append(
                        Relationship(
                            id=self.make_rel_id(sym_id, parent_name, "extends"),
                            source_symbol_id=sym_id,
                            source_file_path=file_path,
                            target_symbol_id=None,
                            target_name=parent_name,
                            relationship_type=RelationshipType.EXTENDS,
                        )
                    )

    def _extract_methods(
        self,
        class_node: "Node",  # type: ignore[name-defined]
        source: str,
        file_id: str,
        file_path: str,
        parent_sym_id: str,
        parent_qualified: str,
        result: ParseResult,
    ) -> None:
        """Extract method declarations within a class."""
        for node in self._find_nodes(class_node, "method_declaration"):
            method_name = self._get_child_text(node, "name", source)
            if not method_name:
                continue

            qualified = f"{parent_qualified}::{method_name}"
            visibility = self._get_visibility(node, source)
            line_start = node.start_point[0] + 1
            line_end = node.end_point[0] + 1
            docstring = self._extract_docstring(node, source)

            sym_id = self.make_symbol_id(file_id, qualified, line_start)
            sym_type = (
                SymbolType.CONSTRUCTOR if method_name == "__construct" else SymbolType.METHOD
            )

            result.symbols.append(
                Symbol(
                    id=sym_id,
                    file_id=file_id,
                    file_path=file_path,
                    name=method_name,
                    qualified_name=qualified,
                    symbol_type=sym_type,
                    line_start=line_start,
                    line_end=line_end,
                    language=Language.PHP,
                    visibility=visibility,
                    docstring=docstring,
                    parent_id=parent_sym_id,
                    parent_name=parent_qualified,
                )
            )

    def _extract_functions(
        self,
        root: "Node",  # type: ignore[name-defined]
        source: str,
        file_id: str,
        file_path: str,
        namespace: str,
        result: ParseResult,
    ) -> None:
        """Extract top-level function declarations."""
        for node in self._find_nodes(root, "function_definition"):
            # Skip if inside a class
            if self._has_ancestor(node, {"class_declaration", "trait_declaration"}):
                continue

            func_name = self._get_child_text(node, "name", source)
            if not func_name:
                continue

            qualified = f"{namespace}\\{func_name}" if namespace else func_name
            line_start = node.start_point[0] + 1
            line_end = node.end_point[0] + 1

            sym_id = self.make_symbol_id(file_id, qualified, line_start)
            result.symbols.append(
                Symbol(
                    id=sym_id,
                    file_id=file_id,
                    file_path=file_path,
                    name=func_name,
                    qualified_name=qualified,
                    symbol_type=SymbolType.FUNCTION,
                    line_start=line_start,
                    line_end=line_end,
                    language=Language.PHP,
                )
            )

    def _extract_eloquent_relationships(
        self,
        class_node: "Node",  # type: ignore[name-defined]
        source: str,
        file_id: str,
        file_path: str,
        model_sym_id: str,
        model_qualified: str,
        result: ParseResult,
    ) -> None:
        """Extract Eloquent relationship calls from a Model class."""
        rel_pattern = re.compile(
            r"\$this->(" + "|".join(_LARAVEL_REL_MAP.keys()) + r")\s*\(\s*([A-Za-z\\]+)::class",
            re.MULTILINE,
        )
        for match in rel_pattern.finditer(source):
            rel_method = match.group(1)
            target_model = match.group(2).split("\\")[-1]
            rel_type = _LARAVEL_REL_MAP.get(rel_method, RelationshipType.DEPENDS_ON)

            result.relationships.append(
                Relationship(
                    id=self.make_rel_id(model_sym_id, target_model, rel_method),
                    source_symbol_id=model_sym_id,
                    source_file_path=file_path,
                    target_symbol_id=None,
                    target_name=target_model,
                    relationship_type=rel_type,
                    metadata={"method": rel_method},
                )
            )

    # ── Regex fallback ────────────────────────────────────────

    def _parse_regex(
        self, source: str, file_id: str, file_path: str, result: ParseResult
    ) -> None:
        """Regex-based PHP symbol extraction (fallback when Tree-sitter unavailable)."""
        # Namespace
        ns_match = re.search(r"^namespace\s+([\w\\]+)\s*;", source, re.MULTILINE)
        namespace = ns_match.group(1) if ns_match else ""

        # Classes
        class_pattern = re.compile(
            r"^(?:abstract\s+|final\s+|readonly\s+)?class\s+(\w+)"
            r"(?:\s+extends\s+([\w\\]+))?",
            re.MULTILINE,
        )
        for match in class_pattern.finditer(source):
            class_name = match.group(1)
            parent_name = match.group(2)
            line_num = source[: match.start()].count("\n") + 1
            qualified = f"{namespace}\\{class_name}" if namespace else class_name
            sym_type = _infer_laravel_type(class_name, parent_name, file_path)
            sym_id = self.make_symbol_id(file_id, qualified, line_num)
            result.symbols.append(
                Symbol(
                    id=sym_id,
                    file_id=file_id,
                    file_path=file_path,
                    name=class_name,
                    qualified_name=qualified,
                    symbol_type=sym_type,
                    line_start=line_num,
                    line_end=line_num,
                    language=Language.PHP,
                )
            )

        # Functions
        func_pattern = re.compile(r"^(?:public|protected|private)?\s*function\s+(\w+)", re.MULTILINE)
        for match in func_pattern.finditer(source):
            func_name = match.group(1)
            line_num = source[: match.start()].count("\n") + 1
            sym_id = self.make_symbol_id(file_id, func_name, line_num)
            result.symbols.append(
                Symbol(
                    id=sym_id,
                    file_id=file_id,
                    file_path=file_path,
                    name=func_name,
                    qualified_name=f"{namespace}\\{func_name}" if namespace else func_name,
                    symbol_type=SymbolType.METHOD,
                    line_start=line_num,
                    line_end=line_num,
                    language=Language.PHP,
                )
            )

    def _extract_supplemental_symbols(
        self,
        source: str,
        file_id: str,
        file_path: str,
        result: ParseResult,
    ) -> None:
        """
        Add broader PHP coverage for imports, properties, constants, implements,
        and trait use relationships using source-level heuristics.
        """
        existing_symbol_ids = {symbol.id for symbol in result.symbols}
        existing_rel_ids = {relationship.id for relationship in result.relationships}

        class_symbols_by_line: dict[int, Symbol] = {
            symbol.line_start: symbol
            for symbol in result.symbols
            if symbol.symbol_type
            in {
                SymbolType.CLASS,
                SymbolType.INTERFACE,
                SymbolType.TRAIT,
                SymbolType.ENUM,
                SymbolType.CONTROLLER,
                SymbolType.MODEL,
                SymbolType.SERVICE,
                SymbolType.REPOSITORY,
                SymbolType.POLICY,
                SymbolType.MIDDLEWARE,
                SymbolType.EVENT,
                SymbolType.LISTENER,
                SymbolType.JOB,
                SymbolType.COMMAND,
                SymbolType.REQUEST,
                SymbolType.RESOURCE,
                SymbolType.NOTIFICATION,
                SymbolType.OBSERVER,
                SymbolType.RULE,
                SymbolType.PROVIDER,
                SymbolType.SEEDER,
                SymbolType.FACTORY,
                SymbolType.MIGRATION,
            }
        }

        # Top-level imports. Restrict to the region before the first class-like
        # declaration so trait "use" statements inside classes are not counted
        # as imports.
        first_class_like = _PHP_CLASS_HEADER_PATTERN.search(source)
        import_region = source[: first_class_like.start()] if first_class_like else source
        for match in _PHP_IMPORT_PATTERN.finditer(import_region):
            target = match.group(1).strip()
            target_name = target.split("\\")[-1].split(" as ")[0].strip()
            rel_id = self.make_rel_id(file_id, target, "imports")
            if rel_id not in existing_rel_ids:
                result.relationships.append(
                    Relationship(
                        id=rel_id,
                        source_symbol_id=file_id,
                        source_file_path=file_path,
                        target_symbol_id=None,
                        target_name=target_name,
                        relationship_type=RelationshipType.IMPORTS,
                        metadata={"import": target},
                    )
                )
                existing_rel_ids.add(rel_id)

        # Global constants.
        for match in _PHP_GLOBAL_CONST_PATTERN.finditer(import_region):
            const_name = match.group(1)
            line_start = source[: match.start()].count("\n") + 1
            sym_id = self.make_symbol_id(file_id, const_name, line_start)
            if sym_id in existing_symbol_ids:
                continue
            result.symbols.append(
                Symbol(
                    id=sym_id,
                    file_id=file_id,
                    file_path=file_path,
                    name=const_name,
                    qualified_name=const_name,
                    symbol_type=SymbolType.CONSTANT,
                    line_start=line_start,
                    line_end=line_start,
                    language=Language.PHP,
                )
            )
            existing_symbol_ids.add(sym_id)

        # Class-like blocks: implements, class constants, properties, trait uses.
        for match in _PHP_CLASS_HEADER_PATTERN.finditer(source):
            kind = match.group("kind")
            name = match.group("name")
            header_line = source[: match.start()].count("\n") + 1
            class_symbol = class_symbols_by_line.get(header_line)
            if class_symbol is None:
                class_symbol = next(
                    (symbol for symbol in result.symbols if symbol.name == name and symbol.line_start == header_line),
                    None,
                )

            block = self._extract_php_block(source, match.end() - 1)
            if block is None:
                continue
            body, _body_start, _body_end = block

            if class_symbol and kind == "class" and match.group("implements"):
                implements = [
                    part.strip()
                    for part in match.group("implements").split(",")
                    if part.strip()
                ]
                for interface_name in implements:
                    rel_id = self.make_rel_id(class_symbol.id, interface_name, "implements")
                    if rel_id not in existing_rel_ids:
                        result.relationships.append(
                            Relationship(
                                id=rel_id,
                                source_symbol_id=class_symbol.id,
                                source_file_path=file_path,
                                target_symbol_id=None,
                                target_name=interface_name.split("\\")[-1],
                                relationship_type=RelationshipType.IMPLEMENTS,
                            )
                        )
                        existing_rel_ids.add(rel_id)

            if class_symbol is None:
                continue

            body_start_line = source[: match.end()].count("\n") + 1
            for line_offset, line in enumerate(body.splitlines(), start=1):
                absolute_line = body_start_line + line_offset
                stripped = line.strip()
                if not stripped:
                    continue

                const_match = _PHP_CLASS_CONST_PATTERN.match(line)
                if const_match:
                    const_name = const_match.group(1)
                    sym_id = self.make_symbol_id(
                        file_id,
                        f"{class_symbol.qualified_name}::{const_name}",
                        absolute_line,
                    )
                    if sym_id not in existing_symbol_ids:
                        result.symbols.append(
                            Symbol(
                                id=sym_id,
                                file_id=file_id,
                                file_path=file_path,
                                name=const_name,
                                qualified_name=f"{class_symbol.qualified_name}::{const_name}",
                                symbol_type=SymbolType.CONSTANT,
                                line_start=absolute_line,
                                line_end=absolute_line,
                                language=Language.PHP,
                                parent_id=class_symbol.id,
                                parent_name=class_symbol.qualified_name,
                            )
                        )
                        existing_symbol_ids.add(sym_id)
                    continue

                property_match = _PHP_PROPERTY_PATTERN.match(line)
                if property_match:
                    property_name = property_match.group(1)
                    sym_id = self.make_symbol_id(
                        file_id,
                        f"{class_symbol.qualified_name}::{property_name}",
                        absolute_line,
                    )
                    if sym_id not in existing_symbol_ids:
                        result.symbols.append(
                            Symbol(
                                id=sym_id,
                                file_id=file_id,
                                file_path=file_path,
                                name=property_name,
                                qualified_name=f"{class_symbol.qualified_name}::{property_name}",
                                symbol_type=SymbolType.PROPERTY,
                                line_start=absolute_line,
                                line_end=absolute_line,
                                language=Language.PHP,
                                parent_id=class_symbol.id,
                                parent_name=class_symbol.qualified_name,
                            )
                        )
                        existing_symbol_ids.add(sym_id)
                    continue

                trait_match = _PHP_TRAIT_USE_PATTERN.match(line)
                if trait_match:
                    trait_name = trait_match.group(1)
                    rel_id = self.make_rel_id(class_symbol.id, trait_name, "uses")
                    if rel_id not in existing_rel_ids:
                        result.relationships.append(
                            Relationship(
                                id=rel_id,
                                source_symbol_id=class_symbol.id,
                                source_file_path=file_path,
                                target_symbol_id=None,
                                target_name=trait_name.split("\\")[-1],
                                relationship_type=RelationshipType.USES,
                            )
                        )
                        existing_rel_ids.add(rel_id)

    def _extract_php_block(
        self, source: str, start_index: int
    ) -> tuple[str, int, int] | None:
        """Return the brace-delimited block that starts at or after *start_index*."""
        open_index = source.find("{", start_index)
        if open_index == -1:
            return None

        depth = 0
        for idx in range(open_index, len(source)):
            char = source[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return source[open_index + 1 : idx], open_index, idx
        return None

    # ── Tree-sitter helpers ───────────────────────────────────

    def _find_nodes(self, root: "Node", node_type: str) -> list["Node"]:  # type: ignore[name-defined]
        """DFS traversal to find all nodes of a given type."""
        results: list["Node"] = []
        stack = [root]
        while stack:
            node = stack.pop()
            if node.type == node_type:
                results.append(node)
            stack.extend(reversed(node.children))
        return results

    def _get_child_text(self, node: "Node", field: str, source: str) -> str:  # type: ignore[name-defined]
        """Extract text of the named field child."""
        child = node.child_by_field_name(field)
        if child:
            return self.safe_decode(child.text)
        return ""

    def _get_node_text(self, node: "Node", source: str) -> str:  # type: ignore[name-defined]
        return self.safe_decode(node.text)

    def _get_first_child(self, node: "Node", child_type: str) -> "Node | None":  # type: ignore[name-defined]
        for child in node.children:
            if child.type == child_type:
                return child
        return None

    def _get_visibility(self, node: "Node", source: str) -> str:  # type: ignore[name-defined]
        for child in node.children:
            if child.type in {"public", "protected", "private"}:
                return self.safe_decode(child.text)
        return "public"

    def _has_ancestor(self, node: "Node", types: set[str]) -> bool:  # type: ignore[name-defined]
        current = node.parent
        while current:
            if current.type in types:
                return True
            current = current.parent
        return False

    def _extract_docstring(self, node: "Node", source: str) -> str | None:  # type: ignore[name-defined]
        """Extract a PHPDoc comment immediately before a declaration."""
        prev = node.prev_named_sibling
        if prev and prev.type == "comment":
            text = self.safe_decode(prev.text).strip()
            if text.startswith("/**"):
                return text
        return None
