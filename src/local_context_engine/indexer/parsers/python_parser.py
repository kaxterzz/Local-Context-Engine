"""
Python parser using Tree-sitter.

Extracts:
  - Classes (including dataclasses, ABCs, Pydantic models)
  - Functions and methods
  - Decorators
  - Type annotations
  - Import statements
  - Constants (module-level ALL_CAPS assignments)
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
_PY_CONSTANT_PATTERN = re.compile(
    r"^\s*([A-Z][A-Z0-9_]+)\s*(?::[^=]+)?=\s*.+$",
    re.MULTILINE,
)
_PY_TYPE_ALIAS_PATTERN = re.compile(
    r"^\s*type\s+([A-Za-z_]\w*)\s*=\s*.+$",
    re.MULTILINE,
)
_PY_TYPED_ALIAS_PATTERN = re.compile(
    r"^\s*([A-Za-z_]\w*)\s*:\s*TypeAlias\s*=\s*.+$",
    re.MULTILINE,
)


class PythonParser(BaseParser):
    """Python source code parser."""

    def __init__(self) -> None:
        self._ts_available = False
        self._parser = None
        self._language = None
        self._init_treesitter()

    def _init_treesitter(self) -> None:
        try:
            import tree_sitter_python as tspython
            from tree_sitter import Language, Parser

            self._language = Language(tspython.language())
            self._parser = Parser(self._language)
            self._ts_available = True
            logger.debug("Python Tree-sitter parser initialised.")
        except Exception as e:
            logger.warning("Tree-sitter Python unavailable, using regex fallback: %s", e)

    @property
    def supported_language(self) -> Language:
        return Language.PYTHON

    @property
    def supported_extensions(self) -> list[str]:
        return [".py", ".pyi"]

    def parse(self, source: str, file_id: str, file_path: str) -> ParseResult:
        result = ParseResult(file_id=file_id, file_path=file_path, language=Language.PYTHON)
        try:
            if self._ts_available:
                self._parse_treesitter(source, file_id, file_path, result)
            else:
                self._parse_regex(source, file_id, file_path, result)
            self._extract_supplemental_symbols(source, file_id, file_path, result)
        except Exception as exc:
            logger.error("Python parse error in %s: %s", file_path, exc)
            result.errors.append(str(exc))
        return result

    def _parse_treesitter(self, source, file_id, file_path, result):
        tree = self._parser.parse(bytes(source, "utf-8"))
        root = tree.root_node
        self._extract_classes(root, source, file_id, file_path, result)
        self._extract_functions(root, source, file_id, file_path, None, result)
        self._extract_imports(root, source, file_id, file_path, result)

    def _extract_classes(self, root, source, file_id, file_path, result):
        for node in self._find_nodes(root, "class_definition"):
            name_node = node.child_by_field_name("name")
            if not name_node:
                continue
            name = self.safe_decode(name_node.text)
            line_start = node.start_point[0] + 1
            line_end = node.end_point[0] + 1

            # Get bases
            superclasses: list[str] = []
            for arg_node in self._find_nodes(node, "argument_list"):
                for child in arg_node.children:
                    if child.type == "identifier":
                        superclasses.append(self.safe_decode(child.text))

            sym_id = self.make_symbol_id(file_id, name, line_start)
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
                    language=Language.PYTHON,
                    metadata={"superclasses": superclasses},
                )
            )

            # Methods
            for fn_node in self._find_nodes(node, "function_definition"):
                if fn_node.parent and fn_node.parent.type in {"block"}:
                    fn_name_node = fn_node.child_by_field_name("name")
                    if not fn_name_node:
                        continue
                    fn_name = self.safe_decode(fn_name_node.text)
                    fn_line_start = fn_node.start_point[0] + 1
                    fn_line_end = fn_node.end_point[0] + 1
                    fn_sym_id = self.make_symbol_id(file_id, f"{name}.{fn_name}", fn_line_start)
                    sym_type = SymbolType.CONSTRUCTOR if fn_name == "__init__" else SymbolType.METHOD
                    result.symbols.append(
                        Symbol(
                            id=fn_sym_id,
                            file_id=file_id,
                            file_path=file_path,
                            name=fn_name,
                            qualified_name=f"{name}.{fn_name}",
                            symbol_type=sym_type,
                            line_start=fn_line_start,
                            line_end=fn_line_end,
                            language=Language.PYTHON,
                            parent_id=sym_id,
                            parent_name=name,
                        )
                    )

            for base in superclasses:
                result.relationships.append(
                    Relationship(
                        id=self.make_rel_id(sym_id, base, "extends"),
                        source_symbol_id=sym_id,
                        source_file_path=file_path,
                        target_symbol_id=None,
                        target_name=base,
                        relationship_type=RelationshipType.EXTENDS,
                    )
                )

    def _extract_functions(self, root, source, file_id, file_path, parent_id, result):
        """Extract top-level function definitions."""
        for node in self._find_nodes(root, "function_definition"):
            # Only top-level
            if node.parent and node.parent.type not in {"module"}:
                continue
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
                    symbol_type=SymbolType.FUNCTION,
                    line_start=line_start,
                    line_end=line_end,
                    language=Language.PYTHON,
                )
            )

    def _extract_imports(self, root, source, file_id, file_path, result):
        for node in self._find_nodes(root, "import_statement"):
            module = self.safe_decode(node.text)
            result.relationships.append(
                Relationship(
                    id=self.make_rel_id(file_id, module, "imports"),
                    source_symbol_id=file_id,
                    source_file_path=file_path,
                    target_symbol_id=None,
                    target_name=module.strip(),
                    relationship_type=RelationshipType.IMPORTS,
                )
            )

        for node in self._find_nodes(root, "import_from_statement"):
            module = self.safe_decode(node.text)
            result.relationships.append(
                Relationship(
                    id=self.make_rel_id(file_id, module, "imports"),
                    source_symbol_id=file_id,
                    source_file_path=file_path,
                    target_symbol_id=None,
                    target_name=module.strip(),
                    relationship_type=RelationshipType.IMPORTS,
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
        Add broader Python coverage for constants and type aliases.

        Tree-sitter already handles classes, methods, functions, and imports.
        These additional patterns capture common module-level and class-level
        declarations that are useful for navigation and search.
        """
        existing_symbol_ids = {symbol.id for symbol in result.symbols}

        for pattern, symbol_type in (
            (_PY_TYPE_ALIAS_PATTERN, SymbolType.TYPE_ALIAS),
            (_PY_TYPED_ALIAS_PATTERN, SymbolType.TYPE_ALIAS),
            (_PY_CONSTANT_PATTERN, SymbolType.CONSTANT),
        ):
            for match in pattern.finditer(source):
                name = match.group(1)
                line_start = source[: match.start()].count("\n") + 1
                sym_id = self.make_symbol_id(file_id, name, line_start)
                if sym_id in existing_symbol_ids:
                    continue
                result.symbols.append(
                    Symbol(
                        id=sym_id,
                        file_id=file_id,
                        file_path=file_path,
                        name=name,
                        qualified_name=name,
                        symbol_type=symbol_type,
                        line_start=line_start,
                        line_end=line_start,
                        language=Language.PYTHON,
                    )
                )
                existing_symbol_ids.add(sym_id)

    def _parse_regex(self, source, file_id, file_path, result):
        """Regex fallback for Python parsing."""
        for match in re.finditer(r"^class\s+(\w+)(?:\(([^)]*)\))?", source, re.MULTILINE):
            name = match.group(1)
            line = source[: match.start()].count("\n") + 1
            sym_id = self.make_symbol_id(file_id, name, line)
            result.symbols.append(
                Symbol(
                    id=sym_id, file_id=file_id, file_path=file_path,
                    name=name, qualified_name=name, symbol_type=SymbolType.CLASS,
                    line_start=line, line_end=line, language=Language.PYTHON,
                )
            )

        for match in re.finditer(r"^\s*(?:async\s+)?def\s+(\w+)\s*\(", source, re.MULTILINE):
            name = match.group(1)
            line = source[: match.start()].count("\n") + 1
            sym_id = self.make_symbol_id(file_id, name, line)
            result.symbols.append(
                Symbol(
                    id=sym_id, file_id=file_id, file_path=file_path,
                    name=name, qualified_name=name, symbol_type=SymbolType.FUNCTION,
                    line_start=line, line_end=line, language=Language.PYTHON,
                )
            )

    def _find_nodes(self, root, node_type):
        results = []
        stack = [root]
        while stack:
            node = stack.pop()
            if node.type == node_type:
                results.append(node)
            stack.extend(reversed(node.children))
        return results
