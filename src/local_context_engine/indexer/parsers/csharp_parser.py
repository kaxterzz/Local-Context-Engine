"""C# source parser with dependency-free symbol extraction."""

from __future__ import annotations

import re

from local_context_engine.core.types import (
    Language, Relationship, RelationshipType, Symbol, SymbolType,
)
from local_context_engine.indexer.parsers.base_parser import BaseParser, ParseResult


_TYPE_RE = re.compile(
    r"(?m)^\s*(?:(?:public|internal|private|protected|static|sealed|partial|abstract|readonly|ref)\s+)*"
    r"(?P<kind>class|interface|struct|record|enum)\s+(?P<name>[A-Za-z_]\w*)"
    r"(?:\s*<[^>{}]+>)?(?:\s*:\s*(?P<bases>[^\{\n]+))?"
)
_METHOD_RE = re.compile(
    r"(?m)^\s*(?P<mods>(?:(?:public|private|protected|internal|static|virtual|override|abstract|async|sealed|new|extern|partial)\s+)*)"
    r"(?P<return>[A-Za-z_][\w.<>,?\[\]\s:]*)\s+(?P<name>[A-Za-z_]\w*)\s*\([^;{}]*\)\s*(?:where[^\{=>]+)?(?:\{|=>)"
)
_NAMESPACE_RE = re.compile(r"(?m)^\s*namespace\s+([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)")
_USING_RE = re.compile(r"(?m)^\s*(?:global\s+)?using\s+(?:static\s+)?(?:[A-Za-z_]\w*\s*=\s*)?([\w.]+)\s*;")


class CSharpParser(BaseParser):
    @property
    def supported_language(self) -> Language:
        return Language.CSHARP

    @property
    def supported_extensions(self) -> list[str]:
        return [".cs"]

    def parse(self, source: str, file_id: str, file_path: str) -> ParseResult:
        result = ParseResult(file_id=file_id, file_path=file_path, language=Language.CSHARP)
        namespace_match = _NAMESPACE_RE.search(source)
        namespace = namespace_match.group(1) if namespace_match else ""

        for match in _TYPE_RE.finditer(source):
            name, kind = match.group("name"), match.group("kind")
            line = source.count("\n", 0, match.start()) + 1
            symbol_type = {
                "class": SymbolType.ABSTRACT_CLASS if "abstract" in match.group(0).split(kind)[0] else SymbolType.CLASS,
                "interface": SymbolType.INTERFACE,
                "enum": SymbolType.ENUM,
                "struct": SymbolType.CLASS,
                "record": SymbolType.CLASS,
            }[kind]
            qualified = f"{namespace}.{name}" if namespace else name
            symbol = self._symbol(file_id, file_path, name, qualified, symbol_type, line)
            symbol.metadata["declaration_kind"] = kind
            result.symbols.append(symbol)
            for index, base in enumerate((match.group("bases") or "").split(",")):
                target = re.sub(r"<.*>", "", base).strip()
                if not target:
                    continue
                rel_type = RelationshipType.EXTENDS if index == 0 and kind != "interface" else RelationshipType.IMPLEMENTS
                result.relationships.append(Relationship(
                    id=self.make_rel_id(symbol.id, target, rel_type.value), source_symbol_id=symbol.id,
                    source_file_path=file_path, target_symbol_id=None, target_name=target,
                    relationship_type=rel_type,
                ))

        type_spans = [(m.start(), m.group("name")) for m in _TYPE_RE.finditer(source)]
        for match in _METHOD_RE.finditer(source):
            name = match.group("name")
            if name in {"if", "for", "foreach", "while", "switch", "catch", "using", "lock"}:
                continue
            line = source.count("\n", 0, match.start()) + 1
            parent = next((n for pos, n in reversed(type_spans) if pos < match.start()), None)
            qualified = ".".join(filter(None, (namespace, parent, name)))
            visibility = next((v for v in ("public", "protected", "internal", "private") if v in match.group("mods").split()), None)
            result.symbols.append(self._symbol(
                file_id, file_path, name, qualified, SymbolType.CONSTRUCTOR if name == parent else SymbolType.METHOD,
                line, parent_name=parent, visibility=visibility,
            ))

        for match in _USING_RE.finditer(source):
            name = match.group(1)
            line = source.count("\n", 0, match.start()) + 1
            result.symbols.append(self._symbol(file_id, file_path, name, name, SymbolType.IMPORT, line))
        return result

    def _symbol(self, file_id: str, file_path: str, name: str, qualified: str,
                kind: SymbolType, line: int, **kwargs: object) -> Symbol:
        return Symbol(
            id=self.make_symbol_id(file_id, qualified, line), file_id=file_id, file_path=file_path,
            name=name, qualified_name=qualified, symbol_type=kind, line_start=line, line_end=line,
            language=Language.CSHARP, **kwargs,
        )
