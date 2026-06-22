"""SQL DDL parser covering common database dialects."""

from __future__ import annotations

import re

from local_context_engine.core.types import Language, Symbol, SymbolType
from local_context_engine.indexer.parsers.base_parser import BaseParser, ParseResult


_DDL_RE = re.compile(
    r"(?im)^\s*create\s+(?:or\s+(?:alter|replace)\s+)?(?:temporary\s+)?"
    r"(?P<kind>table|view|procedure|proc|function|trigger|index)\s+"
    r"(?:if\s+not\s+exists\s+)?(?P<name>(?:[\[\`\"]?[A-Za-z_]\w*[\]\`\"]?\.)*[\[\`\"]?[A-Za-z_]\w*[\]\`\"]?)"
)


class SQLParser(BaseParser):
    @property
    def supported_language(self) -> Language:
        return Language.SQL

    @property
    def supported_extensions(self) -> list[str]:
        return [".sql"]

    def parse(self, source: str, file_id: str, file_path: str) -> ParseResult:
        result = ParseResult(file_id=file_id, file_path=file_path, language=Language.SQL)
        kinds = {
            "table": SymbolType.TABLE, "view": SymbolType.VIEW,
            "procedure": SymbolType.STORED_PROCEDURE, "proc": SymbolType.STORED_PROCEDURE,
            "function": SymbolType.FUNCTION, "trigger": SymbolType.FUNCTION,
            "index": SymbolType.UNKNOWN,
        }
        for match in _DDL_RE.finditer(source):
            qualified = match.group("name").replace("[", "").replace("]", "").replace("`", "").replace('"', "")
            name = qualified.rsplit(".", 1)[-1]
            line = source.count("\n", 0, match.start()) + 1
            result.symbols.append(Symbol(
                id=self.make_symbol_id(file_id, qualified, line), file_id=file_id, file_path=file_path,
                name=name, qualified_name=qualified, symbol_type=kinds[match.group("kind").lower()],
                line_start=line, line_end=line, language=Language.SQL,
                metadata={"ddl_kind": match.group("kind").lower()},
            ))
        return result
