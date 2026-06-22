"""ASP and ASP.NET template parser."""

from __future__ import annotations

import re

from local_context_engine.core.types import Language, Symbol, SymbolType
from local_context_engine.indexer.parsers.base_parser import BaseParser, ParseResult


class AspParser(BaseParser):
    def __init__(self, language: Language = Language.ASPNET) -> None:
        if language not in (Language.ASP, Language.ASPNET):
            raise ValueError("AspParser only supports ASP and ASP.NET")
        self._language = language

    @property
    def supported_language(self) -> Language:
        return self._language

    @property
    def supported_extensions(self) -> list[str]:
        return [".asp"] if self._language == Language.ASP else [".aspx", ".ascx", ".master", ".cshtml", ".razor"]

    def parse(self, source: str, file_id: str, file_path: str) -> ParseResult:
        result = ParseResult(file_id=file_id, file_path=file_path, language=self._language)
        patterns = [
            (r"<%@\s*(?:Page|Control|Master)\b[^%]*?\bInherits\s*=\s*[\"']([^\"']+)", SymbolType.CLASS),
            (r"(?im)^\s*(?:public\s+|private\s+|protected\s+)?(?:async\s+)?(?:sub|function)\s+([A-Za-z_]\w*)", SymbolType.METHOD),
            (r"(?m)^\s*@?(?:functions|code)\s*\{", SymbolType.UNKNOWN),
            (r"(?m)^\s*@page\s+[\"']([^\"']+)[\"']", SymbolType.ROUTE),
        ]
        seen: set[tuple[str, int]] = set()
        for pattern, kind in patterns:
            for match in re.finditer(pattern, source):
                name = match.group(1) if match.lastindex else "code"
                line = source.count("\n", 0, match.start()) + 1
                if (name, line) in seen:
                    continue
                seen.add((name, line))
                result.symbols.append(Symbol(
                    id=self.make_symbol_id(file_id, name, line), file_id=file_id, file_path=file_path,
                    name=name, qualified_name=name, symbol_type=kind, line_start=line, line_end=line,
                    language=self._language,
                ))

        # Extract C#-style methods commonly embedded in Razor and Web Forms script blocks.
        method_re = re.compile(r"(?m)^\s*(?:public|private|protected|internal)?\s*(?:static\s+)?(?:async\s+)?[\w.<>,?\[\]]+\s+([A-Za-z_]\w*)\s*\([^;]*\)\s*(?:\{|=>)")
        for match in method_re.finditer(source):
            name, line = match.group(1), source.count("\n", 0, match.start()) + 1
            if (name, line) not in seen:
                result.symbols.append(Symbol(
                    id=self.make_symbol_id(file_id, name, line), file_id=file_id, file_path=file_path,
                    name=name, qualified_name=name, symbol_type=SymbolType.METHOD,
                    line_start=line, line_end=line, language=self._language,
                ))
        return result
