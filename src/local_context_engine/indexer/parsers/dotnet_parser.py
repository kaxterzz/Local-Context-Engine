"""Parser for .NET solution and MSBuild project metadata."""

from __future__ import annotations

import re

from local_context_engine.core.types import Language, Symbol, SymbolType
from local_context_engine.indexer.parsers.base_parser import BaseParser, ParseResult


class DotNetParser(BaseParser):
    @property
    def supported_language(self) -> Language:
        return Language.DOTNET

    @property
    def supported_extensions(self) -> list[str]:
        return [".sln", ".csproj", ".fsproj", ".vbproj", ".props", ".targets"]

    def parse(self, source: str, file_id: str, file_path: str) -> ParseResult:
        result = ParseResult(file_id=file_id, file_path=file_path, language=Language.DOTNET)
        patterns = [
            (r'(?im)^Project\([^\n]+?=\s*"([^"]+)"\s*,', SymbolType.PROJECT, "solution_project"),
            (r'<ProjectReference\b[^>]*\bInclude\s*=\s*["\']([^"\']+)', SymbolType.PROJECT, "project_reference"),
            (r'<PackageReference\b[^>]*\bInclude\s*=\s*["\']([^"\']+)', SymbolType.PACKAGE, "package_reference"),
            (r'<(?:TargetFramework|TargetFrameworks)>\s*([^<]+)', SymbolType.CONSTANT, "target_framework"),
        ]
        for pattern, kind, metadata_kind in patterns:
            for match in re.finditer(pattern, source):
                value = match.group(1).strip()
                line = source.count("\n", 0, match.start()) + 1
                result.symbols.append(Symbol(
                    id=self.make_symbol_id(file_id, value, line), file_id=file_id, file_path=file_path,
                    name=value.rsplit("\\", 1)[-1].rsplit("/", 1)[-1], qualified_name=value,
                    symbol_type=kind, line_start=line, line_end=line, language=Language.DOTNET,
                    metadata={"dotnet_kind": metadata_kind},
                ))
        return result
