"""
Performance benchmarks for the Local Context Engine.

Run with: pytest tests/benchmarks/ --benchmark-only
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_context_engine.indexer.chunkers.symbol_chunker import SymbolChunker
from local_context_engine.indexer.chunkers.token_counter import TokenCounter
from local_context_engine.indexer.parsers.php_parser import PHPParser
from local_context_engine.indexer.parsers.python_parser import PythonParser
from local_context_engine.indexer.parsers.typescript_parser import TypeScriptParser
from local_context_engine.core.types import FileRecord, IndexingStatus, Language
from local_context_engine.security.pii_masker import PIIMasker
from datetime import datetime, timezone


def _make_record(lang: Language) -> FileRecord:
    return FileRecord(
        id="bench-file",
        path="bench.py",
        absolute_path="/bench.py",
        language=lang,
        size_bytes=10000,
        hash="abc",
        modified_at=datetime.now(tz=timezone.utc),
        status=IndexingStatus.NEW,
    )


# ── Generate large synthetic sources ─────────────────────────

def _large_php_source(num_methods: int = 50) -> str:
    methods = "\n\n".join(
        f"    public function method{i}(string $param): void\n"
        f"    {{\n"
        f"        $result = $this->service->process($param);\n"
        f"        Log::info('method{i} called', ['param' => $param]);\n"
        f"    }}"
        for i in range(num_methods)
    )
    return f"""\
<?php
namespace App\\Http\\Controllers;

class LargeController extends Controller
{{
{methods}
}}
"""


def _large_python_source(num_methods: int = 50) -> str:
    methods = "\n\n".join(
        f"    def method_{i}(self, param: str) -> None:\n"
        f"        result = self.service.process(param)\n"
        f"        print(f'method_{i} called with {{param}}')"
        for i in range(num_methods)
    )
    return f"""\
class LargeService:
    \"\"\"A large service class for benchmarking.\"\"\"

{methods}
"""


def _large_typescript_source(num_functions: int = 30) -> str:
    functions = "\n\n".join(
        f"export function hook{i}() {{\n"
        f"    const {{ data }} = useQuery({{ queryKey: ['key{i}'], queryFn: fetch{i} }});\n"
        f"    return data;\n"
        f"}}"
        for i in range(num_functions)
    )
    return f"import {{ useQuery }} from '@tanstack/react-query';\n\n{functions}"


# ── Benchmark: PHP parsing ────────────────────────────────────

def test_php_parser_benchmark(benchmark) -> None:
    """Benchmark PHP parsing performance."""
    parser = PHPParser()
    source = _large_php_source(50)

    def _parse():
        return parser.parse(source, "bench-001", "LargeController.php")

    result = benchmark(_parse)
    assert result.symbols  # Sanity check


# ── Benchmark: Python parsing ─────────────────────────────────

def test_python_parser_benchmark(benchmark) -> None:
    """Benchmark Python parsing performance."""
    parser = PythonParser()
    source = _large_python_source(50)

    def _parse():
        return parser.parse(source, "bench-001", "large_service.py")

    result = benchmark(_parse)
    assert result.symbols


# ── Benchmark: TypeScript parsing ────────────────────────────

def test_typescript_parser_benchmark(benchmark) -> None:
    """Benchmark TypeScript parsing performance."""
    parser = TypeScriptParser(Language.TYPESCRIPT)
    source = _large_typescript_source(30)

    def _parse():
        return parser.parse(source, "bench-001", "hooks.ts")

    result = benchmark(_parse)
    assert result is not None


# ── Benchmark: Chunking ───────────────────────────────────────

def test_chunking_benchmark(benchmark) -> None:
    """Benchmark the symbol chunker."""
    parser = PythonParser()
    chunker = SymbolChunker(pii_masker=PIIMasker(), tokenizer="approx")
    source = _large_python_source(50)
    file_record = _make_record(Language.PYTHON)
    parse_result = parser.parse(source, file_record.id, file_record.path)

    def _chunk():
        return chunker.chunk(source, file_record, parse_result)

    chunks = benchmark(_chunk)
    assert len(chunks) > 0


# ── Benchmark: Token counting ─────────────────────────────────

def test_token_counter_approx_benchmark(benchmark) -> None:
    """Benchmark approximate token counting."""
    counter = TokenCounter("approx")
    text = "def foo(x: int) -> str:\n    return str(x)\n" * 100

    def _count():
        return counter.count(text)

    count = benchmark(_count)
    assert count > 0


# ── Benchmark: PII masking ────────────────────────────────────

def test_pii_masking_benchmark(benchmark) -> None:
    """Benchmark PII masking throughput."""
    masker = PIIMasker()
    # Text with no PII (worst case for masker — must scan everything)
    text = "public function getUserById(int $id): User\n{\n    return $this->repo->find($id);\n}\n" * 50

    def _mask():
        return masker.mask(text)

    masked, stats = benchmark(_mask)
    assert masked
