"""Unit tests for the chunking engine."""

from __future__ import annotations

import pytest

from local_context_engine.core.types import FileRecord, IndexingStatus, Language
from local_context_engine.indexer.chunkers.symbol_chunker import SymbolChunker
from local_context_engine.indexer.chunkers.token_counter import TokenCounter
from local_context_engine.indexer.parsers.php_parser import PHPParser
from local_context_engine.indexer.parsers.python_parser import PythonParser
from local_context_engine.security.pii_masker import PIIMasker

from datetime import datetime, timezone


def _make_file_record(path: str, lang: Language) -> FileRecord:
    return FileRecord(
        id="test-file-001",
        path=path,
        absolute_path=f"/repo/{path}",
        language=lang,
        size_bytes=1000,
        hash="abc123",
        modified_at=datetime.now(tz=timezone.utc),
        status=IndexingStatus.NEW,
    )


class TestTokenCounter:
    """Tests for the token counter."""

    def test_counts_approximate(self) -> None:
        counter = TokenCounter("approx")
        # 40 chars ≈ 10 tokens
        assert counter.count("x" * 40) == 10

    def test_split_respects_max(self) -> None:
        counter = TokenCounter("approx")
        # Create text with ~400 tokens (1600 chars)
        long_text = ("This is a test line.\n" * 80)
        segments = counter.split_at_token_boundary(long_text, max_tokens=100)
        assert len(segments) > 1
        for seg in segments:
            assert counter.count(seg) <= 100 + 10  # Allow small overflow

    def test_short_text_no_split(self) -> None:
        counter = TokenCounter("approx")
        text = "def foo(): pass"
        segments = counter.split_at_token_boundary(text, max_tokens=100)
        assert len(segments) == 1
        assert segments[0] == text

    def test_empty_text(self) -> None:
        counter = TokenCounter("approx")
        segments = counter.split_at_token_boundary("", max_tokens=100)
        assert segments == [""]


class TestSymbolChunker:
    """Tests for the symbol-aware chunker."""

    @pytest.fixture
    def chunker(self) -> SymbolChunker:
        return SymbolChunker(
            pii_masker=PIIMasker(),
            min_tokens=5,
            target_tokens=100,
            max_tokens=300,
            overlap_tokens=10,
            tokenizer="approx",
        )

    def test_chunks_php_file(
        self, chunker: SymbolChunker, sample_php_source: str
    ) -> None:
        file_record = _make_file_record("UserController.php", Language.PHP)
        parser = PHPParser()
        parse_result = parser.parse(sample_php_source, file_record.id, file_record.path)

        chunks = chunker.chunk(sample_php_source, file_record, parse_result)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.file_id == file_record.id
            assert chunk.content
            assert chunk.hash
            assert chunk.token_count > 0

    def test_chunks_python_file(
        self, chunker: SymbolChunker, sample_python_source: str
    ) -> None:
        file_record = _make_file_record("users.py", Language.PYTHON)
        parser = PythonParser()
        parse_result = parser.parse(sample_python_source, file_record.id, file_record.path)

        chunks = chunker.chunk(sample_python_source, file_record, parse_result)
        assert len(chunks) >= 1

    def test_pii_masking_applied(self, chunker: SymbolChunker) -> None:
        source = "// user email: test@example.com\nfunction hello() { return 'world'; }"
        file_record = _make_file_record("test.js", Language.JAVASCRIPT)

        from local_context_engine.indexer.parsers.base_parser import ParseResult

        parse_result = ParseResult(file_id=file_record.id, file_path=file_record.path, language=Language.JAVASCRIPT)
        chunks = chunker.chunk(source, file_record, parse_result)

        assert len(chunks) >= 1
        for chunk in chunks:
            # PII-masked content should not contain the email
            assert "test@example.com" not in chunk.content_masked

    def test_chunk_has_context_header(
        self, chunker: SymbolChunker, sample_php_source: str
    ) -> None:
        file_record = _make_file_record("UserController.php", Language.PHP)
        parser = PHPParser()
        parse_result = parser.parse(sample_php_source, file_record.id, file_record.path)

        chunks = chunker.chunk(sample_php_source, file_record, parse_result)
        # At least one chunk should have the file path in the header
        has_header = any("File:" in c.content for c in chunks)
        assert has_header

    def test_empty_file_no_crash(self, chunker: SymbolChunker) -> None:
        file_record = _make_file_record("empty.php", Language.PHP)
        from local_context_engine.indexer.parsers.base_parser import ParseResult

        parse_result = ParseResult(file_id=file_record.id, file_path=file_record.path, language=Language.PHP)
        chunks = chunker.chunk("", file_record, parse_result)
        # Empty file may produce 0 chunks or handle gracefully
        assert isinstance(chunks, list)

    def test_chunks_are_within_size_bounds(
        self, chunker: SymbolChunker, sample_python_source: str
    ) -> None:
        file_record = _make_file_record("users.py", Language.PYTHON)
        parser = PythonParser()
        parse_result = parser.parse(sample_python_source, file_record.id, file_record.path)

        chunks = chunker.chunk(sample_python_source, file_record, parse_result)
        counter = TokenCounter("approx")
        for chunk in chunks:
            # All chunks should be within the configured max
            assert counter.count(chunk.content) <= 300 + 50  # Allow small overflow
