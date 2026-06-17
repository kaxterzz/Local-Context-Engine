"""
Symbol-aware chunker — the primary chunking strategy.

Algorithm:
  1. Sort all extracted symbols by line_start.
  2. For each symbol, extract its source lines as a candidate chunk.
  3. If the symbol is too small  (< min_tokens), merge with the next symbol.
  4. If the symbol is too large  (> max_tokens), split at sub-symbol boundaries
     (methods within a class), then fall back to line-based splitting.
  5. Between symbols, capture any "gap" code (loose statements, comments).
  6. All chunks get a context header with file path and symbol name.
  7. PII masking is applied to produce ``content_masked``.

Result: every character of the source file ends up in at least one chunk,
with chunk boundaries following code structure rather than arbitrary character counts.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from typing import Sequence

from local_context_engine.core.types import Chunk, FileRecord, Language, Symbol, SymbolType
from local_context_engine.indexer.chunkers.base_chunker import BaseChunker
from local_context_engine.indexer.chunkers.token_counter import TokenCounter
from local_context_engine.indexer.parsers.base_parser import ParseResult
from local_context_engine.security.pii_masker import PIIMasker

logger = logging.getLogger(__name__)

# Symbol types that define meaningful standalone chunk boundaries
_TOP_LEVEL_TYPES = {
    SymbolType.CLASS,
    SymbolType.ABSTRACT_CLASS,
    SymbolType.INTERFACE,
    SymbolType.TRAIT,
    SymbolType.ENUM,
    SymbolType.FUNCTION,
    SymbolType.REACT_COMPONENT,
    SymbolType.REACT_HOOK,
    SymbolType.CONTROLLER,
    SymbolType.SERVICE,
    SymbolType.REPOSITORY,
    SymbolType.MODEL,
    SymbolType.POLICY,
    SymbolType.EVENT,
    SymbolType.LISTENER,
    SymbolType.JOB,
    SymbolType.COMMAND,
    SymbolType.MIGRATION,
    SymbolType.PROVIDER,
    SymbolType.MIDDLEWARE,
}

_METHOD_TYPES = {
    SymbolType.METHOD,
    SymbolType.CONSTRUCTOR,
    SymbolType.ARROW_FUNCTION,
}


def _make_chunk_id(file_id: str, line_start: int, index: int) -> str:
    key = f"{file_id}::{line_start}::{index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def _hash_content(content: str) -> str:
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def _add_context_header(content: str, file_path: str, symbol_name: str | None) -> str:
    """Prepend a context header that helps the embedding model understand the chunk."""
    if symbol_name:
        return f"// File: {file_path}\n// Symbol: {symbol_name}\n\n{content}"
    return f"// File: {file_path}\n\n{content}"


class SymbolChunker(BaseChunker):
    """
    Produces semantically meaningful chunks aligned to symbol boundaries.

    Falls back to line-based splitting for symbols that exceed ``max_tokens``.
    """

    def __init__(
        self,
        pii_masker: PIIMasker | None = None,
        min_tokens: int = 50,
        target_tokens: int = 500,
        max_tokens: int = 1200,
        overlap_tokens: int = 50,
        tokenizer: str = "approx",
    ) -> None:
        self._masker = pii_masker or PIIMasker()
        self._counter = TokenCounter(tokenizer)
        self._min_tokens = min_tokens
        self._target_tokens = target_tokens
        self._max_tokens = max_tokens
        self._overlap_tokens = overlap_tokens

    def chunk(
        self,
        source: str,
        file_record: FileRecord,
        parse_result: ParseResult,
    ) -> list[Chunk]:
        lines = source.splitlines()
        total_lines = len(lines)
        chunks: list[Chunk] = []
        chunk_index = 0

        # Filter to top-level symbols only (skip methods that are children)
        top_symbols = [
            s for s in parse_result.symbols if s.symbol_type in _TOP_LEVEL_TYPES
        ]
        top_symbols.sort(key=lambda s: s.line_start)

        # Track covered line ranges to capture gaps
        covered_ranges: list[tuple[int, int]] = []
        symbol_chunks_map: dict[str, list[Chunk]] = {}

        for symbol in top_symbols:
            # Extract symbol source (1-based lines → 0-based slice)
            sym_lines = lines[symbol.line_start - 1 : symbol.line_end]
            sym_source = "\n".join(sym_lines)
            token_count = self._counter.count(sym_source)

            covered_ranges.append((symbol.line_start, symbol.line_end))

            if token_count <= self._max_tokens:
                # Symbol fits in a single chunk
                chunk = self._make_chunk(
                    content=sym_source,
                    file_record=file_record,
                    symbol=symbol,
                    line_start=symbol.line_start,
                    line_end=symbol.line_end,
                    chunk_index=chunk_index,
                )
                chunks.append(chunk)
                symbol_chunks_map[symbol.id] = [chunk]
                chunk_index += 1
            else:
                # Symbol too large — split at method boundaries or line-based
                sub_chunks = self._split_large_symbol(
                    sym_source=sym_source,
                    file_record=file_record,
                    symbol=symbol,
                    parse_result=parse_result,
                    lines=lines,
                    chunk_index_start=chunk_index,
                )
                chunks.extend(sub_chunks)
                symbol_chunks_map[symbol.id] = sub_chunks
                chunk_index += len(sub_chunks)

        # Capture gap code between / before / after symbols
        gap_chunks = self._capture_gaps(
            lines=lines,
            total_lines=total_lines,
            covered_ranges=covered_ranges,
            file_record=file_record,
            chunk_index_start=chunk_index,
        )
        chunks.extend(gap_chunks)
        chunk_index += len(gap_chunks)

        # If no symbols were found (e.g. config file, migration SQL), chunk the whole file
        if not chunks:
            chunks = self._chunk_whole_file(source, file_record, chunk_index)

        logger.debug(
            "Chunked %s into %d chunks (%d symbols)",
            file_record.path,
            len(chunks),
            len(top_symbols),
        )
        return chunks

    # ── Private helpers ────────────────────────────────────────

    def _make_chunk(
        self,
        content: str,
        file_record: FileRecord,
        symbol: Symbol | None,
        line_start: int,
        line_end: int,
        chunk_index: int,
    ) -> Chunk:
        """Create a single Chunk object."""
        headed_content = _add_context_header(
            content,
            file_record.path,
            symbol.name if symbol else None,
        )
        masked, _stats = self._masker.mask(headed_content)
        token_count = self._counter.count(headed_content)

        return Chunk(
            id=_make_chunk_id(file_record.id, line_start, chunk_index),
            file_id=file_record.id,
            file_path=file_record.path,
            symbol_id=symbol.id if symbol else None,
            content=headed_content,
            content_masked=masked,
            language=file_record.language,
            symbol_name=symbol.name if symbol else None,
            symbol_type=symbol.symbol_type if symbol else None,
            line_start=line_start,
            line_end=line_end,
            hash=_hash_content(headed_content),
            token_count=token_count,
            chunk_index=chunk_index,
        )

    def _split_large_symbol(
        self,
        sym_source: str,
        file_record: FileRecord,
        symbol: Symbol,
        parse_result: ParseResult,
        lines: list[str],
        chunk_index_start: int,
    ) -> list[Chunk]:
        """Split an oversized symbol at method boundaries, then line-based."""
        chunks: list[Chunk] = []
        chunk_index = chunk_index_start

        # Find child methods within this symbol
        child_methods = [
            s for s in parse_result.symbols
            if s.parent_id == symbol.id and s.symbol_type in _METHOD_TYPES
        ]
        child_methods.sort(key=lambda s: s.line_start)

        if child_methods:
            # Class header (before first method)
            first_method_line = child_methods[0].line_start
            header_lines = lines[symbol.line_start - 1 : first_method_line - 1]
            header_source = "\n".join(header_lines)
            if self._counter.count(header_source) >= self._min_tokens:
                chunks.append(
                    self._make_chunk(
                        content=header_source,
                        file_record=file_record,
                        symbol=symbol,
                        line_start=symbol.line_start,
                        line_end=first_method_line - 1,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1

            # Each method as its own chunk (with merging for tiny methods)
            pending_lines: list[str] = []
            pending_start = first_method_line
            pending_tokens = 0

            for method in child_methods:
                method_lines = lines[method.line_start - 1 : method.line_end]
                method_source = "\n".join(method_lines)
                method_tokens = self._counter.count(method_source)

                if pending_tokens + method_tokens > self._max_tokens and pending_lines:
                    # Flush pending
                    content = "\n".join(pending_lines)
                    if self._counter.count(content) >= self._min_tokens:
                        chunks.append(
                            self._make_chunk(
                                content=content,
                                file_record=file_record,
                                symbol=symbol,
                                line_start=pending_start,
                                line_end=method.line_start - 1,
                                chunk_index=chunk_index,
                            )
                        )
                        chunk_index += 1
                    pending_lines = method_lines
                    pending_start = method.line_start
                    pending_tokens = method_tokens
                elif method_tokens > self._max_tokens:
                    # Single very long method — line-based split
                    sub_segs = self._counter.split_at_token_boundary(
                        method_source, self._max_tokens, self._overlap_tokens
                    )
                    for seg in sub_segs:
                        chunks.append(
                            self._make_chunk(
                                content=seg,
                                file_record=file_record,
                                symbol=method,
                                line_start=method.line_start,
                                line_end=method.line_end,
                                chunk_index=chunk_index,
                            )
                        )
                        chunk_index += 1
                else:
                    pending_lines.extend(method_lines)
                    pending_tokens += method_tokens

            if pending_lines:
                content = "\n".join(pending_lines)
                if self._counter.count(content) >= self._min_tokens:
                    chunks.append(
                        self._make_chunk(
                            content=content,
                            file_record=file_record,
                            symbol=symbol,
                            line_start=pending_start,
                            line_end=symbol.line_end,
                            chunk_index=chunk_index,
                        )
                    )
                    chunk_index += 1
        else:
            # No methods — pure line-based splitting
            segments = self._counter.split_at_token_boundary(
                sym_source, self._max_tokens, self._overlap_tokens
            )
            current_line = symbol.line_start
            for seg in segments:
                seg_lines = seg.count("\n") + 1
                chunks.append(
                    self._make_chunk(
                        content=seg,
                        file_record=file_record,
                        symbol=symbol,
                        line_start=current_line,
                        line_end=current_line + seg_lines - 1,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1
                current_line += seg_lines

        return chunks

    def _capture_gaps(
        self,
        lines: list[str],
        total_lines: int,
        covered_ranges: list[tuple[int, int]],
        file_record: FileRecord,
        chunk_index_start: int,
    ) -> list[Chunk]:
        """Capture source lines not covered by any symbol."""
        if not covered_ranges:
            return []

        chunks: list[Chunk] = []
        chunk_index = chunk_index_start
        covered_ranges_sorted = sorted(covered_ranges, key=lambda r: r[0])

        # Gap before first symbol
        first_start = covered_ranges_sorted[0][0]
        if first_start > 1:
            gap_lines = lines[0 : first_start - 1]
            gap_source = "\n".join(gap_lines).strip()
            if gap_source and self._counter.count(gap_source) >= self._min_tokens:
                chunks.append(
                    self._make_chunk(
                        content=gap_source,
                        file_record=file_record,
                        symbol=None,
                        line_start=1,
                        line_end=first_start - 1,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1

        # Gaps between symbols
        for i in range(len(covered_ranges_sorted) - 1):
            gap_start = covered_ranges_sorted[i][1] + 1
            gap_end = covered_ranges_sorted[i + 1][0] - 1
            if gap_end >= gap_start:
                gap_lines = lines[gap_start - 1 : gap_end]
                gap_source = "\n".join(gap_lines).strip()
                if gap_source and self._counter.count(gap_source) >= self._min_tokens:
                    chunks.append(
                        self._make_chunk(
                            content=gap_source,
                            file_record=file_record,
                            symbol=None,
                            line_start=gap_start,
                            line_end=gap_end,
                            chunk_index=chunk_index,
                        )
                    )
                    chunk_index += 1

        # Gap after last symbol
        last_end = covered_ranges_sorted[-1][1]
        if last_end < total_lines:
            gap_lines = lines[last_end:]
            gap_source = "\n".join(gap_lines).strip()
            if gap_source and self._counter.count(gap_source) >= self._min_tokens:
                chunks.append(
                    self._make_chunk(
                        content=gap_source,
                        file_record=file_record,
                        symbol=None,
                        line_start=last_end + 1,
                        line_end=total_lines,
                        chunk_index=chunk_index,
                    )
                )

        return chunks

    def _chunk_whole_file(
        self, source: str, file_record: FileRecord, chunk_index_start: int
    ) -> list[Chunk]:
        """Fallback: chunk the entire file by line boundaries."""
        segments = self._counter.split_at_token_boundary(
            source, self._max_tokens, self._overlap_tokens
        )
        chunks: list[Chunk] = []
        current_line = 1
        for i, seg in enumerate(segments):
            seg_line_count = seg.count("\n") + 1
            chunks.append(
                self._make_chunk(
                    content=seg,
                    file_record=file_record,
                    symbol=None,
                    line_start=current_line,
                    line_end=current_line + seg_line_count - 1,
                    chunk_index=chunk_index_start + i,
                )
            )
            current_line += seg_line_count
        return chunks

    @classmethod
    def from_config(cls, config: "EngineConfig") -> "SymbolChunker":  # type: ignore[name-defined]
        from local_context_engine.core.config import EngineConfig  # noqa: PLC0415
        from local_context_engine.security.pii_masker import PIIMasker  # noqa: PLC0415

        assert isinstance(config, EngineConfig)
        masker = PIIMasker.from_config(config.pii_masking)
        return cls(
            pii_masker=masker,
            min_tokens=config.chunking.min_tokens,
            target_tokens=config.chunking.target_tokens,
            max_tokens=config.chunking.max_tokens,
            overlap_tokens=config.chunking.overlap_tokens,
            tokenizer=config.chunking.tokenizer,
        )
