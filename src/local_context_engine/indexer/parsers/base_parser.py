"""
Abstract base class and shared utilities for all code parsers.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from local_context_engine.core.types import Language, Relationship, Symbol


@dataclass
class ParseResult:
    """The output of parsing a single source file."""

    file_id: str
    file_path: str
    language: Language
    symbols: list[Symbol] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class BaseParser(ABC):
    """
    Abstract code parser interface.

    Subclasses implement :meth:`parse` for a specific language using
    Tree-sitter grammars and optionally pattern-based heuristics.
    """

    @property
    @abstractmethod
    def supported_language(self) -> Language:
        """The language this parser handles."""

    @property
    def supported_extensions(self) -> list[str]:
        """File extensions handled by this parser."""
        return []

    @abstractmethod
    def parse(self, source: str, file_id: str, file_path: str) -> ParseResult:
        """
        Parse source code and extract symbols + relationships.

        Args:
            source:    Raw source code string.
            file_id:   Database ID of the file record.
            file_path: Relative path (used for relationship metadata).

        Returns:
            :class:`ParseResult` with extracted symbols and relationships.
        """

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def make_symbol_id(file_id: str, qualified_name: str, line_start: int) -> str:
        """Generate a stable, deterministic symbol ID."""
        key = f"{file_id}::{qualified_name}::{line_start}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, key))

    @staticmethod
    def make_rel_id(source_id: str, target_name: str, rel_type: str) -> str:
        """Generate a stable relationship ID."""
        key = f"{source_id}->{target_name}::{rel_type}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, key))

    @staticmethod
    def safe_decode(node_text: bytes) -> str:
        """Safely decode bytes to string."""
        return node_text.decode("utf-8", errors="replace")
