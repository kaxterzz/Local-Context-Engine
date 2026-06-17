"""Abstract base class for code chunkers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from local_context_engine.core.types import Chunk, FileRecord
from local_context_engine.indexer.parsers.base_parser import ParseResult


class BaseChunker(ABC):
    """
    Abstract chunker interface.

    Subclasses implement :meth:`chunk` to split parsed source files
    into bounded, semantically meaningful :class:`Chunk` objects.
    """

    @abstractmethod
    def chunk(
        self,
        source: str,
        file_record: FileRecord,
        parse_result: ParseResult,
    ) -> list[Chunk]:
        """
        Produce chunks from a parsed source file.

        Args:
            source:       Raw source code string.
            file_record:  The file metadata record.
            parse_result: Symbols extracted by the parser.

        Returns:
            Ordered list of :class:`Chunk` objects ready for embedding.
        """
