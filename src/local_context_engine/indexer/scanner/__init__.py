"""File system scanner with gitignore support and language detection."""

from local_context_engine.indexer.scanner.file_scanner import FileScanner
from local_context_engine.indexer.scanner.gitignore_handler import GitignoreHandler

__all__ = ["FileScanner", "GitignoreHandler"]
