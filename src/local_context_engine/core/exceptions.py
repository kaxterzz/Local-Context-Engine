"""
Custom exception hierarchy for Local Context Engine.

All exceptions inherit from ``ContextEngineError`` so callers can catch
the entire family with a single ``except`` clause when needed.
"""


class ContextEngineError(Exception):
    """Base exception for all Local Context Engine errors."""


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────


class ConfigurationError(ContextEngineError):
    """Raised when the configuration is invalid or missing required values."""


# ─────────────────────────────────────────────────────────────
# Security
# ─────────────────────────────────────────────────────────────


class SecurityError(ContextEngineError):
    """Raised when a security policy violation is detected."""


class BlacklistedFileError(SecurityError):
    """Raised when an attempt is made to index a blacklisted file."""

    def __init__(self, path: str, pattern: str) -> None:
        self.path = path
        self.pattern = pattern
        super().__init__(f"File '{path}' is blacklisted by pattern '{pattern}'")


# ─────────────────────────────────────────────────────────────
# Scanner
# ─────────────────────────────────────────────────────────────


class ScannerError(ContextEngineError):
    """Raised by the file scanner."""


class RepositoryNotFoundError(ScannerError):
    """Raised when the target repository root directory does not exist."""

    def __init__(self, path: str) -> None:
        self.path = path
        super().__init__(f"Repository not found: '{path}'")


# ─────────────────────────────────────────────────────────────
# Parsing
# ─────────────────────────────────────────────────────────────


class ParseError(ContextEngineError):
    """Raised when a file cannot be parsed."""

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"Parse error in '{path}': {reason}")


class UnsupportedLanguageError(ParseError):
    """Raised when no parser is registered for the file's language."""

    def __init__(self, path: str, language: str) -> None:
        self.language = language
        super().__init__(path, f"No parser registered for language '{language}'")


# ─────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────


class ChunkingError(ContextEngineError):
    """Raised when chunking fails."""


# ─────────────────────────────────────────────────────────────
# Embedding
# ─────────────────────────────────────────────────────────────


class EmbeddingError(ContextEngineError):
    """Raised by the embedding engine."""


class ModelNotAvailableError(EmbeddingError):
    """Raised when the requested embedding model cannot be loaded."""

    def __init__(self, model_name: str, reason: str) -> None:
        self.model_name = model_name
        super().__init__(f"Model '{model_name}' not available: {reason}")


# ─────────────────────────────────────────────────────────────
# Vector Store
# ─────────────────────────────────────────────────────────────


class VectorStoreError(ContextEngineError):
    """Raised by the vector store."""


class VectorDimensionMismatchError(VectorStoreError):
    """Raised when inserting a vector with the wrong dimension."""

    def __init__(self, expected: int, got: int) -> None:
        super().__init__(f"Vector dimension mismatch: expected {expected}, got {got}")


# ─────────────────────────────────────────────────────────────
# Metadata Store
# ─────────────────────────────────────────────────────────────


class MetadataStoreError(ContextEngineError):
    """Raised by the metadata store."""


class RecordNotFoundError(MetadataStoreError):
    """Raised when a requested record does not exist."""

    def __init__(self, entity: str, id_: str) -> None:
        self.entity = entity
        self.id = id_
        super().__init__(f"{entity} not found: '{id_}'")


# ─────────────────────────────────────────────────────────────
# Symbol Graph
# ─────────────────────────────────────────────────────────────


class SymbolGraphError(ContextEngineError):
    """Raised by the symbol graph layer."""


# ─────────────────────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────────────────────


class RetrievalError(ContextEngineError):
    """Raised by the retrieval layer."""


class IndexNotReadyError(RetrievalError):
    """Raised when a search is attempted before the index is built."""

    def __init__(self) -> None:
        super().__init__(
            "The index is not ready. Run 'context index <repo_path>' first."
        )


# ─────────────────────────────────────────────────────────────
# Memory
# ─────────────────────────────────────────────────────────────


class MemoryError(ContextEngineError):
    """Raised by the agent memory module."""


# ─────────────────────────────────────────────────────────────
# MCP Server
# ─────────────────────────────────────────────────────────────


class MCPServerError(ContextEngineError):
    """Raised by the MCP server."""
