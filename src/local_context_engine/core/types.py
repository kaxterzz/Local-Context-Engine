"""
Core domain types that flow through the entire Local Context Engine pipeline.

These are pure data structures with no business logic. All modules depend on
these types; no module here depends on any other module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np


# ─────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────


class Language(str, Enum):
    """Supported source code languages."""

    PHP = "php"
    TYPESCRIPT = "typescript"
    TSX = "tsx"
    JAVASCRIPT = "javascript"
    JSX = "jsx"
    PYTHON = "python"
    CSHARP = "csharp"
    DOTNET = "dotnet"
    ASP = "asp"
    ASPNET = "aspnet"
    CSS = "css"
    HTML = "html"
    BLADE = "blade"  # Laravel Blade templates
    JSON = "json"
    YAML = "yaml"
    SQL = "sql"
    MARKDOWN = "markdown"
    SHELL = "shell"
    UNKNOWN = "unknown"


class SymbolType(str, Enum):
    """Types of code symbols extracted by parsers."""

    # OOP constructs
    CLASS = "class"
    ABSTRACT_CLASS = "abstract_class"
    INTERFACE = "interface"
    TRAIT = "trait"
    ENUM = "enum"

    # Functions / Methods
    FUNCTION = "function"
    METHOD = "method"
    CONSTRUCTOR = "constructor"
    ARROW_FUNCTION = "arrow_function"

    # Properties / Constants
    PROPERTY = "property"
    CONSTANT = "constant"
    VARIABLE = "variable"

    # Module system
    IMPORT = "import"
    EXPORT = "export"
    NAMESPACE = "namespace"

    # React / Frontend
    REACT_COMPONENT = "react_component"
    REACT_HOOK = "hook"
    CONTEXT_PROVIDER = "context_provider"
    CONTEXT_CONSUMER = "context_consumer"
    HOC = "hoc"  # Higher-Order Component

    # Laravel / Backend
    CONTROLLER = "controller"
    SERVICE = "service"
    REPOSITORY = "repository"
    MODEL = "model"
    POLICY = "policy"
    MIDDLEWARE = "middleware"
    MIGRATION = "migration"
    SEEDER = "seeder"
    FACTORY = "factory"
    EVENT = "event"
    LISTENER = "listener"
    JOB = "job"
    COMMAND = "command"
    REQUEST = "request"          # Form Request
    RESOURCE = "resource"        # API Resource
    NOTIFICATION = "notification"
    OBSERVER = "observer"
    RULE = "rule"                # Validation Rule
    PROVIDER = "provider"        # Service Provider
    ROUTE = "route"
    RELATIONSHIP = "relationship"

    # Database
    TABLE = "table"
    VIEW = "view"
    STORED_PROCEDURE = "stored_procedure"
    PROJECT = "project"
    PACKAGE = "package"

    # Generic
    TYPE_ALIAS = "type_alias"
    DECORATOR = "decorator"
    UNKNOWN = "unknown"


class RelationshipType(str, Enum):
    """Types of relationships between symbols in the symbol graph."""

    # Code structure
    EXTENDS = "extends"
    IMPLEMENTS = "implements"
    USES = "uses"           # PHP trait usage
    IMPORTS = "imports"

    # Calls / Dependencies
    CALLS = "calls"
    DEPENDS_ON = "depends_on"
    INJECTS = "injects"     # Dependency injection

    # Laravel relationships
    HAS_ONE = "has_one"
    HAS_MANY = "has_many"
    BELONGS_TO = "belongs_to"
    BELONGS_TO_MANY = "belongs_to_many"
    HAS_ONE_THROUGH = "has_one_through"
    HAS_MANY_THROUGH = "has_many_through"
    MORPH_ONE = "morph_one"
    MORPH_MANY = "morph_many"

    # Application flow
    TRIGGERS = "triggers"       # Event dispatching
    HANDLES = "handles"         # Event/Job handling
    DISPATCHES = "dispatches"   # Job dispatching
    ROUTE_TO = "route_to"       # Route → Controller

    # React / Data
    QUERIES = "queries"         # TanStack Query hook
    MUTATES = "mutates"         # TanStack Mutation
    RENDERS = "renders"         # Component → Child component
    PROVIDES = "provides"       # Context Provider
    CONSUMES = "consumes"       # Context Consumer

    # Migration
    MIGRATES = "migrates"       # Model → Migration


class IndexingStatus(str, Enum):
    """Status of a file during incremental indexing."""

    NEW = "new"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"
    DELETED = "deleted"
    FAILED = "failed"
    SKIPPED = "skipped"  # Security-blacklisted or too large


# ─────────────────────────────────────────────────────────────
# Core Data Records
# ─────────────────────────────────────────────────────────────


@dataclass
class FileRecord:
    """
    Represents a source file tracked by the indexer.

    The ``path`` is always relative to the repository root.
    ``absolute_path`` is used internally for reading content.
    """

    id: str
    path: str              # Relative to repo root
    absolute_path: str
    language: Language
    size_bytes: int
    hash: str              # MD5 or SHA256 for change detection
    modified_at: datetime
    line_count: int = 0
    indexed_at: datetime | None = None
    status: IndexingStatus = IndexingStatus.NEW
    framework: str | None = None   # "laravel", "react", "vue", etc.


@dataclass
class Symbol:
    """
    A named code entity extracted from a parsed file.

    Symbols are the primary units of semantic understanding:
    classes, functions, React components, Laravel controllers, etc.
    """

    id: str
    file_id: str
    file_path: str         # Relative path (convenience)
    name: str              # Short name (e.g. "UserController")
    qualified_name: str    # Full name (e.g. "App\\Http\\Controllers\\UserController")
    symbol_type: SymbolType
    line_start: int
    line_end: int
    language: Language
    visibility: str | None = None    # "public" | "protected" | "private"
    docstring: str | None = None
    parent_id: str | None = None     # For methods → their class
    parent_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """
    A text chunk ready for embedding.

    Chunks are bounded segments of source code, preferably aligned
    to symbol boundaries. ``content`` is the original code;
    ``content_masked`` has PII replaced before embedding generation.
    """

    id: str
    file_id: str
    file_path: str         # Relative path (convenience)
    symbol_id: str | None  # The primary symbol this chunk covers
    content: str           # Raw source content
    content_masked: str    # PII-masked content used for embedding
    language: Language
    symbol_name: str | None
    symbol_type: SymbolType | None
    line_start: int
    line_end: int
    hash: str              # Hash of content for change detection
    token_count: int
    chunk_index: int = 0   # Order within file
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkVector:
    """Pairs a chunk ID with its embedding vector."""

    chunk_id: str
    embedding: np.ndarray
    model_name: str
    dimension: int


@dataclass
class Relationship:
    """
    A directed relationship between two symbols in the symbol graph.

    ``target_symbol_id`` may be None when the target symbol has not yet
    been indexed (forward reference). ``target_name`` always holds the
    string name for later resolution.
    """

    id: str
    source_symbol_id: str
    source_file_path: str
    target_symbol_id: str | None   # Resolved after full indexing pass
    target_name: str               # Always present for reference
    relationship_type: RelationshipType
    confidence: float = 1.0        # 0–1 confidence of inferred relationship
    metadata: dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────
# Retrieval Types
# ─────────────────────────────────────────────────────────────


@dataclass
class SearchResult:
    """
    A single result returned by the retrieval layer.

    Contains everything an AI agent needs to understand the result:
    file path, line range, symbol info, the code snippet, and the
    composite relevance score with a breakdown by component.
    """

    chunk_id: str
    file_path: str
    symbol_name: str | None
    symbol_type: str | None
    language: str
    snippet: str
    line_start: int
    line_end: int
    score: float                                     # Final hybrid score
    score_breakdown: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchQuery:
    """Structured query for the retrieval layer."""

    text: str
    limit: int = 10
    language_filter: Language | None = None
    symbol_type_filter: SymbolType | None = None
    path_filter: str | None = None          # Glob pattern on file path
    min_score: float = 0.0
    include_snippets: bool = True
    expand_query: bool = True               # Enable query expansion


# ─────────────────────────────────────────────────────────────
# Indexing / Stats Types
# ─────────────────────────────────────────────────────────────


@dataclass
class IndexingProgress:
    """Real-time progress reported during indexing."""

    phase: str                  # "scanning" | "parsing" | "chunking" | "embedding"
    total: int = 0
    processed: int = 0
    failed: int = 0
    current_file: str | None = None
    elapsed_seconds: float = 0.0


@dataclass
class IndexingStats:
    """Summary statistics produced after an indexing run."""

    total_files_scanned: int = 0
    files_indexed: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    total_symbols: int = 0
    total_chunks: int = 0
    total_vectors: int = 0
    scan_time_seconds: float = 0.0
    parse_time_seconds: float = 0.0
    chunk_time_seconds: float = 0.0
    embedding_time_seconds: float = 0.0
    total_time_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


@dataclass
class RepositoryStats:
    """High-level statistics for a fully indexed repository."""

    root_path: str
    total_files: int = 0
    total_lines: int = 0
    total_symbols: int = 0
    total_chunks: int = 0
    total_vectors: int = 0
    languages: dict[str, int] = field(default_factory=dict)  # language → file count
    symbol_types: dict[str, int] = field(default_factory=dict)
    last_indexed_at: datetime | None = None
    embedding_model: str | None = None
    vector_dimension: int = 0
    index_size_bytes: int = 0
    db_size_bytes: int = 0
