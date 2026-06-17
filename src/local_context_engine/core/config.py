"""
Configuration management for Local Context Engine.

Configuration is loaded from (in order of precedence):
  1. Environment variables  (LCE_EMBEDDING__MODEL, LCE_SECURITY__ENABLE_PII_MASKING, …)
  2. Project-local         .context/config.yaml
  3. User-level            ~/.config/local-context-engine/config.yaml
  4. Defaults              config/default.yaml bundled with the package

Pydantic-settings handles environment variable binding. YAML files are
merged with ``dict.update()`` so only overridden keys need to be present.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ─────────────────────────────────────────────────────────────
# Sub-models
# ─────────────────────────────────────────────────────────────


class EmbeddingConfig(BaseModel):
    """Local embedding model settings."""

    model: str = "BAAI/bge-small-en-v1.5"
    device: str = "auto"
    backend: str = "torch"
    batch_size: int = Field(32, ge=1, le=512)
    max_seq_length: int = Field(512, ge=64, le=8192)
    normalize_embeddings: bool = True
    cache_dir: Path | None = None
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    document_prefix: str = ""

    @model_validator(mode="after")
    def validate_device(self) -> "EmbeddingConfig":
        allowed = {"cpu", "cuda", "mps", "auto"}
        if self.device not in allowed:
            raise ValueError(f"device must be one of {allowed}, got '{self.device}'")
        if self.device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            except ImportError:
                self.device = "cpu"
        return self


class VectorStoreConfig(BaseModel):
    """Vector database settings."""

    backend: str = "faiss"
    dimension: int = Field(384, ge=64, le=4096)
    index_type: str = "hnsw"
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 50
    storage_path: Path = Path(".context/vectors")


class MetadataStoreConfig(BaseModel):
    """SQLite metadata store settings."""

    db_path: Path = Path(".context/metadata.db")
    wal_mode: bool = True
    pool_size: int = Field(5, ge=1, le=20)


class ChunkingConfig(BaseModel):
    """Chunk boundary and size settings."""

    min_tokens: int = Field(50, ge=10)
    target_tokens: int = Field(500, ge=100)
    max_tokens: int = Field(1200, ge=200)
    overlap_tokens: int = Field(50, ge=0)
    strategy: str = "symbol"
    tokenizer: str = "approx"

    @model_validator(mode="after")
    def validate_sizes(self) -> "ChunkingConfig":
        if self.min_tokens >= self.target_tokens:
            raise ValueError("min_tokens must be less than target_tokens")
        if self.target_tokens >= self.max_tokens:
            raise ValueError("target_tokens must be less than max_tokens")
        return self


class SecurityConfig(BaseModel):
    """Security and file blacklisting settings."""

    enable_pii_masking: bool = True
    never_index_patterns: list[str] = Field(
        default_factory=lambda: [
            "**/.env",
            "**/.env.*",
            "**/*.key",
            "**/*.pem",
            "**/*.crt",
            "**/*.cert",
            "**/*.cer",
            "**/*.p12",
            "**/*.pfx",
            "**/id_rsa",
            "**/id_ecdsa",
            "**/id_ed25519",
            "**/.ssh/**",
            "**/secrets/**",
            "**/storage/logs/**",
            "**/storage/sessions/**",
            "**/storage/framework/cache/**",
            "**/storage/framework/sessions/**",
            "**/storage/uploads/**",
            "**/public/storage/**",
            "**/vendor/**",
            "**/node_modules/**",
            "**/.git/**",
            "**/dist/**",
            "**/build/**",
            "**/.next/**",
            "**/__pycache__/**",
            "**/*.pyc",
            "**/*.log",
            "**/package-lock.json",
            "**/composer.lock",
        ]
    )
    max_file_size_bytes: int = Field(2 * 1024 * 1024, ge=1024)  # 2 MB default
    binary_extensions: list[str] = Field(
        default_factory=lambda: [
            ".exe", ".dll", ".so", ".dylib", ".bin", ".obj", ".o", ".a", ".lib",
            ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
            ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".ico", ".webp",
            ".mp4", ".mp3", ".wav", ".avi",
            ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
            ".woff", ".woff2", ".ttf", ".otf", ".eot",
            ".db", ".sqlite", ".sqlite3",
        ]
    )


class PIIMaskingConfig(BaseModel):
    """Controls which PII patterns are masked before embedding."""

    mask_email: bool = True
    mask_phone: bool = True
    mask_nic: bool = True
    mask_tin: bool = True
    mask_vat: bool = True
    mask_jwt: bool = True
    mask_api_keys: bool = True
    mask_access_tokens: bool = True
    mask_signed_urls: bool = True
    mask_ip_addresses: bool = False
    mask_credit_cards: bool = True


class SymbolGraphConfig(BaseModel):
    """Symbol graph persistence and traversal settings."""

    persist: bool = True
    storage_path: Path = Path(".context/graph.pkl")
    max_traversal_depth: int = Field(5, ge=1, le=20)


class RetrievalConfig(BaseModel):
    """Hybrid retrieval scoring and limits."""

    default_limit: int = Field(10, ge=1, le=100)
    max_limit: int = Field(50, ge=1, le=500)
    semantic_weight: float = Field(0.50, ge=0.0, le=1.0)
    bm25_weight: float = Field(0.20, ge=0.0, le=1.0)
    symbol_weight: float = Field(0.20, ge=0.0, le=1.0)
    graph_weight: float = Field(0.10, ge=0.0, le=1.0)
    min_score: float = Field(0.05, ge=0.0, le=1.0)
    candidate_multiplier: int = Field(5, ge=1, le=20)

    @model_validator(mode="after")
    def validate_weights(self) -> "RetrievalConfig":
        total = (
            self.semantic_weight
            + self.bm25_weight
            + self.symbol_weight
            + self.graph_weight
        )
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"Retrieval weights must sum to 1.0, got {total:.3f}. "
                "Adjust semantic_weight, bm25_weight, symbol_weight, graph_weight."
            )
        return self


class MemoryConfig(BaseModel):
    """Agent memory persistence settings."""

    storage_path: Path = Path(".context/memory.db")
    stale_after_days: int = Field(90, ge=1)
    max_memories_per_category: int = Field(100, ge=10)


class PerformanceConfig(BaseModel):
    """Parallelism and performance tuning."""

    scan_workers: int = Field(4, ge=1, le=32)
    parse_workers: int = Field(4, ge=1, le=32)
    embedding_batch_size: int = Field(32, ge=1, le=512)
    incremental: bool = True
    hash_algorithm: str = "md5"


class MCPServerConfig(BaseModel):
    """MCP server transport and identity settings."""

    transport: str = "stdio"
    host: str = "127.0.0.1"
    port: int = Field(8765, ge=1024, le=65535)
    name: str = "local-context-engine"
    version: str = "0.1.0"


class LoggingConfig(BaseModel):
    """Logging settings."""

    level: str = "INFO"
    format: str = "rich"
    file: Path | None = None


# ─────────────────────────────────────────────────────────────
# Root Config
# ─────────────────────────────────────────────────────────────


class EngineConfig(BaseSettings):
    """
    Root configuration model for the Local Context Engine.

    Environment variables are prefixed with ``LCE_`` and use double
    underscores as nested separators:
        LCE_EMBEDDING__MODEL=nomic-ai/nomic-embed-text-v1.5
        LCE_SECURITY__ENABLE_PII_MASKING=false
    """

    model_config = SettingsConfigDict(
        env_prefix="LCE_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    metadata_store: MetadataStoreConfig = Field(default_factory=MetadataStoreConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    pii_masking: PIIMaskingConfig = Field(default_factory=PIIMaskingConfig)
    symbol_graph: SymbolGraphConfig = Field(default_factory=SymbolGraphConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    mcp_server: MCPServerConfig = Field(default_factory=MCPServerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


# ─────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge *override* into *base*, returning a new dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


# Config keys whose values are file-system paths that should be resolved
# relative to the project root when they are not absolute.
_PATH_KEYS = {"db_path", "storage_path", "cache_dir", "file"}


def _resolve_relative_paths(data: dict[str, Any], base: Path) -> dict[str, Any]:
    """
    Walk a nested config dict and resolve relative path strings relative to *base*.

    Only string values whose key is in ``_PATH_KEYS`` and whose value is not
    already an absolute path (and not null) are rewritten.
    """
    result: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, dict):
            result[k] = _resolve_relative_paths(v, base)
        elif k in _PATH_KEYS and isinstance(v, str) and v and not Path(v).is_absolute():
            result[k] = str(base / v)
        else:
            result[k] = v
    return result


def load_config(
    project_root: Path | None = None,
    config_override: Path | None = None,
) -> EngineConfig:
    """
    Load the engine configuration.

    Priority (highest wins):
      1. Environment variables (``LCE_*``)
      2. ``config_override`` path (if supplied)
      3. ``<project_root>/.context/config.yaml``
      4. ``~/.config/local-context-engine/config.yaml``
      5. Bundled ``config/default.yaml``

    Args:
        project_root: Repository root (used to find ``.context/config.yaml``).
        config_override: Explicit path to a YAML config file.

    Returns:
        Fully-merged :class:`EngineConfig` instance.
    """
    # 1. Bundled defaults
    default_config_path = Path(__file__).parent.parent.parent.parent / "config" / "default.yaml"
    merged: dict[str, Any] = _load_yaml(default_config_path)

    # 2. User-level config
    user_config_path = (
        Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
        / "local-context-engine"
        / "config.yaml"
    )
    merged = _merge(merged, _load_yaml(user_config_path))

    # 3. Project-level config
    if project_root is not None:
        project_config = project_root / ".context" / "config.yaml"
        merged = _merge(merged, _load_yaml(project_config))

    # 4. Explicit override
    if config_override is not None:
        merged = _merge(merged, _load_yaml(config_override))

    # 5. Resolve relative data paths relative to the project root so that
    #    each indexed project stores its index inside its own .context/ dir.
    if project_root is not None:
        merged = _resolve_relative_paths(merged, project_root.resolve())

    # 6. Environment variables handled by pydantic-settings
    return EngineConfig(**merged)


# ─────────────────────────────────────────────────────────────
# Singleton helper
# ─────────────────────────────────────────────────────────────

_config_instance: EngineConfig | None = None


def get_config() -> EngineConfig:
    """Return the global singleton config, loading defaults if not yet initialised."""
    global _config_instance
    if _config_instance is None:
        _config_instance = load_config()
    return _config_instance


def set_config(config: EngineConfig) -> None:
    """Replace the global singleton config (useful for tests)."""
    global _config_instance
    _config_instance = config
