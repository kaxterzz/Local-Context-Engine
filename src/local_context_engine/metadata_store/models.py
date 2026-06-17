"""
SQLAlchemy ORM models for the Local Context Engine metadata store.

All models map to SQLite tables. Relationships between ORM models use
``back_populates`` for bidirectional navigation without circular imports.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Shared declarative base for all metadata ORM models."""


# ─────────────────────────────────────────────────────────────
# Files
# ─────────────────────────────────────────────────────────────


class FileModel(Base):
    """
    Tracks every source file known to the index.

    ``hash`` is recomputed on each scan; if it changes, the file is
    re-parsed and re-embedded. Deleted files are soft-deleted via ``is_deleted``.
    """

    __tablename__ = "files"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    path: Mapped[str] = mapped_column(String(4096), nullable=False)  # Relative to repo root
    absolute_path: Mapped[str] = mapped_column(String(4096), nullable=False)
    language: Mapped[str] = mapped_column(String(32), nullable=False)
    framework: Mapped[str | None] = mapped_column(String(64), nullable=True)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    line_count: Mapped[int] = mapped_column(Integer, default=0)
    hash: Mapped[str] = mapped_column(String(64), nullable=False)
    modified_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="new")
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False)

    # Relationships
    symbols: Mapped[list[SymbolModel]] = relationship(
        "SymbolModel", back_populates="file", cascade="all, delete-orphan"
    )
    chunks: Mapped[list[ChunkModel]] = relationship(
        "ChunkModel", back_populates="file", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_files_path", "path"),
        Index("ix_files_language", "language"),
        Index("ix_files_hash", "hash"),
        Index("ix_files_status", "status"),
        UniqueConstraint("path", name="uq_files_path"),
    )


# ─────────────────────────────────────────────────────────────
# Symbols
# ─────────────────────────────────────────────────────────────


class SymbolModel(Base):
    """
    A named code entity (class, function, component, etc.) extracted from a file.

    ``qualified_name`` is the fully namespaced name used for cross-file references.
    """

    __tablename__ = "symbols"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    file_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("files.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(512), nullable=False)
    qualified_name: Mapped[str] = mapped_column(String(1024), nullable=False)
    symbol_type: Mapped[str] = mapped_column(String(64), nullable=False)
    language: Mapped[str] = mapped_column(String(32), nullable=False)
    line_start: Mapped[int] = mapped_column(Integer, nullable=False)
    line_end: Mapped[int] = mapped_column(Integer, nullable=False)
    visibility: Mapped[str | None] = mapped_column(String(16), nullable=True)
    docstring: Mapped[str | None] = mapped_column(Text, nullable=True)
    parent_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("symbols.id"), nullable=True
    )
    parent_name: Mapped[str | None] = mapped_column(String(512), nullable=True)
    symbol_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Relationships
    file: Mapped[FileModel] = relationship("FileModel", back_populates="symbols")
    chunks: Mapped[list[ChunkModel]] = relationship(
        "ChunkModel", back_populates="symbol", foreign_keys="ChunkModel.symbol_id"
    )
    outgoing_relationships: Mapped[list[RelationshipModel]] = relationship(
        "RelationshipModel",
        back_populates="source_symbol",
        foreign_keys="RelationshipModel.source_symbol_id",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_symbols_file_id", "file_id"),
        Index("ix_symbols_name", "name"),
        Index("ix_symbols_qualified_name", "qualified_name"),
        Index("ix_symbols_type", "symbol_type"),
        Index("ix_symbols_language", "language"),
    )


# ─────────────────────────────────────────────────────────────
# Chunks
# ─────────────────────────────────────────────────────────────


class ChunkModel(Base):
    """
    A bounded segment of source code ready for embedding.

    ``content`` stores original text; ``content_masked`` stores PII-scrubbed
    text that was used to generate the embedding.
    """

    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    file_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("files.id", ondelete="CASCADE"), nullable=False
    )
    symbol_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("symbols.id", ondelete="SET NULL"), nullable=True
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_masked: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str] = mapped_column(String(32), nullable=False)
    symbol_name: Mapped[str | None] = mapped_column(String(512), nullable=True)
    symbol_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    line_start: Mapped[int] = mapped_column(Integer, nullable=False)
    line_end: Mapped[int] = mapped_column(Integer, nullable=False)
    hash: Mapped[str] = mapped_column(String(64), nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    is_embedded: Mapped[bool] = mapped_column(Boolean, default=False)
    chunk_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Relationships
    file: Mapped[FileModel] = relationship("FileModel", back_populates="chunks")
    symbol: Mapped[SymbolModel | None] = relationship(
        "SymbolModel", back_populates="chunks", foreign_keys=[symbol_id]
    )

    __table_args__ = (
        Index("ix_chunks_file_id", "file_id"),
        Index("ix_chunks_symbol_id", "symbol_id"),
        Index("ix_chunks_hash", "hash"),
        Index("ix_chunks_language", "language"),
        Index("ix_chunks_is_embedded", "is_embedded"),
    )


# ─────────────────────────────────────────────────────────────
# Relationships
# ─────────────────────────────────────────────────────────────


class RelationshipModel(Base):
    """
    A directed edge in the symbol graph.

    ``target_symbol_id`` is resolved after the full indexing pass; it may be
    NULL for forward references that have not yet been resolved.
    """

    __tablename__ = "relationships"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_symbol_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("symbols.id", ondelete="CASCADE"), nullable=False
    )
    source_file_path: Mapped[str] = mapped_column(String(4096), nullable=False)
    target_symbol_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    target_name: Mapped[str] = mapped_column(String(512), nullable=False)
    relationship_type: Mapped[str] = mapped_column(String(64), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    rel_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Relationships
    source_symbol: Mapped[SymbolModel] = relationship(
        "SymbolModel",
        back_populates="outgoing_relationships",
        foreign_keys=[source_symbol_id],
    )

    __table_args__ = (
        Index("ix_rel_source", "source_symbol_id"),
        Index("ix_rel_target_name", "target_name"),
        Index("ix_rel_type", "relationship_type"),
    )


# ─────────────────────────────────────────────────────────────
# Indexing Run Log
# ─────────────────────────────────────────────────────────────


class IndexingRunModel(Base):
    """Records the outcome of each indexing run for diagnostics."""

    __tablename__ = "indexing_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    repository_path: Mapped[str] = mapped_column(String(4096), nullable=False)
    total_files_scanned: Mapped[int] = mapped_column(Integer, default=0)
    files_indexed: Mapped[int] = mapped_column(Integer, default=0)
    files_skipped: Mapped[int] = mapped_column(Integer, default=0)
    files_failed: Mapped[int] = mapped_column(Integer, default=0)
    total_chunks: Mapped[int] = mapped_column(Integer, default=0)
    total_vectors: Mapped[int] = mapped_column(Integer, default=0)
    embedding_model: Mapped[str | None] = mapped_column(String(256), nullable=True)
    is_incremental: Mapped[bool] = mapped_column(Boolean, default=False)
    error_log: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (Index("ix_runs_started_at", "started_at"),)
