"""
Data access layer (Repository pattern) for the metadata store.

Each repository handles CRUD + query operations for a single aggregate root.
All methods are async and operate within a provided :class:`AsyncSession`.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Sequence

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from local_context_engine.core.types import (
    Chunk,
    FileRecord,
    IndexingStatus,
    Language,
    Relationship,
    RelationshipType,
    Symbol,
    SymbolType,
)
from local_context_engine.metadata_store.models import (
    ChunkModel,
    FileModel,
    RelationshipModel,
    SymbolModel,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Mappers (ORM ↔ domain types)
# ─────────────────────────────────────────────────────────────


def _file_to_domain(m: FileModel) -> FileRecord:
    return FileRecord(
        id=m.id,
        path=m.path,
        absolute_path=m.absolute_path,
        language=Language(m.language),
        size_bytes=m.size_bytes,
        hash=m.hash,
        modified_at=m.modified_at,
        line_count=m.line_count,
        indexed_at=m.indexed_at,
        status=IndexingStatus(m.status),
        framework=m.framework,
    )


def _symbol_to_domain(m: SymbolModel) -> Symbol:
    return Symbol(
        id=m.id,
        file_id=m.file_id,
        file_path=m.file.path if m.file else "",
        name=m.name,
        qualified_name=m.qualified_name,
        symbol_type=SymbolType(m.symbol_type),
        line_start=m.line_start,
        line_end=m.line_end,
        language=Language(m.language),
        visibility=m.visibility,
        docstring=m.docstring,
        parent_id=m.parent_id,
        parent_name=m.parent_name,
        metadata=m.symbol_metadata or {},
    )


def _chunk_to_domain(m: ChunkModel) -> Chunk:
    return Chunk(
        id=m.id,
        file_id=m.file_id,
        file_path=m.file.path if m.file else "",
        symbol_id=m.symbol_id,
        content=m.content,
        content_masked=m.content_masked,
        language=Language(m.language),
        symbol_name=m.symbol_name,
        symbol_type=SymbolType(m.symbol_type) if m.symbol_type else None,
        line_start=m.line_start,
        line_end=m.line_end,
        hash=m.hash,
        token_count=m.token_count,
        chunk_index=m.chunk_index,
        metadata=m.chunk_metadata or {},
    )


def _relationship_to_domain(m: RelationshipModel) -> Relationship:
    return Relationship(
        id=m.id,
        source_symbol_id=m.source_symbol_id,
        source_file_path=m.source_file_path,
        target_symbol_id=m.target_symbol_id,
        target_name=m.target_name,
        relationship_type=RelationshipType(m.relationship_type),
        confidence=m.confidence,
        metadata=m.rel_metadata or {},
    )


# ─────────────────────────────────────────────────────────────
# FileRepository
# ─────────────────────────────────────────────────────────────


class FileRepository:
    """CRUD operations for :class:`FileModel` / :class:`FileRecord`."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert(self, record: FileRecord) -> None:
        """Insert or update a file record."""
        existing = await self._session.get(FileModel, record.id)
        if existing:
            existing.path = record.path
            existing.absolute_path = record.absolute_path
            existing.language = record.language.value
            existing.framework = record.framework
            existing.size_bytes = record.size_bytes
            existing.line_count = record.line_count
            existing.hash = record.hash
            existing.modified_at = record.modified_at
            existing.indexed_at = record.indexed_at
            existing.status = record.status.value
            existing.is_deleted = False
        else:
            self._session.add(
                FileModel(
                    id=record.id,
                    path=record.path,
                    absolute_path=record.absolute_path,
                    language=record.language.value,
                    framework=record.framework,
                    size_bytes=record.size_bytes,
                    line_count=record.line_count,
                    hash=record.hash,
                    modified_at=record.modified_at,
                    indexed_at=record.indexed_at,
                    status=record.status.value,
                )
            )
        await self._session.flush()

    async def get_by_path(self, path: str) -> FileRecord | None:
        """Return the file record for the given relative path, or None."""
        result = await self._session.execute(
            select(FileModel).where(FileModel.path == path, FileModel.is_deleted == False)  # noqa: E712
        )
        model = result.scalar_one_or_none()
        return _file_to_domain(model) if model else None

    async def get_by_id(self, file_id: str) -> FileRecord | None:
        model = await self._session.get(FileModel, file_id)
        return _file_to_domain(model) if model else None

    async def get_all(self) -> list[FileRecord]:
        result = await self._session.execute(
            select(FileModel).where(FileModel.is_deleted == False)  # noqa: E712
        )
        return [_file_to_domain(m) for m in result.scalars()]

    async def get_hash(self, path: str) -> str | None:
        """Return just the stored hash for a given path (fast change detection)."""
        result = await self._session.execute(
            select(FileModel.hash).where(
                FileModel.path == path, FileModel.is_deleted == False  # noqa: E712
            )
        )
        return result.scalar_one_or_none()

    async def mark_deleted(self, path: str) -> None:
        await self._session.execute(
            update(FileModel).where(FileModel.path == path).values(is_deleted=True)
        )

    async def delete_by_id(self, file_id: str) -> None:
        await self._session.execute(
            delete(FileModel).where(FileModel.id == file_id)
        )

    async def count(self) -> int:
        from sqlalchemy import func

        result = await self._session.execute(
            select(func.count()).where(FileModel.is_deleted == False)  # noqa: E712
            .select_from(FileModel)
        )
        return result.scalar_one() or 0

    async def language_counts(self) -> dict[str, int]:
        from sqlalchemy import func

        result = await self._session.execute(
            select(FileModel.language, func.count())
            .where(FileModel.is_deleted == False)  # noqa: E712
            .group_by(FileModel.language)
        )
        return {row[0]: row[1] for row in result}


# ─────────────────────────────────────────────────────────────
# SymbolRepository
# ─────────────────────────────────────────────────────────────


class SymbolRepository:
    """CRUD operations for :class:`SymbolModel` / :class:`Symbol`."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def insert_many(self, symbols: Sequence[Symbol]) -> None:
        """Bulk insert symbols (does not update existing)."""
        self._session.add_all(
            [
                SymbolModel(
                    id=s.id,
                    file_id=s.file_id,
                    name=s.name,
                    qualified_name=s.qualified_name,
                    symbol_type=s.symbol_type.value,
                    language=s.language.value,
                    line_start=s.line_start,
                    line_end=s.line_end,
                    visibility=s.visibility,
                    docstring=s.docstring,
                    parent_id=s.parent_id,
                    parent_name=s.parent_name,
                    symbol_metadata=s.metadata,
                )
                for s in symbols
            ]
        )
        await self._session.flush()

    async def delete_by_file(self, file_id: str) -> None:
        await self._session.execute(
            delete(SymbolModel).where(SymbolModel.file_id == file_id)
        )

    async def find_by_name(self, name: str, language: str | None = None) -> list[Symbol]:
        stmt = (
            select(SymbolModel)
            .where(SymbolModel.name.ilike(f"%{name}%"))
            .options(selectinload(SymbolModel.file))
        )
        if language:
            stmt = stmt.where(SymbolModel.language == language)
        result = await self._session.execute(stmt)
        return [_symbol_to_domain(m) for m in result.scalars().all()]

    async def find_by_type(
        self, symbol_type: str, file_id: str | None = None
    ) -> list[Symbol]:
        stmt = (
            select(SymbolModel)
            .where(SymbolModel.symbol_type == symbol_type)
            .options(selectinload(SymbolModel.file))
        )
        if file_id:
            stmt = stmt.where(SymbolModel.file_id == file_id)
        result = await self._session.execute(stmt)
        return [_symbol_to_domain(m) for m in result.scalars()]

    async def get_by_id(self, symbol_id: str) -> Symbol | None:
        result = await self._session.execute(
            select(SymbolModel)
            .where(SymbolModel.id == symbol_id)
            .options(selectinload(SymbolModel.file))
        )
        model = result.scalar_one_or_none()
        return _symbol_to_domain(model) if model else None

    async def get_by_file_id(self, file_id: str) -> list[Symbol]:
        """Return all symbols for a given file, ordered by line number."""
        result = await self._session.execute(
            select(SymbolModel)
            .where(SymbolModel.file_id == file_id)
            .options(selectinload(SymbolModel.file))
            .order_by(SymbolModel.line_start)
        )
        return [_symbol_to_domain(m) for m in result.scalars()]

    async def get_all(self) -> list[Symbol]:
        """Return all symbols — used for building the symbol graph."""
        result = await self._session.execute(
            select(SymbolModel).options(selectinload(SymbolModel.file))
        )
        return [_symbol_to_domain(m) for m in result.scalars()]

    async def count(self) -> int:
        from sqlalchemy import func

        result = await self._session.execute(
            select(func.count()).select_from(SymbolModel)
        )
        return result.scalar_one() or 0


# ─────────────────────────────────────────────────────────────
# ChunkRepository
# ─────────────────────────────────────────────────────────────


class ChunkRepository:
    """CRUD operations for :class:`ChunkModel` / :class:`Chunk`."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def insert_many(self, chunks: Sequence[Chunk]) -> None:
        self._session.add_all(
            [
                ChunkModel(
                    id=c.id,
                    file_id=c.file_id,
                    symbol_id=c.symbol_id,
                    content=c.content,
                    content_masked=c.content_masked,
                    language=c.language.value,
                    symbol_name=c.symbol_name,
                    symbol_type=c.symbol_type.value if c.symbol_type else None,
                    line_start=c.line_start,
                    line_end=c.line_end,
                    hash=c.hash,
                    token_count=c.token_count,
                    chunk_index=c.chunk_index,
                    chunk_metadata=c.metadata,
                )
                for c in chunks
            ]
        )
        await self._session.flush()

    async def delete_by_file(self, file_id: str) -> None:
        await self._session.execute(
            delete(ChunkModel).where(ChunkModel.file_id == file_id)
        )

    async def get_by_id(self, chunk_id: str) -> Chunk | None:
        result = await self._session.execute(
            select(ChunkModel)
            .where(ChunkModel.id == chunk_id)
            .options(selectinload(ChunkModel.file))
        )
        model = result.scalar_one_or_none()
        return _chunk_to_domain(model) if model else None

    async def get_pending_embedding(self, limit: int = 500) -> list[Chunk]:
        """Return chunks not yet embedded, ordered by file."""
        result = await self._session.execute(
            select(ChunkModel)
            .where(ChunkModel.is_embedded == False)  # noqa: E712
            .options(selectinload(ChunkModel.file))
            .limit(limit)
        )
        return [_chunk_to_domain(m) for m in result.scalars()]

    async def get_ids_by_file_path(self, file_path: str) -> list[str]:
        """Return all chunk IDs for the given file (relative path)."""
        result = await self._session.execute(
            select(ChunkModel.id)
            .join(FileModel, ChunkModel.file_id == FileModel.id)
            .where(FileModel.path == file_path)
        )
        return [row[0] for row in result.all()]

    async def mark_embedded(self, chunk_ids: list[str]) -> None:
        await self._session.execute(
            update(ChunkModel)
            .where(ChunkModel.id.in_(chunk_ids))
            .values(is_embedded=True)
        )

    async def count(self) -> int:
        from sqlalchemy import func

        result = await self._session.execute(
            select(func.count()).select_from(ChunkModel)
        )
        return result.scalar_one() or 0

    async def get_content_by_ids(self, chunk_ids: list[str]) -> dict[str, str]:
        """Return {chunk_id: content} for BM25 corpus building."""
        result = await self._session.execute(
            select(ChunkModel.id, ChunkModel.content).where(
                ChunkModel.id.in_(chunk_ids)
            )
        )
        return {row[0]: row[1] for row in result}

    async def get_all_contents(self) -> list[tuple[str, str]]:
        """Return [(chunk_id, content)] for full BM25 index rebuild."""
        result = await self._session.execute(
            select(ChunkModel.id, ChunkModel.content)
        )
        return list(result.all())


# ─────────────────────────────────────────────────────────────
# RelationshipRepository
# ─────────────────────────────────────────────────────────────


class RelationshipRepository:
    """CRUD operations for :class:`RelationshipModel` / :class:`Relationship`."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def insert_many(self, relationships: Sequence[Relationship]) -> None:
        self._session.add_all(
            [
                RelationshipModel(
                    id=r.id,
                    source_symbol_id=r.source_symbol_id,
                    source_file_path=r.source_file_path,
                    target_symbol_id=r.target_symbol_id,
                    target_name=r.target_name,
                    relationship_type=r.relationship_type.value,
                    confidence=r.confidence,
                    rel_metadata=r.metadata,
                )
                for r in relationships
            ]
        )
        await self._session.flush()

    async def delete_by_source_file(self, file_path: str) -> None:
        await self._session.execute(
            delete(RelationshipModel).where(
                RelationshipModel.source_file_path == file_path
            )
        )

    async def find_by_source(self, symbol_id: str) -> list[Relationship]:
        result = await self._session.execute(
            select(RelationshipModel).where(
                RelationshipModel.source_symbol_id == symbol_id
            )
        )
        return [_relationship_to_domain(m) for m in result.scalars()]

    async def find_by_target_name(self, target_name: str) -> list[Relationship]:
        result = await self._session.execute(
            select(RelationshipModel).where(
                RelationshipModel.target_name == target_name
            )
        )
        return [_relationship_to_domain(m) for m in result.scalars()]

    async def get_all(self) -> list[Relationship]:
        result = await self._session.execute(select(RelationshipModel))
        return [_relationship_to_domain(m) for m in result.scalars()]
