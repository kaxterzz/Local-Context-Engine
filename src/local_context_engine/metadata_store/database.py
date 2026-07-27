"""
Database connection management for the metadata store.

Uses SQLAlchemy 2.0 with async SQLite via ``aiosqlite``.
The connection string always points to a local file — no network databases.
"""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from local_context_engine.metadata_store.models import Base

logger = logging.getLogger(__name__)


class Database:
    """
    Async SQLite database connection manager.

    Usage::

        db = Database(path)
        await db.init()

        async with db.session() as session:
            # use session
            ...

        await db.close()
    """

    def __init__(self, db_path: Path, wal_mode: bool = True) -> None:
        self._db_path = db_path
        self._wal_mode = wal_mode
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None

    async def init(self) -> None:
        """Create the database file, run DDL migrations, and configure pragmas."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite+aiosqlite:///{self._db_path.as_posix()}"

        self._engine = create_async_engine(
            url,
            echo=False,
            pool_pre_ping=True,
        )

        # Register synchronous SQLite pragmas via the connection event
        @event.listens_for(self._engine.sync_engine, "connect")
        def _set_pragmas(dbapi_connection: object, _: object) -> None:
            cursor = dbapi_connection.cursor()  # type: ignore[union-attr]
            cursor.execute("PRAGMA journal_mode=WAL") if self._wal_mode else None
            cursor.execute("PRAGMA synchronous=NORMAL")
            # 16 MB page cache per connection — the previous 64 MB multiplied
            # across pooled connections and processes was a significant fixed
            # RSS cost for marginal benefit on an indexing workload.
            cursor.execute("PRAGMA cache_size=-16384")
            cursor.execute("PRAGMA temp_store=MEMORY")
            cursor.execute("PRAGMA mmap_size=268435456") # 256 MB mmap (file-backed, evictable)
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        # Create all tables
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        logger.info("Database initialised at %s", self._db_path)

    def session(self) -> AsyncSession:
        """Return a new :class:`AsyncSession` (use as async context manager)."""
        if self._session_factory is None:
            raise RuntimeError("Database.init() must be called before accessing sessions.")
        return self._session_factory()

    async def close(self) -> None:
        """Dispose the connection pool and release all resources."""
        if self._engine is not None:
            await self._engine.dispose()
            logger.debug("Database connection pool closed.")

    async def vacuum(self) -> None:
        """Run VACUUM to reclaim space after bulk deletes."""
        async with self.session() as session:
            await session.execute(text("VACUUM"))
            await session.commit()

    @property
    def path(self) -> Path:
        return self._db_path

    @classmethod
    def from_config(cls, config: "MetadataStoreConfig") -> "Database":  # type: ignore[name-defined]
        """Create a :class:`Database` from a :class:`MetadataStoreConfig`."""
        from local_context_engine.core.config import MetadataStoreConfig  # noqa: PLC0415

        assert isinstance(config, MetadataStoreConfig)
        return cls(db_path=config.db_path, wal_mode=config.wal_mode)
