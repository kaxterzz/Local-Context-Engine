"""
Agent memory store.

Persists long-term knowledge that AI coding agents accumulate across sessions:
  - Architectural decisions
  - Coding conventions and patterns
  - Known bug workarounds
  - Project-specific naming styles
  - Preferred folder structures
  - Domain terminology

Backed by SQLite for simplicity and zero dependencies.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MemoryCategory(str, Enum):
    """Categories of agent memory entries."""

    ARCHITECTURE = "architecture"       # High-level design decisions
    CONVENTION = "convention"           # Coding patterns, naming styles
    DECISION = "decision"               # Why something was built this way
    BUG = "bug"                         # Known issues and workarounds
    API = "api"                         # External API behaviours
    DOMAIN = "domain"                   # Business domain knowledge
    TOOLING = "tooling"                 # Build, deploy, CI/CD specifics
    GENERAL = "general"                 # Uncategorised notes


class MemoryEntry:
    """A single persistent memory record."""

    def __init__(
        self,
        id: str,
        category: MemoryCategory,
        content: str,
        tags: list[str],
        created_at: datetime,
        updated_at: datetime,
        access_count: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.id = id
        self.category = category
        self.content = content
        self.tags = tags
        self.created_at = created_at
        self.updated_at = updated_at
        self.access_count = access_count
        self.metadata = metadata or {}


class AgentMemory:
    """
    SQLite-backed persistent memory store for AI coding agents.

    Usage::

        memory = AgentMemory(Path(".context/memory.db"))
        memory.init()

        memory.save_memory(
            category=MemoryCategory.ARCHITECTURE,
            content="All API responses use ResourceCollection wrappers.",
            tags=["laravel", "api", "response"],
        )

        results = memory.retrieve_memory("API response format")
    """

    def __init__(
        self,
        db_path: Path,
        stale_after_days: int = 90,
        max_per_category: int = 100,
    ) -> None:
        self._db_path = db_path
        self._stale_after_days = stale_after_days
        self._max_per_category = max_per_category
        self._conn: sqlite3.Connection | None = None

    def init(self) -> None:
        """Initialise the memory database."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self._db_path), check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._create_tables()
        logger.info("Agent memory initialised at %s", self._db_path)

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                content TEXT NOT NULL,
                tags TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                access_count INTEGER NOT NULL DEFAULT 0,
                extra_metadata TEXT DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS ix_memories_category ON memories(category);
            CREATE INDEX IF NOT EXISTS ix_memories_updated_at ON memories(updated_at);
        """)
        self._conn.commit()

    def save_memory(
        self,
        category: MemoryCategory,
        content: str,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Store a new memory entry.

        Args:
            category: :class:`MemoryCategory` enum value.
            content:  The knowledge to persist.
            tags:     Optional list of tags for filtering.
            metadata: Optional additional key/value data.

        Returns:
            The generated memory ID.
        """
        if self._conn is None:
            raise RuntimeError("Call init() before using AgentMemory.")

        memory_id = str(uuid.uuid4())
        now = datetime.now(tz=timezone.utc).isoformat()

        self._conn.execute(
            """INSERT INTO memories (id, category, content, tags, created_at, updated_at, extra_metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                memory_id,
                category.value,
                content,
                json.dumps(tags or []),
                now,
                now,
                json.dumps(metadata or {}),
            ),
        )
        self._conn.commit()
        self._evict_excess(category)
        logger.debug("Memory saved: %s (%s)", memory_id, category.value)
        return memory_id

    def retrieve_memory(
        self,
        query: str,
        category: MemoryCategory | None = None,
        tags: list[str] | None = None,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """
        Retrieve memories matching the query.

        Uses simple SQLite FTS-style LIKE search. For semantic search,
        extend this with an embedding index on memory content.

        Args:
            query:    Search string matched against content.
            category: Optional category filter.
            tags:     Optional tag filter (any tag must match).
            limit:    Maximum results.

        Returns:
            List of :class:`MemoryEntry` sorted by recency.
        """
        if self._conn is None:
            raise RuntimeError("Call init() before using AgentMemory.")

        where_clauses = ["content LIKE ?"]
        params: list[Any] = [f"%{query}%"]

        if category:
            where_clauses.append("category = ?")
            params.append(category.value)

        if tags:
            tag_conditions = " OR ".join(["tags LIKE ?" for _ in tags])
            where_clauses.append(f"({tag_conditions})")
            params.extend(f'%"{tag}"%' for tag in tags)

        sql = f"""
            SELECT * FROM memories
            WHERE {' AND '.join(where_clauses)}
            ORDER BY updated_at DESC
            LIMIT ?
        """
        params.append(limit)

        rows = self._conn.execute(sql, params).fetchall()

        results = [self._row_to_entry(row) for row in rows]

        # Increment access counts
        if results:
            ids = [e.id for e in results]
            self._conn.execute(
                f"UPDATE memories SET access_count = access_count + 1 "
                f"WHERE id IN ({','.join('?' for _ in ids)})",
                ids,
            )
            self._conn.commit()

        return results

    def update_memory(self, memory_id: str, content: str) -> bool:
        """Update an existing memory's content."""
        if self._conn is None:
            raise RuntimeError("Call init() before using AgentMemory.")

        now = datetime.now(tz=timezone.utc).isoformat()
        result = self._conn.execute(
            "UPDATE memories SET content = ?, updated_at = ? WHERE id = ?",
            (content, now, memory_id),
        )
        self._conn.commit()
        return result.rowcount > 0

    def forget_memory(self, memory_id: str) -> bool:
        """Delete a specific memory by ID."""
        if self._conn is None:
            raise RuntimeError("Call init() before using AgentMemory.")

        result = self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._conn.commit()
        return result.rowcount > 0

    def forget_stale(self) -> int:
        """Delete memories older than ``stale_after_days`` with low access counts."""
        if self._conn is None:
            raise RuntimeError("Call init() before using AgentMemory.")

        cutoff = (
            datetime.now(tz=timezone.utc) - timedelta(days=self._stale_after_days)
        ).isoformat()

        result = self._conn.execute(
            "DELETE FROM memories WHERE updated_at < ? AND access_count < 3",
            (cutoff,),
        )
        self._conn.commit()
        count = result.rowcount
        if count > 0:
            logger.info("Forgot %d stale memories.", count)
        return count

    def list_all(
        self, category: MemoryCategory | None = None, limit: int = 50
    ) -> list[MemoryEntry]:
        """List all memories, optionally filtered by category."""
        if self._conn is None:
            raise RuntimeError("Call init() before using AgentMemory.")

        if category:
            rows = self._conn.execute(
                "SELECT * FROM memories WHERE category = ? ORDER BY updated_at DESC LIMIT ?",
                (category.value, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM memories ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()

        return [self._row_to_entry(row) for row in rows]

    def stats(self) -> dict[str, Any]:
        """Return memory statistics."""
        if self._conn is None:
            raise RuntimeError("Call init() before using AgentMemory.")

        total = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        by_category = self._conn.execute(
            "SELECT category, COUNT(*) FROM memories GROUP BY category"
        ).fetchall()

        return {
            "total": total,
            "by_category": {row[0]: row[1] for row in by_category},
        }

    def close(self) -> None:
        if self._conn:
            self._conn.close()

    # ── Internal helpers ───────────────────────────────────────

    def _row_to_entry(self, row: sqlite3.Row) -> MemoryEntry:
        return MemoryEntry(
            id=row["id"],
            category=MemoryCategory(row["category"]),
            content=row["content"],
            tags=json.loads(row["tags"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            access_count=row["access_count"],
            metadata=json.loads(row["extra_metadata"] or "{}"),
        )

    def _evict_excess(self, category: MemoryCategory) -> None:
        """Evict oldest low-access memories when category limit is exceeded."""
        count = self._conn.execute(
            "SELECT COUNT(*) FROM memories WHERE category = ?", (category.value,)
        ).fetchone()[0]

        if count > self._max_per_category:
            excess = count - self._max_per_category
            self._conn.execute(
                """DELETE FROM memories WHERE id IN (
                       SELECT id FROM memories WHERE category = ?
                       ORDER BY access_count ASC, updated_at ASC LIMIT ?
                   )""",
                (category.value, excess),
            )
            self._conn.commit()

    @classmethod
    def from_config(cls, config: "MemoryConfig") -> "AgentMemory":  # type: ignore[name-defined]
        from local_context_engine.core.config import MemoryConfig  # noqa: PLC0415

        assert isinstance(config, MemoryConfig)
        return cls(
            db_path=config.storage_path,
            stale_after_days=config.stale_after_days,
            max_per_category=config.max_memories_per_category,
        )
