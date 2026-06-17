"""
SQLite-backed metadata store.

Stores file records, symbols, chunks, relationships, and indexing history.
All data is local. The store supports incremental re-indexing via hash comparison.
"""

from local_context_engine.metadata_store.database import Database
from local_context_engine.metadata_store.repositories import (
    ChunkRepository,
    FileRepository,
    RelationshipRepository,
    SymbolRepository,
)

__all__ = [
    "Database",
    "FileRepository",
    "SymbolRepository",
    "ChunkRepository",
    "RelationshipRepository",
]
