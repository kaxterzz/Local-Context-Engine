"""Unit tests for metadata repository bulk operations."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from local_context_engine.metadata_store.repositories import ChunkRepository


@pytest.mark.asyncio
async def test_mark_embedded_batches_large_id_lists() -> None:
    session = AsyncMock()
    repository = ChunkRepository(session)

    await repository.mark_embedded([f"chunk-{index}" for index in range(36_241)])

    # 73 statements of at most 500 IDs instead of one 36k-variable statement.
    assert session.execute.await_count == 73
