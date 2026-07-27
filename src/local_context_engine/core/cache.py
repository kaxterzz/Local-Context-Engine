"""
Bounded TTL + LRU cache.

Used for hot-path caching (e.g. MCP chunk reads) where an unbounded dict
would grow with every unique request. Eviction is two-fold:

  - LRU: when ``max_items`` is exceeded, the least recently used entry goes.
  - TTL: entries older than ``ttl_seconds`` are treated as misses and purged.

Thread-safe; cheap enough for per-request use.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Hashable
from typing import Any


class TTLLRUCache:
    """A small thread-safe cache with LRU eviction and TTL expiry."""

    def __init__(self, max_items: int = 500, ttl_seconds: float = 1800.0) -> None:
        self._max_items = max_items
        self._ttl = ttl_seconds
        self._data: OrderedDict[Hashable, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: Hashable, default: Any = None) -> Any:
        now = time.monotonic()
        with self._lock:
            item = self._data.get(key)
            if item is None:
                self.misses += 1
                return default
            stored_at, value = item
            if now - stored_at > self._ttl:
                del self._data[key]
                self.misses += 1
                return default
            self._data.move_to_end(key)
            self.hits += 1
            return value

    def put(self, key: Hashable, value: Any) -> None:
        if self._max_items <= 0:
            return
        with self._lock:
            self._data[key] = (time.monotonic(), value)
            self._data.move_to_end(key)
            while len(self._data) > self._max_items:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def purge_expired(self) -> int:
        """Drop all expired entries; returns how many were removed."""
        now = time.monotonic()
        with self._lock:
            expired = [
                k for k, (stored_at, _) in self._data.items()
                if now - stored_at > self._ttl
            ]
            for k in expired:
                del self._data[k]
        return len(expired)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def __contains__(self, key: Hashable) -> bool:
        return self.get(key, default=_SENTINEL) is not _SENTINEL


_SENTINEL = object()
