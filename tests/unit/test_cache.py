"""Tests for the bounded TTL + LRU cache."""

from __future__ import annotations

import time

from local_context_engine.core.cache import TTLLRUCache


class TestTTLLRUCache:
    def test_basic_put_get(self) -> None:
        cache = TTLLRUCache(max_items=10, ttl_seconds=60)
        cache.put("a", 1)
        assert cache.get("a") == 1
        assert cache.get("missing") is None
        assert cache.get("missing", default="x") == "x"

    def test_lru_eviction(self) -> None:
        cache = TTLLRUCache(max_items=3, ttl_seconds=60)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        cache.get("a")  # refresh 'a' → 'b' becomes least recently used
        cache.put("d", 4)
        assert cache.get("a") == 1
        assert cache.get("b") is None  # evicted
        assert cache.get("c") == 3
        assert cache.get("d") == 4
        assert len(cache) == 3

    def test_never_exceeds_max_items(self) -> None:
        cache = TTLLRUCache(max_items=50, ttl_seconds=60)
        for i in range(500):
            cache.put(i, i)
            assert len(cache) <= 50
        assert len(cache) == 50

    def test_ttl_expiry(self) -> None:
        cache = TTLLRUCache(max_items=10, ttl_seconds=0.05)
        cache.put("a", 1)
        assert cache.get("a") == 1
        time.sleep(0.08)
        assert cache.get("a") is None
        assert len(cache) == 0  # expired entry was purged on access

    def test_purge_expired(self) -> None:
        cache = TTLLRUCache(max_items=10, ttl_seconds=0.05)
        cache.put("a", 1)
        cache.put("b", 2)
        time.sleep(0.08)
        cache.put("c", 3)
        assert cache.purge_expired() == 2
        assert len(cache) == 1

    def test_zero_capacity_stores_nothing(self) -> None:
        cache = TTLLRUCache(max_items=0, ttl_seconds=60)
        cache.put("a", 1)
        assert cache.get("a") is None
        assert len(cache) == 0

    def test_clear(self) -> None:
        cache = TTLLRUCache(max_items=10, ttl_seconds=60)
        cache.put("a", 1)
        cache.clear()
        assert len(cache) == 0
        assert cache.get("a") is None

    def test_hit_miss_counters(self) -> None:
        cache = TTLLRUCache(max_items=10, ttl_seconds=60)
        cache.put("a", 1)
        cache.get("a")
        cache.get("b")
        assert cache.hits == 1
        assert cache.misses == 1
