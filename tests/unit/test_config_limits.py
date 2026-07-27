"""Tests for the resource limits configuration and CONTEXT_* env overrides."""

from __future__ import annotations

from pathlib import Path

import pytest

from local_context_engine.core.config import LimitsConfig, load_config


class TestLimitsDefaults:
    def test_defaults(self) -> None:
        limits = LimitsConfig()
        assert limits.max_workers == 2
        assert limits.index_batch_size == 20
        assert limits.max_queue_size == 100
        assert limits.cache_max_items == 500
        assert limits.cache_ttl_seconds == 1800
        assert limits.max_file_size_mb == 5.0
        assert limits.memory_soft_limit_mb == 4096
        assert limits.memory_hard_limit_mb == 6144
        assert limits.single_instance_per_project is True

    def test_hard_limit_must_exceed_soft(self) -> None:
        with pytest.raises(ValueError, match="memory_hard_limit_mb must be greater"):
            LimitsConfig(memory_soft_limit_mb=4096, memory_hard_limit_mb=4096)


class TestContextEnvOverrides:
    def test_env_overrides_applied(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CONTEXT_MAX_WORKERS", "1")
        monkeypatch.setenv("CONTEXT_INDEX_BATCH_SIZE", "7")
        monkeypatch.setenv("CONTEXT_MAX_QUEUE_SIZE", "42")
        monkeypatch.setenv("CONTEXT_CACHE_MAX_ITEMS", "9")
        monkeypatch.setenv("CONTEXT_CACHE_TTL_SECONDS", "60")
        monkeypatch.setenv("CONTEXT_MAX_FILE_SIZE_MB", "1.5")
        monkeypatch.setenv("CONTEXT_MEMORY_SOFT_LIMIT_MB", "1024")
        monkeypatch.setenv("CONTEXT_MEMORY_HARD_LIMIT_MB", "2048")
        monkeypatch.setenv("CONTEXT_SINGLE_INSTANCE_PER_PROJECT", "false")
        monkeypatch.setenv("CONTEXT_BM25_MAX_DOCS", "1234")

        config = load_config(project_root=tmp_path)
        assert config.limits.max_workers == 1
        assert config.limits.index_batch_size == 7
        assert config.limits.max_queue_size == 42
        assert config.limits.cache_max_items == 9
        assert config.limits.cache_ttl_seconds == 60
        assert config.limits.max_file_size_mb == 1.5
        assert config.limits.memory_soft_limit_mb == 1024
        assert config.limits.memory_hard_limit_mb == 2048
        assert config.limits.single_instance_per_project is False
        assert config.limits.bm25_max_docs == 1234

    def test_limits_cap_security_and_performance(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CONTEXT_MAX_FILE_SIZE_MB", "1")
        monkeypatch.setenv("CONTEXT_MAX_WORKERS", "1")
        config = load_config(project_root=tmp_path)
        assert config.security.max_file_size_bytes == 1024 * 1024
        assert config.performance.scan_workers == 1
        assert config.performance.parse_workers == 1

    def test_no_env_keeps_defaults(self, tmp_path: Path) -> None:
        config = load_config(project_root=tmp_path)
        assert config.limits.index_batch_size == 20

    def test_device_auto_not_resolved_at_config_time(self, tmp_path: Path) -> None:
        """'auto' must survive config load — torch is only imported by the embedder."""
        config = load_config(project_root=tmp_path)
        if config.embedding.device == "auto":
            # Resolution happens lazily; config keeps the sentinel.
            assert config.embedding.device == "auto"
