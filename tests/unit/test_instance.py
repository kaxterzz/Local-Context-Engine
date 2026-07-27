"""Tests for single-instance locking, the process registry, and memory limits."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from local_context_engine.core import instance as instance_mod
from local_context_engine.core.instance import (
    InstanceRegistry,
    MemoryMonitor,
    MemoryPressure,
    ProjectLock,
    ProjectLockError,
    canonical_project_path,
)

# ─────────────────────────────────────────────────────────────
# ProjectLock
# ─────────────────────────────────────────────────────────────


class TestProjectLock:
    def test_acquire_and_release(self, tmp_path: Path) -> None:
        lock = ProjectLock(tmp_path, role="index")
        assert lock.acquire() is True
        assert lock.held
        info = lock.holder_info()
        assert info is not None
        assert info["pid"] == os.getpid()
        lock.release()
        assert not lock.held

    def test_duplicate_acquire_fails(self, tmp_path: Path) -> None:
        first = ProjectLock(tmp_path, role="mcp")
        second = ProjectLock(tmp_path, role="mcp")
        assert first.acquire() is True
        # A second open file description cannot flock the same file.
        assert second.acquire() is False
        first.release()
        assert second.acquire() is True
        second.release()

    def test_different_roles_do_not_conflict(self, tmp_path: Path) -> None:
        mcp = ProjectLock(tmp_path, role="mcp")
        indexer = ProjectLock(tmp_path, role="index")
        assert mcp.acquire() is True
        assert indexer.acquire() is True
        mcp.release()
        indexer.release()

    def test_different_projects_do_not_conflict(self, tmp_path: Path) -> None:
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        a = ProjectLock(tmp_path / "a", role="mcp")
        b = ProjectLock(tmp_path / "b", role="mcp")
        assert a.acquire() is True
        assert b.acquire() is True
        a.release()
        b.release()

    def test_symlinked_path_shares_lock(self, tmp_path: Path) -> None:
        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real)
        assert canonical_project_path(link) == canonical_project_path(real)
        a = ProjectLock(real, role="mcp")
        b = ProjectLock(link, role="mcp")
        assert a.acquire() is True
        assert b.acquire() is False
        a.release()

    def test_stale_lock_file_from_dead_process_is_reusable(self, tmp_path: Path) -> None:
        """A leftover lock file without a live flock must not block acquisition."""
        lock_path = tmp_path / ".context" / "locks" / "mcp.lock"
        lock_path.parent.mkdir(parents=True)
        # Simulate a crashed process: metadata file exists but no flock is held
        # (the kernel released it when the process died).
        lock_path.write_text(json.dumps({"pid": 99999999, "role": "mcp"}))
        lock = ProjectLock(tmp_path, role="mcp")
        assert lock.acquire() is True
        lock.release()

    def test_lock_survives_across_processes(self, tmp_path: Path) -> None:
        """A child process holding the lock blocks the parent."""
        child = subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import sys, time; "
                    "sys.path.insert(0, sys.argv[2]); "
                    "from local_context_engine.core.instance import ProjectLock; "
                    "lock = ProjectLock(sys.argv[1], role='mcp'); "
                    "assert lock.acquire(); print('locked', flush=True); "
                    "time.sleep(30)"
                ),
                str(tmp_path),
                str(Path(__file__).resolve().parents[2] / "src"),
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            assert child.stdout is not None
            assert child.stdout.readline().strip() == "locked"
            lock = ProjectLock(tmp_path, role="mcp")
            assert lock.acquire() is False
        finally:
            child.terminate()
            child.wait(timeout=10)
        # After the child dies, the kernel released the flock: no stale lock.
        deadline = time.monotonic() + 5
        acquired = False
        while time.monotonic() < deadline:
            if lock.acquire():
                acquired = True
                break
            time.sleep(0.1)
        assert acquired
        lock.release()

    def test_context_manager_raises_on_conflict(self, tmp_path: Path) -> None:
        holder = ProjectLock(tmp_path, role="mcp")
        assert holder.acquire()
        with pytest.raises(ProjectLockError):
            with ProjectLock(tmp_path, role="mcp"):
                pass
        holder.release()


# ─────────────────────────────────────────────────────────────
# InstanceRegistry
# ─────────────────────────────────────────────────────────────


class TestInstanceRegistry:
    def test_register_list_unregister(self, tmp_path: Path) -> None:
        registry = InstanceRegistry(base_dir=tmp_path)
        registry.register("mcp", tmp_path / "proj")
        instances = registry.list_instances()
        assert len(instances) == 1
        assert instances[0]["pid"] == os.getpid()
        assert instances[0]["role"] == "mcp"
        assert instances[0]["rss_mb"] is None or instances[0]["rss_mb"] > 0
        registry.unregister()
        assert registry.list_instances() == []

    def test_stale_entries_are_swept(self, tmp_path: Path) -> None:
        registry = InstanceRegistry(base_dir=tmp_path)
        entries_dir = tmp_path / "instances"
        entries_dir.mkdir(parents=True)
        (entries_dir / "99999999.json").write_text(
            json.dumps({"pid": 99999999, "role": "mcp", "project": "/x"})
        )
        (entries_dir / "corrupt.json").write_text("{not json")
        assert registry.list_instances(sweep=True) == []
        assert list(entries_dir.glob("*.json")) == []

    def test_pid_reuse_detected_via_create_time(self, tmp_path: Path) -> None:
        registry = InstanceRegistry(base_dir=tmp_path)
        entries_dir = tmp_path / "instances"
        entries_dir.mkdir(parents=True)
        # Current pid but a create_time from long ago → treated as stale.
        (entries_dir / f"{os.getpid()}.json").write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "role": "mcp",
                    "project": "/x",
                    "create_time": 12345.0,
                }
            )
        )
        assert registry.list_instances(sweep=True) == []


# ─────────────────────────────────────────────────────────────
# MemoryMonitor
# ─────────────────────────────────────────────────────────────


class TestMemoryMonitor:
    def test_thresholds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monitor = MemoryMonitor(soft_limit_mb=100, hard_limit_mb=200)

        monkeypatch.setattr(instance_mod, "current_rss_mb", lambda: 50.0)
        assert monitor.check() is MemoryPressure.NORMAL

        monkeypatch.setattr(instance_mod, "current_rss_mb", lambda: 150.0)
        assert monitor.check() is MemoryPressure.SOFT

        monkeypatch.setattr(instance_mod, "current_rss_mb", lambda: 250.0)
        assert monitor.check() is MemoryPressure.HARD

    def test_invalid_limits_rejected(self) -> None:
        with pytest.raises(ValueError, match="hard limit must exceed soft limit"):
            MemoryMonitor(soft_limit_mb=200, hard_limit_mb=100)

    def test_current_rss_positive(self) -> None:
        assert instance_mod.current_rss_mb() > 0


# ─────────────────────────────────────────────────────────────
# Shutdown cleanup
# ─────────────────────────────────────────────────────────────


class TestShutdownCleanup:
    def test_cleanup_runs_once(self) -> None:
        calls: list[int] = []
        instance_mod._cleanup_callbacks.append(lambda: calls.append(1))
        instance_mod._run_cleanups()
        instance_mod._run_cleanups()
        assert calls == [1]

    def test_sigterm_cleans_registry_and_lock(self, tmp_path: Path) -> None:
        """A SIGTERM'd process removes its registry entry and lock."""
        src = Path(__file__).resolve().parents[2] / "src"
        (tmp_path / "proj").mkdir()
        child = subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import sys, time; "
                    "sys.path.insert(0, sys.argv[3]); "
                    "from local_context_engine.core.instance import ("
                    "    InstanceRegistry, ProjectLock, install_shutdown_handler); "
                    "from pathlib import Path; "
                    "lock = ProjectLock(sys.argv[1], role='mcp'); "
                    "assert lock.acquire(); "
                    "reg = InstanceRegistry(base_dir=Path(sys.argv[2])); "
                    "reg.register('mcp', sys.argv[1]); "
                    "install_shutdown_handler(lock.release); "
                    "install_shutdown_handler(reg.unregister); "
                    "print('ready', flush=True); "
                    "time.sleep(30)"
                ),
                str(tmp_path / "proj"),
                str(tmp_path / "state"),
                str(src),
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            assert child.stdout is not None
            assert child.stdout.readline().strip() == "ready"
            child.terminate()  # SIGTERM
            child.wait(timeout=10)
        finally:
            if child.poll() is None:
                child.kill()

        registry = InstanceRegistry(base_dir=tmp_path / "state")
        assert registry.list_instances(sweep=False) == []
        lock_file = tmp_path / "proj" / ".context" / "locks" / "mcp.lock"
        assert not lock_file.exists()
