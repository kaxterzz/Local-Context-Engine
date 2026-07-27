"""
Process instance management: single-instance locking, a cross-process
registry, memory pressure monitoring, and clean shutdown handling.

Why this exists
---------------
Multiple ``context mcp`` / ``context index`` processes for the same project
each load a multi-GB working set (torch + embedding model + FAISS index +
BM25 corpus). Running them concurrently multiplied that footprint until the
machine exhausted RAM and swap. This module guarantees:

  - Only one process per (canonical project path, role) via an OS-level
    ``flock`` that the kernel releases automatically if the process dies —
    stale locks cannot outlive their owner.
  - A registry of live engine processes under the XDG state directory,
    powering ``context ps`` diagnostics.
  - Soft/hard RSS limits with graceful degradation instead of OOM freezes.
"""

from __future__ import annotations

import atexit
import enum
import json
import logging
import os
import signal
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import fcntl

    _HAS_FCNTL = True
except ImportError:  # pragma: no cover - Windows
    _HAS_FCNTL = False


# ─────────────────────────────────────────────────────────────
# Canonical paths
# ─────────────────────────────────────────────────────────────


def canonical_project_path(path: Path | str) -> Path:
    """Resolve symlinks/`..`/case so equivalent paths share one lock."""
    return Path(path).expanduser().resolve()


def _state_dir() -> Path:
    base = os.environ.get("XDG_STATE_HOME") or (Path.home() / ".local" / "state")
    return Path(base) / "local-context-engine"


# ─────────────────────────────────────────────────────────────
# ProjectLock
# ─────────────────────────────────────────────────────────────


class ProjectLockError(RuntimeError):
    """Raised when a project lock is held by another live process."""

    def __init__(self, project: Path, role: str, holder: dict[str, Any] | None) -> None:
        self.project = project
        self.role = role
        self.holder = holder or {}
        pid = self.holder.get("pid", "unknown")
        super().__init__(
            f"Another '{role}' process (pid {pid}) already operates on "
            f"{project}. Refusing to start a duplicate instance. "
            f"Run 'context ps' to inspect running instances, or set "
            f"CONTEXT_SINGLE_INSTANCE_PER_PROJECT=false to override."
        )


class ProjectLock:
    """
    Exclusive per-project, per-role advisory file lock.

    Uses ``flock`` so the kernel releases the lock if the holder crashes or
    is SIGKILLed — there is no stale-lock window. The lock file additionally
    stores holder metadata (pid, started_at) for diagnostics; the metadata
    may be stale but the flock itself never is.

    Usage::

        lock = ProjectLock(project_root, role="mcp")
        if not lock.acquire():
            raise ProjectLockError(project_root, "mcp", lock.holder_info())
        ...
        lock.release()
    """

    def __init__(self, project_root: Path | str, role: str = "mcp") -> None:
        self._project = canonical_project_path(project_root)
        self._role = role
        self._lock_dir = self._project / ".context" / "locks"
        self._lock_path = self._lock_dir / f"{role}.lock"
        self._fd: int | None = None

    @property
    def lock_path(self) -> Path:
        return self._lock_path

    def acquire(self) -> bool:
        """Try to acquire the lock. Returns ``False`` if held elsewhere."""
        if self._fd is not None:
            return True
        self._lock_dir.mkdir(parents=True, exist_ok=True)

        # Open + flock + verify-inode loop. The verification guards against
        # the classic lockfile race: we may have opened an inode that the
        # previous holder unlinked between our open() and flock(), in which
        # case our lock is on an orphaned file and a concurrent process could
        # hold a "valid" lock on the new inode. Retry on mismatch.
        for _ in range(5):
            fd = os.open(self._lock_path, os.O_RDWR | os.O_CREAT, 0o644)
            try:
                if _HAS_FCNTL:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                else:  # pragma: no cover - Windows fallback
                    import msvcrt

                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            except OSError:
                os.close(fd)
                return False

            try:
                if os.fstat(fd).st_ino == os.stat(self._lock_path).st_ino:
                    break  # locked the live inode
            except FileNotFoundError:
                pass  # path was unlinked under us — retry
            os.close(fd)
        else:
            return False

        payload = json.dumps(
            {
                "pid": os.getpid(),
                "role": self._role,
                "project": str(self._project),
                "started_at": time.time(),
                "argv": sys.argv,
            }
        ).encode()
        os.ftruncate(fd, 0)
        os.pwrite(fd, payload, 0)
        os.fsync(fd)
        self._fd = fd
        logger.debug("Acquired %s lock for %s", self._role, self._project)
        return True

    def holder_info(self) -> dict[str, Any] | None:
        """Best-effort metadata about the current/last lock holder."""
        try:
            return json.loads(self._lock_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def release(self) -> None:
        if self._fd is None:
            return
        # Unlink BEFORE closing, while the flock is still held. Doing it the
        # other way round races: another process could flock the old inode
        # right after close(), then the unlink would let a third process
        # create (and lock) a fresh inode — two live "holders" at once.
        # With unlink-first, every process that opens the path after this
        # point gets the same new inode, so mutual exclusion is preserved.
        try:
            self._lock_path.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            if _HAS_FCNTL:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
        except OSError:
            pass
        self._fd = None
        logger.debug("Released %s lock for %s", self._role, self._project)

    @property
    def held(self) -> bool:
        return self._fd is not None

    def __enter__(self) -> ProjectLock:
        if not self.acquire():
            raise ProjectLockError(self._project, self._role, self.holder_info())
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


# ─────────────────────────────────────────────────────────────
# InstanceRegistry
# ─────────────────────────────────────────────────────────────


class InstanceRegistry:
    """
    Cross-process registry of live engine instances.

    One JSON file per process under ``~/.local/state/local-context-engine/
    instances/``. Stale entries (dead pids, or pid reuse detected via the
    process creation time) are swept on every listing.
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        self._dir = (base_dir or _state_dir()) / "instances"

    def _entry_path(self, pid: int) -> Path:
        return self._dir / f"{pid}.json"

    def register(self, role: str, project: Path | str) -> Path:
        """Register the current process. Returns the entry path."""
        self._dir.mkdir(parents=True, exist_ok=True)
        pid = os.getpid()
        entry = {
            "pid": pid,
            "role": role,
            "project": str(canonical_project_path(project)),
            "started_at": time.time(),
            "create_time": _process_create_time(pid),
            "argv": sys.argv,
        }
        path = self._entry_path(pid)
        path.write_text(json.dumps(entry, indent=2), encoding="utf-8")
        return path

    def unregister(self, pid: int | None = None) -> None:
        try:
            self._entry_path(pid or os.getpid()).unlink(missing_ok=True)
        except OSError:
            pass

    def list_instances(self, sweep: bool = True) -> list[dict[str, Any]]:
        """Return live instances; optionally delete stale entries."""
        if not self._dir.exists():
            return []
        alive: list[dict[str, Any]] = []
        for entry_file in sorted(self._dir.glob("*.json")):
            try:
                entry = json.loads(entry_file.read_text(encoding="utf-8"))
                pid = int(entry["pid"])
            except (OSError, ValueError, KeyError):
                if sweep:
                    entry_file.unlink(missing_ok=True)
                continue

            if not _pid_alive(pid, entry.get("create_time")):
                if sweep:
                    entry_file.unlink(missing_ok=True)
                continue

            entry["rss_mb"] = _process_rss_mb(pid)
            alive.append(entry)
        return alive


def _pid_alive(pid: int, expected_create_time: float | None) -> bool:
    """Check pid liveness, guarding against pid reuse."""
    try:
        import psutil

        proc = psutil.Process(pid)
        if expected_create_time is not None:
            # Allow 1s slack: create_time precision differs between reads.
            return abs(proc.create_time() - expected_create_time) < 1.0
        return proc.is_running()
    except ImportError:  # pragma: no cover
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False
    except Exception:
        return False


def _process_create_time(pid: int) -> float | None:
    try:
        import psutil

        return psutil.Process(pid).create_time()
    except Exception:  # pragma: no cover
        return None


def _process_rss_mb(pid: int) -> float | None:
    try:
        import psutil

        return round(psutil.Process(pid).memory_info().rss / (1024 * 1024), 1)
    except Exception:  # pragma: no cover
        return None


# ─────────────────────────────────────────────────────────────
# Memory monitoring
# ─────────────────────────────────────────────────────────────


class MemoryPressure(enum.Enum):
    NORMAL = "normal"
    SOFT = "soft"
    HARD = "hard"


def current_rss_mb() -> float:
    """Current process RSS in MB (psutil, with a /proc fallback)."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:  # pragma: no cover
        try:
            with open("/proc/self/status", encoding="ascii") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024
        except OSError:
            pass
        return 0.0


class MemoryMonitor:
    """
    Classifies current RSS against soft/hard limits.

    - ``SOFT``: callers should shrink batches, flush caches, and warn.
    - ``HARD``: callers must stop taking on new work gracefully.
    """

    def __init__(self, soft_limit_mb: float, hard_limit_mb: float) -> None:
        if hard_limit_mb <= soft_limit_mb:
            raise ValueError("hard limit must exceed soft limit")
        self.soft_limit_mb = soft_limit_mb
        self.hard_limit_mb = hard_limit_mb

    def check(self) -> MemoryPressure:
        rss = current_rss_mb()
        if rss >= self.hard_limit_mb:
            return MemoryPressure.HARD
        if rss >= self.soft_limit_mb:
            return MemoryPressure.SOFT
        return MemoryPressure.NORMAL

    @property
    def rss_mb(self) -> float:
        return current_rss_mb()


# ─────────────────────────────────────────────────────────────
# Shutdown handling
# ─────────────────────────────────────────────────────────────

_cleanup_callbacks: list[Callable[[], None]] = []
_handlers_installed = False


def _run_cleanups() -> None:
    while _cleanup_callbacks:
        cb = _cleanup_callbacks.pop()
        try:
            cb()
        except Exception as exc:  # noqa: BLE001 — never fail during shutdown
            logger.debug("Cleanup callback failed: %s", exc)


def install_shutdown_handler(cleanup: Callable[[], None]) -> None:
    """
    Register *cleanup* to run on normal exit, SIGTERM, and SIGINT.

    Signal handlers chain to any previously installed handler so libraries
    (e.g. asyncio/typer KeyboardInterrupt behaviour) keep working. Callbacks
    run at most once.
    """
    global _handlers_installed
    _cleanup_callbacks.append(cleanup)
    if _handlers_installed:
        return
    _handlers_installed = True

    atexit.register(_run_cleanups)

    for signum in (signal.SIGTERM, signal.SIGINT):
        previous = signal.getsignal(signum)

        def _handler(sig: int, frame: object, _previous=previous) -> None:
            _run_cleanups()
            if callable(_previous):
                _previous(sig, frame)
            elif _previous == signal.SIG_DFL:
                signal.signal(sig, signal.SIG_DFL)
                os.kill(os.getpid(), sig)

        try:
            signal.signal(signum, _handler)
        except (ValueError, OSError):  # pragma: no cover — non-main thread
            pass


# ─────────────────────────────────────────────────────────────
# Structured logging helper
# ─────────────────────────────────────────────────────────────


def log_process_status(
    log: logging.Logger,
    *,
    project: Path | str,
    phase: str,
    **fields: Any,
) -> None:
    """
    Emit a single structured status line with pid/RSS plus custom fields.

    Example output::

        [status] pid=1234 project=/repo phase=embedding rss_mb=812.4 batch=20 ...
    """
    parts = [
        f"pid={os.getpid()}",
        f"project={project}",
        f"phase={phase}",
        f"rss_mb={current_rss_mb():.1f}",
    ]
    parts += [f"{k}={v}" for k, v in fields.items()]
    log.info("[status] %s", " ".join(parts))
