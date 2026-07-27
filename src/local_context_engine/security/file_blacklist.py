"""
File blacklist enforcement.

Uses ``pathspec`` (the same gitignore-matching library) to evaluate
whether a candidate file path matches any blacklisted pattern.

All checks happen BEFORE a file is opened or read.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pathspec

from local_context_engine.core.exceptions import BlacklistedFileError

logger = logging.getLogger(__name__)


# Patterns that are ALWAYS blocked, regardless of user configuration.
# These cannot be overridden.
_IMMUTABLE_BLOCKLIST: list[str] = [
    # Secrets / credentials
    "**/.env",
    "**/.env.*",
    "**/*.key",
    "**/*.pem",
    "**/*.crt",
    "**/*.cert",
    "**/*.cer",
    "**/*.p12",
    "**/*.pfx",
    "**/*.p8",
    "**/id_rsa",
    "**/id_rsa.pub",
    "**/id_ecdsa",
    "**/id_ecdsa.pub",
    "**/id_ed25519",
    "**/id_ed25519.pub",
    "**/.ssh/**",
    "**/secrets/**",
    "**/credentials",
    "**/credentials.json",
    "**/service-account*.json",
    "**/serviceaccount*.json",
    # Session / cache / upload data
    "**/storage/logs/**",
    "**/storage/sessions/**",
    "**/storage/framework/cache/**",
    "**/storage/framework/sessions/**",
    "**/storage/uploads/**",
    "**/public/storage/**",
    # VCS
    "**/.git/**",
    # Dependency trees — huge, generated, and a memory hazard to index.
    "**/vendor/**",
    "**/node_modules/**",
    "**/__pycache__/**",
    "**/.venv/**",
    "**/venv/**",
    # The engine's own index artifacts. Without this, the vector index,
    # metadata DB WAL, and graph pickle inside <repo>/.context/ get indexed
    # and change on every run — a self-indexing feedback loop that re-embeds
    # forever and bloats the index.
    "**/.context/**",
    # SQLite sidecar files (change constantly, pure binary churn)
    "**/*.db-wal",
    "**/*.db-shm",
    "**/*.sqlite-wal",
    "**/*.sqlite-shm",
    "**/*-journal",
]

# Extensions that indicate binary or non-text files
_BINARY_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".exe", ".dll", ".so", ".dylib", ".bin", ".obj", ".o", ".a", ".lib",
        ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
        ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".ico", ".webp", ".tiff", ".avif",
        ".mp4", ".avi", ".mov", ".mkv", ".mp3", ".wav", ".ogg", ".flac",
        ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".odt",
        ".woff", ".woff2", ".ttf", ".otf", ".eot",
        ".db", ".sqlite", ".sqlite3", ".mdb",
        ".db-wal", ".db-shm", ".sqlite-wal", ".sqlite-shm",
        ".pkl", ".pickle", ".index", ".faiss",
        ".pyc", ".pyd", ".pyo",
        ".class", ".jar",
        ".dmg", ".iso", ".img",
    }
)


class FileBlacklist:
    """
    Evaluates whether a file path is permitted for indexing.

    Combines:
    - Immutable built-in patterns (always active, cannot be disabled)
    - User-configured patterns from :class:`~local_context_engine.core.config.SecurityConfig`
    - Binary extension filtering
    - Maximum file size enforcement
    """

    def __init__(
        self,
        user_patterns: list[str] | None = None,
        max_file_size_bytes: int = 2 * 1024 * 1024,
        extra_binary_extensions: list[str] | None = None,
    ) -> None:
        """
        Args:
            user_patterns:             Additional glob patterns from the project config.
            max_file_size_bytes:       Files larger than this are skipped.
            extra_binary_extensions:   Additional binary extensions to reject.
        """
        all_patterns = list(_IMMUTABLE_BLOCKLIST)
        if user_patterns:
            all_patterns.extend(user_patterns)

        self._spec = pathspec.PathSpec.from_lines("gitwildmatch", all_patterns)
        self._max_size = max_file_size_bytes
        self._binary_exts = _BINARY_EXTENSIONS | frozenset(extra_binary_extensions or [])

        logger.debug(
            "FileBlacklist initialised with %d patterns, max_size=%d bytes",
            len(all_patterns),
            max_file_size_bytes,
        )

    # ── Public API ────────────────────────────────────────────

    def is_allowed(self, path: Path, relative_to: Path | None = None) -> bool:
        """
        Return ``True`` if the file is safe to index, ``False`` otherwise.

        Does NOT raise; use :meth:`assert_allowed` if you want an exception.

        Args:
            path:        Absolute or relative file path.
            relative_to: If given, normalise ``path`` relative to this root
                         before matching. Should be the repository root.
        """
        check_path = (
            path.relative_to(relative_to) if relative_to and path.is_absolute() else path
        )
        str_path = check_path.as_posix()

        # 1. Extension check (fast path)
        if path.suffix.lower() in self._binary_exts:
            logger.debug("Skipping binary file: %s", str_path)
            return False

        # 2. Pattern check
        if self._spec.match_file(str_path):
            logger.debug("Blacklisted by pattern: %s", str_path)
            return False

        # 3. Size check (only if file exists)
        if path.exists() and path.is_file():
            size = path.stat().st_size
            if size > self._max_size:
                logger.debug("Skipping oversized file (%d bytes): %s", size, str_path)
                return False

        return True

    def assert_allowed(self, path: Path, relative_to: Path | None = None) -> None:
        """
        Assert that the file is allowed. Raise :exc:`BlacklistedFileError` if not.

        Args:
            path:        Absolute or relative file path.
            relative_to: Repository root for relative path normalisation.
        """
        check_path = (
            path.relative_to(relative_to) if relative_to and path.is_absolute() else path
        )
        str_path = check_path.as_posix()

        if path.suffix.lower() in self._binary_exts:
            raise BlacklistedFileError(str_path, f"binary extension '{path.suffix}'")

        for pattern in _IMMUTABLE_BLOCKLIST:
            spec = pathspec.PathSpec.from_lines("gitwildmatch", [pattern])
            if spec.match_file(str_path):
                raise BlacklistedFileError(str_path, pattern)

        if self._spec.match_file(str_path):
            raise BlacklistedFileError(str_path, "<user-defined pattern>")

    def filter_paths(
        self, paths: list[Path], relative_to: Path | None = None
    ) -> list[Path]:
        """Return only the paths that pass all blacklist checks."""
        return [p for p in paths if self.is_allowed(p, relative_to)]

    def rejected_count(self, paths: list[Path], relative_to: Path | None = None) -> int:
        """Count how many paths would be rejected."""
        return sum(1 for p in paths if not self.is_allowed(p, relative_to))

    @classmethod
    def from_config(cls, config: "SecurityConfig") -> "FileBlacklist":  # type: ignore[name-defined]
        """Create a :class:`FileBlacklist` from a :class:`SecurityConfig` instance."""
        from local_context_engine.core.config import SecurityConfig  # noqa: PLC0415

        assert isinstance(config, SecurityConfig)
        return cls(
            user_patterns=config.never_index_patterns,
            max_file_size_bytes=config.max_file_size_bytes,
            extra_binary_extensions=config.binary_extensions,
        )
