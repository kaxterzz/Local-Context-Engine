"""
Repository file scanner with incremental indexing support.

Responsibilities:
  - Recursive directory traversal
  - Language detection from file extension
  - Gitignore + blacklist filtering
  - File hash computation (MD5 / SHA256)
  - Change detection (new / modified / deleted / unchanged)
  - Parallel scanning for large repositories

The scanner emits :class:`~local_context_engine.core.types.FileRecord` objects
that feed into the parsing pipeline.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from local_context_engine.core.exceptions import RepositoryNotFoundError
from local_context_engine.core.types import FileRecord, IndexingStatus, Language
from local_context_engine.indexer.scanner.gitignore_handler import GitignoreHandler
from local_context_engine.security.file_blacklist import FileBlacklist

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Language detection
# ─────────────────────────────────────────────────────────────

_EXTENSION_MAP: dict[str, Language] = {
    # PHP
    ".php": Language.PHP,
    # TypeScript / TSX
    ".ts": Language.TYPESCRIPT,
    ".tsx": Language.TSX,
    # JavaScript / JSX
    ".js": Language.JAVASCRIPT,
    ".jsx": Language.JSX,
    ".mjs": Language.JAVASCRIPT,
    ".cjs": Language.JAVASCRIPT,
    # Python
    ".py": Language.PYTHON,
    ".pyi": Language.PYTHON,
    # CSS / SCSS / LESS
    ".css": Language.CSS,
    ".scss": Language.CSS,
    ".less": Language.CSS,
    # HTML / Blade
    ".html": Language.HTML,
    ".htm": Language.HTML,
    ".blade.php": Language.BLADE,  # Matched before .php
    # Data formats
    ".json": Language.JSON,
    ".yaml": Language.YAML,
    ".yml": Language.YAML,
    # SQL
    ".sql": Language.SQL,
    # Documentation
    ".md": Language.MARKDOWN,
    ".mdx": Language.MARKDOWN,
    # Shell
    ".sh": Language.SHELL,
    ".bash": Language.SHELL,
    ".zsh": Language.SHELL,
}

# Framework detection heuristics based on path patterns
_FRAMEWORK_PATTERNS: list[tuple[str, str]] = [
    # Laravel
    ("app/Http/Controllers", "laravel"),
    ("app/Models", "laravel"),
    ("routes/web.php", "laravel"),
    ("artisan", "laravel"),
    # React / Next.js
    ("pages/_app.tsx", "nextjs"),
    ("pages/_app.jsx", "nextjs"),
    ("next.config", "nextjs"),
    # React (generic)
    ("src/App.tsx", "react"),
    ("src/App.jsx", "react"),
    ("vite.config", "react"),
]


def detect_language(path: Path) -> Language:
    """Detect the programming language from file extension."""
    name = path.name.lower()

    # Blade templates must be checked before .php
    if name.endswith(".blade.php"):
        return Language.BLADE

    suffix = path.suffix.lower()
    return _EXTENSION_MAP.get(suffix, Language.UNKNOWN)


def detect_framework(repo_root: Path) -> str | None:
    """
    Heuristically detect the primary framework used in a repository.

    Returns the framework name string or ``None`` if undetected.
    """
    for pattern, framework in _FRAMEWORK_PATTERNS:
        if (repo_root / pattern).exists():
            return framework
    return None


def _compute_hash(path: Path, algorithm: str = "md5") -> str:
    """Compute the hash of a file's contents."""
    h = hashlib.new(algorithm)
    try:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()


def _count_lines(path: Path) -> int:
    """Fast line count that avoids loading the entire file into memory."""
    try:
        count = 0
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                count += chunk.count(b"\n")
        return count
    except OSError:
        return 0


def _scan_single_file(
    file_path: Path,
    repo_root: Path,
    blacklist: FileBlacklist,
    gitignore: GitignoreHandler,
    hash_algorithm: str,
    existing_hashes: dict[str, str],
) -> FileRecord | None:
    """
    Evaluate a single file and return a :class:`FileRecord` or ``None``
    if the file should be skipped.

    This runs in a thread pool; must not use asyncio primitives.
    """
    try:
        rel_path = file_path.relative_to(repo_root)
    except ValueError:
        return None

    rel_str = rel_path.as_posix()

    # Security: blacklist check first
    if not blacklist.is_allowed(file_path, relative_to=repo_root):
        logger.debug("Blacklisted: %s", rel_str)
        return None

    # Gitignore check
    if gitignore.is_ignored(file_path):
        logger.debug("Gitignored: %s", rel_str)
        return None

    # Must be a regular file
    if not file_path.is_file():
        return None

    language = detect_language(file_path)
    if language == Language.UNKNOWN:
        # Index anyway for full-text search, but skip structured parsing
        pass

    stat = file_path.stat()
    file_hash = _compute_hash(file_path, hash_algorithm)
    line_count = _count_lines(file_path)

    # Determine status via hash comparison
    existing_hash = existing_hashes.get(rel_str)
    if existing_hash is None:
        status = IndexingStatus.NEW
    elif existing_hash == file_hash:
        status = IndexingStatus.UNCHANGED
    else:
        status = IndexingStatus.MODIFIED

    return FileRecord(
        id=str(uuid.uuid5(uuid.NAMESPACE_URL, rel_str)),
        path=rel_str,
        absolute_path=str(file_path),
        language=language,
        size_bytes=stat.st_size,
        hash=file_hash,
        modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
        line_count=line_count,
        status=status,
    )


class FileScanner:
    """
    Scans a repository directory tree and returns :class:`FileRecord` objects.

    Supports:
    - Gitignore-aware filtering
    - Security blacklist enforcement
    - Incremental scanning (comparing against known hashes)
    - Parallel scanning via thread pool
    """

    def __init__(
        self,
        blacklist: FileBlacklist,
        hash_algorithm: str = "md5",
        max_workers: int = 4,
    ) -> None:
        self._blacklist = blacklist
        self._hash_algorithm = hash_algorithm
        self._max_workers = max_workers

    async def scan_repository(
        self,
        repo_root: Path,
        existing_hashes: dict[str, str] | None = None,
        *,
        incremental: bool = True,
    ) -> list[FileRecord]:
        """
        Scan ``repo_root`` and return a list of :class:`FileRecord` objects.

        Args:
            repo_root:       Root directory of the repository.
            existing_hashes: Dict of ``{relative_path: hash}`` from the last
                             scan, used for incremental change detection.
            incremental:     If ``False``, treat every file as ``NEW``
                             regardless of existing hashes.

        Returns:
            All discovered file records, including unchanged files.
            Callers should filter by ``status`` as needed.
        """
        if not repo_root.exists():
            raise RepositoryNotFoundError(str(repo_root))

        hashes = existing_hashes if (incremental and existing_hashes) else {}

        gitignore = GitignoreHandler(repo_root)
        framework = detect_framework(repo_root)

        # Collect all candidate paths first (fast)
        all_paths = [p for p in repo_root.rglob("*") if p.is_file()]
        logger.info("Discovered %d candidate files in %s", len(all_paths), repo_root)

        loop = asyncio.get_event_loop()
        records: list[FileRecord | None] = []

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = [
                loop.run_in_executor(
                    executor,
                    _scan_single_file,
                    path,
                    repo_root,
                    self._blacklist,
                    gitignore,
                    self._hash_algorithm,
                    hashes,
                )
                for path in all_paths
            ]
            results = await asyncio.gather(*futures, return_exceptions=True)

        skipped = 0
        for result in results:
            if isinstance(result, Exception):
                logger.warning("Scan error: %s", result)
                skipped += 1
            elif result is None:
                skipped += 1
            else:
                if framework and result.framework is None:
                    result.framework = framework
                records.append(result)

        valid_records: list[FileRecord] = [r for r in records if r is not None]  # type: ignore[misc]

        new_count = sum(1 for r in valid_records if r.status == IndexingStatus.NEW)
        modified_count = sum(1 for r in valid_records if r.status == IndexingStatus.MODIFIED)
        unchanged_count = sum(1 for r in valid_records if r.status == IndexingStatus.UNCHANGED)

        logger.info(
            "Scan complete: %d total | %d new | %d modified | %d unchanged | %d skipped",
            len(valid_records),
            new_count,
            modified_count,
            unchanged_count,
            skipped,
        )

        return valid_records

    async def detect_deleted(
        self,
        repo_root: Path,
        known_paths: set[str],
    ) -> list[str]:
        """
        Return relative paths that existed in the index but no longer exist on disk.

        Args:
            repo_root:   Repository root.
            known_paths: Set of relative paths from the metadata store.

        Returns:
            List of relative paths that should be marked as deleted.
        """
        deleted: list[str] = []
        for rel_path in known_paths:
            abs_path = repo_root / rel_path
            if not abs_path.exists():
                deleted.append(rel_path)
        return deleted

    @classmethod
    def from_config(cls, config: "EngineConfig") -> "FileScanner":  # type: ignore[name-defined]
        """Create a :class:`FileScanner` from the engine config."""
        from local_context_engine.core.config import EngineConfig  # noqa: PLC0415
        from local_context_engine.security.file_blacklist import FileBlacklist  # noqa: PLC0415

        assert isinstance(config, EngineConfig)
        blacklist = FileBlacklist.from_config(config.security)
        return cls(
            blacklist=blacklist,
            hash_algorithm=config.performance.hash_algorithm,
            max_workers=config.performance.scan_workers,
        )
