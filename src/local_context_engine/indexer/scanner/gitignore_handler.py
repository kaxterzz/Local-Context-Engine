"""
Gitignore-aware path filter.

Loads ``.gitignore`` files from the repository root (and any sub-directories)
and evaluates whether a given path should be excluded from scanning.

Uses ``pathspec`` which implements the same gitignore matching algorithm used
by Git itself, including negation patterns (``!important.log``).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pathspec

logger = logging.getLogger(__name__)


class GitignoreHandler:
    """
    Loads and evaluates ``.gitignore`` rules for a repository.

    Multiple ``.gitignore`` files are supported (root + subdirectories).
    Each rule is evaluated relative to the directory containing the
    ``.gitignore`` file it came from, mirroring Git's behaviour.
    """

    def __init__(self, repo_root: Path) -> None:
        self._repo_root = repo_root
        self._specs: list[tuple[Path, pathspec.PathSpec]] = []
        self._load()

    def _load(self) -> None:
        """Walk the repository and load all ``.gitignore`` files."""
        gitignore_files_found = 0

        for gitignore_path in self._repo_root.rglob(".gitignore"):
            # Skip .gitignore files inside .git
            try:
                rel = gitignore_path.relative_to(self._repo_root)
            except ValueError:
                continue
            if ".git" in rel.parts:
                continue

            try:
                content = gitignore_path.read_text(encoding="utf-8", errors="ignore")
                spec = pathspec.PathSpec.from_lines("gitwildmatch", content.splitlines())
                self._specs.append((gitignore_path.parent, spec))
                gitignore_files_found += 1
            except OSError as exc:
                logger.warning("Failed to read %s: %s", gitignore_path, exc)

        logger.debug(
            "Loaded %d .gitignore file(s) from %s",
            gitignore_files_found,
            self._repo_root,
        )

    def is_ignored(self, path: Path) -> bool:
        """
        Return ``True`` if ``path`` is excluded by any loaded ``.gitignore``.

        Args:
            path: Absolute path to the file or directory.
        """
        for gitignore_dir, spec in self._specs:
            try:
                rel = path.relative_to(gitignore_dir)
            except ValueError:
                continue
            if spec.match_file(rel.as_posix()):
                return True
        return False

    def filter_paths(self, paths: list[Path]) -> list[Path]:
        """Return only paths not ignored by any ``.gitignore``."""
        return [p for p in paths if not self.is_ignored(p)]

    def reload(self) -> None:
        """Re-scan the repository for updated ``.gitignore`` files."""
        self._specs.clear()
        self._load()
