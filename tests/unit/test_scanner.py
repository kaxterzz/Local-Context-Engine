"""Unit tests for the file scanner."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

import pytest

from local_context_engine.core.types import IndexingStatus, Language
from local_context_engine.indexer.scanner.file_scanner import FileScanner, detect_language
from local_context_engine.indexer.scanner.gitignore_handler import GitignoreHandler
from local_context_engine.security.file_blacklist import FileBlacklist


class TestLanguageDetection:
    """Tests for the language detection function."""

    def test_php_detection(self) -> None:
        assert detect_language(Path("Controller.php")) == Language.PHP

    def test_typescript_detection(self) -> None:
        assert detect_language(Path("component.ts")) == Language.TYPESCRIPT

    def test_tsx_detection(self) -> None:
        assert detect_language(Path("App.tsx")) == Language.TSX

    def test_javascript_detection(self) -> None:
        assert detect_language(Path("script.js")) == Language.JAVASCRIPT

    def test_python_detection(self) -> None:
        assert detect_language(Path("service.py")) == Language.PYTHON

    @pytest.mark.parametrize("name", ["Program.cs", "HomeController.aspx.cs"])
    def test_csharp_detection(self, name: str) -> None:
        assert detect_language(Path(name)) == Language.CSHARP

    @pytest.mark.parametrize("name", ["App.sln", "Api.csproj", "Common.props", "Build.targets"])
    def test_dotnet_project_detection(self, name: str) -> None:
        assert detect_language(Path(name)) == Language.DOTNET

    @pytest.mark.parametrize("name", ["Index.cshtml", "App.razor", "Default.aspx", "Widget.ascx", "Site.master"])
    def test_aspnet_detection(self, name: str) -> None:
        assert detect_language(Path(name)) == Language.ASPNET

    def test_classic_asp_detection(self) -> None:
        assert detect_language(Path("default.asp")) == Language.ASP

    def test_sql_detection(self) -> None:
        assert detect_language(Path("schema.sql")) == Language.SQL

    def test_blade_detection(self) -> None:
        assert detect_language(Path("layout.blade.php")) == Language.BLADE

    def test_unknown_detection(self) -> None:
        assert detect_language(Path("Makefile")) == Language.UNKNOWN


class TestGitignoreHandler:
    """Tests for gitignore-based filtering."""

    def test_ignores_files_matching_gitignore(self, tmp_path: Path) -> None:
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("*.log\nbuild/\n")

        log_file = tmp_path / "error.log"
        log_file.touch()
        build_file = tmp_path / "build" / "output.js"
        build_file.parent.mkdir()
        build_file.touch()

        handler = GitignoreHandler(tmp_path)
        assert handler.is_ignored(log_file)
        assert handler.is_ignored(build_file)

    def test_allows_non_ignored_files(self, tmp_path: Path) -> None:
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("*.log\n")

        php_file = tmp_path / "Controller.php"
        php_file.touch()

        handler = GitignoreHandler(tmp_path)
        assert not handler.is_ignored(php_file)

    def test_handles_missing_gitignore(self, tmp_path: Path) -> None:
        handler = GitignoreHandler(tmp_path)
        php_file = tmp_path / "app.php"
        php_file.touch()
        assert not handler.is_ignored(php_file)


class TestFileScanner:
    """Tests for the repository file scanner."""

    @pytest.fixture
    def scanner(self) -> FileScanner:
        return FileScanner(
            blacklist=FileBlacklist(),
            hash_algorithm="md5",
            max_workers=2,
        )

    @pytest.fixture
    def sample_repo(self, tmp_path: Path) -> Path:
        """Create a minimal sample repository."""
        # PHP file
        php_dir = tmp_path / "app" / "Http" / "Controllers"
        php_dir.mkdir(parents=True)
        (php_dir / "UserController.php").write_text("<?php class UserController {}")

        # TypeScript file
        ts_dir = tmp_path / "src" / "components"
        ts_dir.mkdir(parents=True)
        (ts_dir / "UserList.tsx").write_text("export default function UserList() {}")

        # Python file
        (tmp_path / "service.py").write_text("class UserService: pass")

        # File to be ignored
        vendor_dir = tmp_path / "vendor"
        vendor_dir.mkdir()
        (vendor_dir / "autoload.php").write_text("<?php")

        # .env to be ignored
        (tmp_path / ".env").write_text("DB_PASSWORD=secret")

        return tmp_path

    @pytest.mark.asyncio
    async def test_scans_files(self, scanner: FileScanner, sample_repo: Path) -> None:
        records = await scanner.scan_repository(sample_repo)
        paths = {r.path for r in records}

        assert "app/Http/Controllers/UserController.php" in paths
        assert "src/components/UserList.tsx" in paths
        assert "service.py" in paths

    @pytest.mark.asyncio
    async def test_excludes_vendor(self, scanner: FileScanner, sample_repo: Path) -> None:
        records = await scanner.scan_repository(sample_repo)
        paths = {r.path for r in records}
        assert "vendor/autoload.php" not in paths

    @pytest.mark.asyncio
    async def test_excludes_env(self, scanner: FileScanner, sample_repo: Path) -> None:
        records = await scanner.scan_repository(sample_repo)
        paths = {r.path for r in records}
        assert ".env" not in paths

    @pytest.mark.asyncio
    async def test_detects_correct_languages(
        self, scanner: FileScanner, sample_repo: Path
    ) -> None:
        records = await scanner.scan_repository(sample_repo)
        by_path = {r.path: r for r in records}

        php_rec = by_path.get("app/Http/Controllers/UserController.php")
        assert php_rec is not None
        assert php_rec.language == Language.PHP

    @pytest.mark.asyncio
    async def test_incremental_detects_unchanged(
        self, scanner: FileScanner, sample_repo: Path
    ) -> None:
        # First scan
        records = await scanner.scan_repository(sample_repo, incremental=True)
        hashes = {r.path: r.hash for r in records}

        # Second scan with same hashes
        records2 = await scanner.scan_repository(
            sample_repo, existing_hashes=hashes, incremental=True
        )
        unchanged = [r for r in records2 if r.status == IndexingStatus.UNCHANGED]
        assert len(unchanged) == len(records2)

    @pytest.mark.asyncio
    async def test_incremental_detects_modified(
        self, scanner: FileScanner, sample_repo: Path
    ) -> None:
        # First scan
        records = await scanner.scan_repository(sample_repo)
        hashes = {r.path: r.hash for r in records}

        # Modify a file
        controller = sample_repo / "app" / "Http" / "Controllers" / "UserController.php"
        controller.write_text("<?php class UserController { public function index() {} }")

        # Second scan
        records2 = await scanner.scan_repository(
            sample_repo, existing_hashes=hashes, incremental=True
        )
        modified = [r for r in records2 if r.status == IndexingStatus.MODIFIED]
        assert len(modified) >= 1
