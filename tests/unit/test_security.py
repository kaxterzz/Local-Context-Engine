"""Unit tests for the security layer."""

from __future__ import annotations

from pathlib import Path

import pytest

from local_context_engine.core.config import SecurityConfig
from local_context_engine.security.file_blacklist import FileBlacklist
from local_context_engine.security.pii_masker import PIIMasker


class TestFileBlacklist:
    """Tests for FileBlacklist."""

    def test_blocks_env_files(self, tmp_path: Path) -> None:
        bl = FileBlacklist()
        env_file = tmp_path / ".env"
        env_file.touch()
        assert not bl.is_allowed(env_file, relative_to=tmp_path)

    def test_blocks_pem_files(self, tmp_path: Path) -> None:
        bl = FileBlacklist()
        pem = tmp_path / "server.pem"
        pem.touch()
        assert not bl.is_allowed(pem, relative_to=tmp_path)

    def test_blocks_vendor_directory(self, tmp_path: Path) -> None:
        bl = FileBlacklist()
        vendor_file = tmp_path / "vendor" / "autoload.php"
        vendor_file.parent.mkdir()
        vendor_file.touch()
        assert not bl.is_allowed(vendor_file, relative_to=tmp_path)

    def test_blocks_node_modules(self, tmp_path: Path) -> None:
        bl = FileBlacklist()
        node_file = tmp_path / "node_modules" / "react" / "index.js"
        node_file.parent.mkdir(parents=True)
        node_file.touch()
        assert not bl.is_allowed(node_file, relative_to=tmp_path)

    def test_allows_php_source(self, tmp_path: Path) -> None:
        bl = FileBlacklist()
        php_file = tmp_path / "app" / "Controllers" / "UserController.php"
        php_file.parent.mkdir(parents=True)
        php_file.touch()
        assert bl.is_allowed(php_file, relative_to=tmp_path)

    def test_allows_typescript_source(self, tmp_path: Path) -> None:
        bl = FileBlacklist()
        ts_file = tmp_path / "src" / "components" / "UserList.tsx"
        ts_file.parent.mkdir(parents=True)
        ts_file.touch()
        assert bl.is_allowed(ts_file, relative_to=tmp_path)

    def test_blocks_binary_extensions(self, tmp_path: Path) -> None:
        bl = FileBlacklist()
        for ext in [".exe", ".dll", ".jpg", ".mp4", ".pdf"]:
            binary = tmp_path / f"file{ext}"
            binary.touch()
            assert not bl.is_allowed(binary, relative_to=tmp_path), f"Should block {ext}"

    def test_blocks_oversized_files(self, tmp_path: Path) -> None:
        bl = FileBlacklist(max_file_size_bytes=100)
        large_file = tmp_path / "large.php"
        large_file.write_bytes(b"x" * 200)
        assert not bl.is_allowed(large_file, relative_to=tmp_path)

    def test_filter_paths(self, tmp_path: Path) -> None:
        bl = FileBlacklist()
        paths = [
            tmp_path / "app" / "User.php",
            tmp_path / "vendor" / "laravel" / "framework" / "Facade.php",
            tmp_path / ".env",
        ]
        for p in paths:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()

        allowed = bl.filter_paths(paths, relative_to=tmp_path)
        assert len(allowed) == 1
        assert allowed[0].name == "User.php"

    def test_user_patterns(self, tmp_path: Path) -> None:
        bl = FileBlacklist(user_patterns=["**/secrets/**"])
        secret = tmp_path / "secrets" / "db_password.txt"
        secret.parent.mkdir()
        secret.touch()
        assert not bl.is_allowed(secret, relative_to=tmp_path)


class TestPIIMasker:
    """Tests for the PII masker."""

    def test_masks_email(self) -> None:
        masker = PIIMasker(mask_email=True)
        text = "Contact admin at admin@example.com for help."
        masked, stats = masker.mask(text)
        assert "admin@example.com" not in masked
        assert "[REDACTED_EMAIL]" in masked
        assert stats.emails == 1

    def test_masks_jwt(self) -> None:
        masker = PIIMasker(mask_jwt=True)
        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        text = f"Authorization: Bearer {jwt}"
        masked, stats = masker.mask(text)
        assert jwt not in masked
        assert stats.jwts == 1

    def test_masks_api_key(self) -> None:
        masker = PIIMasker(mask_api_keys=True)
        text = "api_key = 'sk-abcdefghijklmnopqrstuvwxyz1234567890ABCDE'"
        masked, stats = masker.mask(text)
        assert stats.api_keys >= 1

    def test_preserves_code_structure(self) -> None:
        masker = PIIMasker(mask_email=True)
        text = "public function handle() {\n    return $this->service->run();\n}"
        masked, stats = masker.mask(text)
        assert "handle" in masked
        assert "return" in masked
        assert stats.total == 0

    def test_no_masking_when_disabled(self) -> None:
        masker = PIIMasker(mask_email=False, mask_jwt=False)
        email = "user@test.com"
        masked, stats = masker.mask(email)
        assert email in masked
        assert stats.emails == 0

    def test_contains_pii(self) -> None:
        masker = PIIMasker()
        assert masker.contains_pii("email@example.com")
        assert not masker.contains_pii("public function store()")


class TestExtraNeverIndexPatterns:
    """
    Project configs must be able to ADD exclusions without discarding the
    defaults. YAML lists merge by replacement, so a project-level
    ``never_index_patterns`` silently re-enables bin/, obj/, packages/ and
    every designer file — ``extra_never_index_patterns`` avoids that trap.
    """

    def test_extra_patterns_are_appended_not_replacing(self, tmp_path: Path) -> None:
        from local_context_engine.core.config import SecurityConfig

        config = SecurityConfig(extra_never_index_patterns=["**/MyVendorLib/**"])
        bl = FileBlacklist.from_config(config)

        # The extra pattern applies …
        assert not bl.is_allowed(
            tmp_path / "src" / "MyVendorLib" / "a.js", relative_to=tmp_path
        )
        # … and the defaults still apply.
        assert not bl.is_allowed(
            tmp_path / "obj" / "Debug" / "x.cs", relative_to=tmp_path
        )
        assert not bl.is_allowed(
            tmp_path / "Views" / "Home.designer.cs", relative_to=tmp_path
        )
        # Real source is untouched.
        assert bl.is_allowed(
            tmp_path / "Controllers" / "HomeController.cs", relative_to=tmp_path
        )

    def test_dotnet_build_output_excluded_by_default(self, tmp_path: Path) -> None:
        """bin/ holds referenced DLLs in classic ASP.NET, not just bin/Debug/."""
        bl = FileBlacklist.from_config(SecurityConfig())
        for rel in [
            "MIS.UI/bin/Telerik.Web.UI.xml",
            "MIS.REPORT/bin/Generated.cs",
            "MIS.DATA/obj/Debug/x.cs",
            "packages/EntityFramework.6.4.4/lib/EntityFramework.xml",
            ".vs/config/applicationhost.config",
            "Pages/Default.aspx.designer.cs",
            "Properties/AssemblyInfo.cs",
            "Scripts/jquery-1.10.2.min.js",
        ]:
            assert not bl.is_allowed(tmp_path / rel, relative_to=tmp_path), rel

        assert bl.is_allowed(tmp_path / "Pages/Default.aspx.cs", relative_to=tmp_path)
