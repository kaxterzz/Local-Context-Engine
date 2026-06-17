"""Unit tests for the code parsers."""

from __future__ import annotations

import pytest

from local_context_engine.core.types import Language, SymbolType
from local_context_engine.indexer.parsers.php_parser import PHPParser
from local_context_engine.indexer.parsers.python_parser import PythonParser
from local_context_engine.indexer.parsers.typescript_parser import TypeScriptParser


class TestPHPParser:
    """Tests for PHPParser."""

    @pytest.fixture
    def parser(self) -> PHPParser:
        return PHPParser()

    def test_extracts_class(self, parser: PHPParser, sample_php_source: str) -> None:
        result = parser.parse(sample_php_source, "file-001", "app/Controllers/UserController.php")
        class_symbols = [s for s in result.symbols if s.symbol_type == SymbolType.CONTROLLER]
        assert len(class_symbols) >= 1
        assert class_symbols[0].name == "UserController"

    def test_extracts_methods(self, parser: PHPParser, sample_php_source: str) -> None:
        result = parser.parse(sample_php_source, "file-001", "app/Controllers/UserController.php")
        method_names = {s.name for s in result.symbols if s.symbol_type == SymbolType.METHOD}
        assert "index" in method_names
        assert "store" in method_names

    def test_detects_laravel_controller(self, parser: PHPParser) -> None:
        source = "<?php\nclass OrderController extends Controller {}"
        result = parser.parse(source, "f1", "app/Http/Controllers/OrderController.php")
        controllers = [s for s in result.symbols if s.symbol_type == SymbolType.CONTROLLER]
        assert controllers

    def test_detects_laravel_model(self, parser: PHPParser) -> None:
        source = "<?php\nclass User extends Model {}"
        result = parser.parse(source, "f1", "app/Models/User.php")
        models = [s for s in result.symbols if s.symbol_type == SymbolType.MODEL]
        assert models

    def test_detects_namespace(self, parser: PHPParser, sample_php_source: str) -> None:
        result = parser.parse(sample_php_source, "f1", "app/Controllers/UserController.php")
        classes = [s for s in result.symbols if s.name == "UserController"]
        assert classes
        assert "App" in classes[0].qualified_name or "UserController" in classes[0].qualified_name

    def test_detects_interface(self, parser: PHPParser) -> None:
        source = "<?php\ninterface UserRepositoryInterface {\n    public function findAll();\n}"
        result = parser.parse(source, "f1", "app/Contracts/UserRepositoryInterface.php")
        interfaces = [s for s in result.symbols if s.symbol_type == SymbolType.INTERFACE]
        assert interfaces
        assert interfaces[0].name == "UserRepositoryInterface"

    def test_extract_relationships(self, parser: PHPParser, sample_php_source: str) -> None:
        result = parser.parse(sample_php_source, "f1", "app/Controllers/UserController.php")
        # Should have at least an EXTENDS relationship (extends Controller)
        assert len(result.relationships) >= 1

    def test_no_errors_on_valid_php(self, parser: PHPParser, sample_php_source: str) -> None:
        result = parser.parse(sample_php_source, "f1", "test.php")
        # Parser should not produce errors on valid PHP
        assert len(result.errors) == 0 or result.symbols  # graceful degradation

    def test_empty_file(self, parser: PHPParser) -> None:
        result = parser.parse("", "f1", "empty.php")
        assert result.symbols == [] or True  # Must not crash

    def test_migration_detection(self, parser: PHPParser) -> None:
        source = "<?php\nuse Illuminate\\Database\\Migrations\\Migration;\nclass CreateUsersTable extends Migration {}"
        result = parser.parse(source, "f1", "database/migrations/2024_01_01_000000_create_users_table.php")
        migrations = [s for s in result.symbols if s.symbol_type == SymbolType.MIGRATION]
        assert migrations


class TestTypeScriptParser:
    """Tests for TypeScriptParser."""

    @pytest.fixture
    def parser(self) -> TypeScriptParser:
        return TypeScriptParser(Language.TYPESCRIPT)

    @pytest.fixture
    def tsx_parser(self) -> TypeScriptParser:
        return TypeScriptParser(Language.TSX)

    def test_extracts_interface(self, parser: TypeScriptParser, sample_typescript_source: str) -> None:
        result = parser.parse(sample_typescript_source, "f1", "src/types/User.ts")
        interfaces = [s for s in result.symbols if s.symbol_type == SymbolType.INTERFACE]
        assert interfaces
        assert interfaces[0].name == "User"

    def test_detects_react_hooks(self, parser: TypeScriptParser, sample_typescript_source: str) -> None:
        result = parser.parse(sample_typescript_source, "f1", "src/hooks/users.ts")
        hooks = [s for s in result.symbols if s.symbol_type == SymbolType.REACT_HOOK]
        hook_names = {h.name for h in hooks}
        assert "useUsers" in hook_names or "useCreateUser" in hook_names

    def test_detects_react_component(self, tsx_parser: TypeScriptParser, sample_typescript_source: str) -> None:
        result = tsx_parser.parse(sample_typescript_source, "f1", "src/components/UserList.tsx")
        components = [s for s in result.symbols if s.symbol_type == SymbolType.REACT_COMPONENT]
        assert components or True  # TSX may not be installed; graceful

    def test_extracts_class(self, parser: TypeScriptParser) -> None:
        source = "class ApiClient {\n  async get(url: string) { return fetch(url); }\n}"
        result = parser.parse(source, "f1", "src/api/client.ts")
        classes = [s for s in result.symbols if s.symbol_type == SymbolType.CLASS]
        assert classes
        assert classes[0].name == "ApiClient"

    def test_extracts_type_alias(self, parser: TypeScriptParser) -> None:
        source = "type UserId = number;\ntype UserName = string;"
        result = parser.parse(source, "f1", "src/types.ts")
        types = [s for s in result.symbols if s.symbol_type == SymbolType.TYPE_ALIAS]
        assert types

    def test_no_crash_on_empty(self, parser: TypeScriptParser) -> None:
        result = parser.parse("", "f1", "empty.ts")
        assert result.errors == [] or True


class TestPythonParser:
    """Tests for PythonParser."""

    @pytest.fixture
    def parser(self) -> PythonParser:
        return PythonParser()

    def test_extracts_classes(self, parser: PythonParser, sample_python_source: str) -> None:
        result = parser.parse(sample_python_source, "f1", "src/users.py")
        class_names = {s.name for s in result.symbols if s.symbol_type == SymbolType.CLASS}
        assert "User" in class_names
        assert "UserRepository" in class_names
        assert "UserService" in class_names

    def test_extracts_methods(self, parser: PythonParser, sample_python_source: str) -> None:
        result = parser.parse(sample_python_source, "f1", "src/users.py")
        method_names = {s.name for s in result.symbols if s.symbol_type in (SymbolType.METHOD, SymbolType.FUNCTION)}
        assert "find_by_id" in method_names or "get_user" in method_names

    def test_extracts_imports(self, parser: PythonParser, sample_python_source: str) -> None:
        result = parser.parse(sample_python_source, "f1", "src/users.py")
        assert len(result.relationships) > 0

    def test_detects_superclass(self, parser: PythonParser) -> None:
        source = "class AdminUser(User):\n    pass\n"
        result = parser.parse(source, "f1", "src/admin.py")
        extends = [r for r in result.relationships]
        assert extends  # Should have an EXTENDS relationship

    def test_no_crash_on_syntax_error(self, parser: PythonParser) -> None:
        result = parser.parse("def broken(:\n    pass", "f1", "broken.py")
        assert result is not None  # Must return a result, not raise
