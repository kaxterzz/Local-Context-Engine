"""Unit tests for React relationship inference."""

from __future__ import annotations

from local_context_engine.core.types import Language, RelationshipType, Symbol, SymbolType
from local_context_engine.symbol_graph.analyzers.react_analyzer import ReactAnalyzer


def _symbol(
    symbol_id: str,
    name: str,
    symbol_type: SymbolType,
    file_path: str,
    line_start: int,
    line_end: int,
) -> Symbol:
    return Symbol(
        id=symbol_id,
        file_id=f"file-{symbol_id}",
        file_path=file_path,
        name=name,
        qualified_name=name,
        symbol_type=symbol_type,
        line_start=line_start,
        line_end=line_end,
        language=Language.TSX if file_path.endswith(".tsx") else Language.TYPESCRIPT,
    )


def test_infers_component_hook_and_mutation_function_calls() -> None:
    symbols = [
        _symbol(
            "login-page",
            "LoginPage",
            SymbolType.REACT_COMPONENT,
            "app/pages/LoginPage.tsx",
            1,
            3,
        ),
        _symbol("use-login", "useLogin", SymbolType.REACT_HOOK, "app/api/auth.ts", 1, 5),
        _symbol("login-api", "loginApi", SymbolType.FUNCTION, "app/api/auth.ts", 7, 9),
    ]
    sources = {
        "app/pages/LoginPage.tsx": (
            "const LoginPage = () => {\n"
            "  const loginMutation = useLogin();\n"
            "}"
        ),
        "app/api/auth.ts": (
            "export const useLogin = () => {\n"
            "  return useMutation({\n"
            "    mutationFn: loginApi,\n"
            "  });\n"
            "}\n\n"
            "const loginApi = async () => {};"
        ),
    }

    relationships = ReactAnalyzer().analyze(symbols, sources)
    call_edges = {
        (rel.source_symbol_id, rel.target_symbol_id, rel.relationship_type)
        for rel in relationships
    }

    assert ("login-page", "use-login", RelationshipType.CALLS) in call_edges
    assert ("use-login", "login-api", RelationshipType.CALLS) in call_edges
