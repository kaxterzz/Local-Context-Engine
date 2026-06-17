"""
Parser registry — maps file languages to parser instances.

Uses lazy initialisation so Tree-sitter grammars are only loaded
when the relevant language is first encountered.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from local_context_engine.core.types import Language
from local_context_engine.indexer.parsers.base_parser import BaseParser, ParseResult

logger = logging.getLogger(__name__)


class ParserRegistry:
    """
    Centralised lookup for :class:`BaseParser` instances.

    Usage::

        registry = ParserRegistry.default()
        result = registry.parse(source, language, file_id, file_path)
    """

    def __init__(self) -> None:
        self._parsers: dict[Language, BaseParser] = {}

    def register(self, parser: BaseParser) -> None:
        """Register a parser for its declared language."""
        self._parsers[parser.supported_language] = parser
        logger.debug("Registered parser for %s", parser.supported_language)

    def get(self, language: Language) -> BaseParser | None:
        """Return the parser for *language*, or ``None`` if unsupported."""
        return self._parsers.get(language)

    def parse(
        self, source: str, language: Language, file_id: str, file_path: str
    ) -> ParseResult:
        """
        Parse *source* using the registered parser for *language*.

        Returns an empty :class:`ParseResult` with an error message if no
        parser is available — never raises.
        """
        parser = self.get(language)
        if parser is None:
            logger.debug("No parser for language %s (%s)", language, file_path)
            return ParseResult(
                file_id=file_id,
                file_path=file_path,
                language=language,
                errors=[f"No parser registered for language '{language.value}'"],
            )
        return parser.parse(source, file_id, file_path)

    def supported_languages(self) -> list[Language]:
        return list(self._parsers.keys())

    @classmethod
    @lru_cache(maxsize=1)
    def default(cls) -> "ParserRegistry":
        """
        Create and return the default parser registry with all built-in parsers.

        The registry is cached as a singleton after first creation.
        """
        from local_context_engine.indexer.parsers.php_parser import PHPParser
        from local_context_engine.indexer.parsers.python_parser import PythonParser
        from local_context_engine.indexer.parsers.typescript_parser import TypeScriptParser

        registry = cls()

        # PHP
        registry.register(PHPParser())

        # TypeScript
        ts_parser = TypeScriptParser(Language.TYPESCRIPT)
        registry.register(ts_parser)

        # TSX — separate parser for TSX grammar
        tsx_parser = TypeScriptParser(Language.TSX)
        registry._parsers[Language.TSX] = tsx_parser

        # JavaScript
        js_parser = TypeScriptParser(Language.JAVASCRIPT)
        registry._parsers[Language.JAVASCRIPT] = js_parser

        # JSX
        jsx_parser = TypeScriptParser(Language.JSX)
        registry._parsers[Language.JSX] = jsx_parser

        # Python
        registry.register(PythonParser())

        logger.info(
            "ParserRegistry initialised with %d parsers: %s",
            len(registry._parsers),
            [lang.value for lang in registry._parsers],
        )
        return registry
