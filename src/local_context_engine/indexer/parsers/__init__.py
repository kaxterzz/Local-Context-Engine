"""
Code parsers for PHP, TypeScript, JavaScript, Python, C#, ASP.NET, and SQL.

Each parser extracts structured :class:`~local_context_engine.core.types.Symbol`
objects and :class:`~local_context_engine.core.types.Relationship` edges
for the symbol graph.
"""

from local_context_engine.indexer.parsers.base_parser import BaseParser, ParseResult
from local_context_engine.indexer.parsers.registry import ParserRegistry

__all__ = ["BaseParser", "ParseResult", "ParserRegistry"]
