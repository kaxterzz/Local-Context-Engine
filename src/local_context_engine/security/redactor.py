"""
Content redactor — a final sweep over text before it reaches the MCP layer.

The redactor sits at the output boundary and ensures that even if a chunk
slips through the PII masker (e.g. because a new pattern wasn't yet covered),
it receives a secondary pass before being returned to the calling AI agent.

Redaction is intentionally lightweight and fast; the heavy lifting is done
by :class:`~local_context_engine.security.pii_masker.PIIMasker`.
"""

from __future__ import annotations

from local_context_engine.security.pii_masker import PIIMasker


class ContentRedactor:
    """
    Applies masking to any text string that will be returned to an agent.

    Usage::

        redactor = ContentRedactor(masker)
        safe_snippet = redactor.redact(raw_snippet)
    """

    def __init__(self, masker: PIIMasker) -> None:
        self._masker = masker

    def redact(self, text: str) -> str:
        """Apply PII masking to *text* and return the sanitised version."""
        return self._masker.mask_text(text)

    def redact_result(self, result: "SearchResult") -> "SearchResult":  # type: ignore[name-defined]
        """Redact the ``snippet`` field of a :class:`~local_context_engine.core.types.SearchResult`."""
        from local_context_engine.core.types import SearchResult  # noqa: PLC0415

        assert isinstance(result, SearchResult)
        return SearchResult(
            chunk_id=result.chunk_id,
            file_path=result.file_path,
            symbol_name=result.symbol_name,
            symbol_type=result.symbol_type,
            language=result.language,
            snippet=self.redact(result.snippet),
            line_start=result.line_start,
            line_end=result.line_end,
            score=result.score,
            score_breakdown=result.score_breakdown,
            metadata=result.metadata,
        )

    def redact_results(self, results: list["SearchResult"]) -> list["SearchResult"]:  # type: ignore[name-defined]
        """Redact all snippets in a list of search results."""
        return [self.redact_result(r) for r in results]

    @classmethod
    def from_masker(cls, masker: PIIMasker) -> "ContentRedactor":
        """Create a redactor wrapping the given masker."""
        return cls(masker)
