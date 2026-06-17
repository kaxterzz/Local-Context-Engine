"""
Token counting utilities.

Provides fast approximate token counting without requiring the full
``tiktoken`` library. The approximation uses ~4 characters per token,
which is accurate enough for chunk boundary decisions.

If ``tiktoken`` is installed, the more accurate ``cl100k_base`` tokenizer
is used instead (OpenAI's tokenizer, shared by many LLMs).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_tiktoken_available = False
_enc = None

try:
    import tiktoken

    _enc = tiktoken.get_encoding("cl100k_base")
    _tiktoken_available = True
    logger.debug("tiktoken available — using cl100k_base for token counting.")
except ImportError:
    logger.debug("tiktoken not installed — using character-based approximation.")


class TokenCounter:
    """
    Counts tokens in text for chunk boundary decisions.

    Prefers ``tiktoken`` (cl100k_base) when available.
    Falls back to ``len(text) // 4`` approximation otherwise.
    """

    def __init__(self, method: str = "auto") -> None:
        """
        Args:
            method: ``"auto"`` selects tiktoken if available, else approx.
                    ``"approx"`` always uses the fast approximation.
                    ``"tiktoken"`` forces tiktoken (raises if unavailable).
        """
        if method == "tiktoken":
            if not _tiktoken_available:
                raise ImportError("tiktoken is not installed. Run: pip install tiktoken")
            self._use_tiktoken = True
        elif method == "approx":
            self._use_tiktoken = False
        else:  # "auto"
            self._use_tiktoken = _tiktoken_available

    def count(self, text: str) -> int:
        """Return the approximate token count for *text*."""
        if self._use_tiktoken and _enc is not None:
            return len(_enc.encode(text, disallowed_special=()))
        # Approximation: 4 chars ≈ 1 token (accurate within ~10% for code)
        return max(1, len(text) // 4)

    def split_at_token_boundary(
        self, text: str, max_tokens: int, overlap_tokens: int = 0
    ) -> list[str]:
        """
        Split *text* into segments each containing at most *max_tokens*.

        Args:
            text:           Input text to split.
            max_tokens:     Maximum tokens per segment.
            overlap_tokens: Number of tokens to repeat from the end of each
                            segment into the start of the next (context overlap).

        Returns:
            List of text segments. The last segment may be shorter.
        """
        if self.count(text) <= max_tokens:
            return [text]

        lines = text.splitlines(keepends=True)
        segments: list[str] = []
        current_lines: list[str] = []
        current_tokens = 0
        overlap_buffer: list[str] = []

        for line in lines:
            line_tokens = self.count(line)

            if current_tokens + line_tokens > max_tokens and current_lines:
                # Flush current segment
                segment = "".join(current_lines)
                segments.append(segment)

                # Compute overlap buffer from end of current segment
                overlap_lines: list[str] = []
                overlap_count = 0
                for prev_line in reversed(current_lines):
                    lt = self.count(prev_line)
                    if overlap_count + lt > overlap_tokens:
                        break
                    overlap_lines.insert(0, prev_line)
                    overlap_count += lt

                current_lines = overlap_lines + [line]
                current_tokens = overlap_count + line_tokens
            else:
                current_lines.append(line)
                current_tokens += line_tokens

        if current_lines:
            segments.append("".join(current_lines))

        return segments
