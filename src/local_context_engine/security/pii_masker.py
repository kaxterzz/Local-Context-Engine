"""
PII (Personally Identifiable Information) masking engine.

Replaces sensitive data with typed placeholders BEFORE embeddings are generated.
This ensures that even if embeddings are extracted from the vector store,
they do not encode sensitive information.

Masking is performed on ``content_masked`` — the original ``content`` is
preserved in the database but never fed to the embedding model.

Supported patterns:
  - Email addresses
  - Phone numbers (international formats)
  - National Identity Card numbers (Sri Lanka IC / generic patterns)
  - Tax Identification Numbers (TIN)
  - VAT registration numbers
  - JWT tokens
  - Bearer tokens / API keys
  - OAuth access/refresh tokens
  - Signed URLs (AWS, GCS, Azure SAS)
  - IP addresses (optional)
  - Credit card numbers
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import NamedTuple

# Use the ``regex`` library for better Unicode support and POSIX character classes
import regex


@dataclass
class MaskingStats:
    """Counts of each PII type found in a single masking operation."""

    emails: int = 0
    phones: int = 0
    nics: int = 0
    tins: int = 0
    vats: int = 0
    jwts: int = 0
    api_keys: int = 0
    access_tokens: int = 0
    signed_urls: int = 0
    ip_addresses: int = 0
    credit_cards: int = 0

    @property
    def total(self) -> int:
        return (
            self.emails + self.phones + self.nics + self.tins + self.vats
            + self.jwts + self.api_keys + self.access_tokens + self.signed_urls
            + self.ip_addresses + self.credit_cards
        )


class _Rule(NamedTuple):
    name: str
    pattern: re.Pattern[str]
    placeholder: str
    stat_field: str


def _compile(pattern: str, flags: int = 0) -> re.Pattern[str]:
    return regex.compile(pattern, flags | regex.IGNORECASE | regex.MULTILINE)


# ─────────────────────────────────────────────────────────────
# Regex patterns
# ─────────────────────────────────────────────────────────────

_EMAIL_PATTERN = _compile(
    r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}"
)

_PHONE_PATTERN = _compile(
    r"""
    (?:
        \+?(?:1[-.\s]?)?     # country code optional
        (?:\(\d{1,4}\)|\d{1,4})  # area code
        [-.\s]?
        \d{3}
        [-.\s]?
        \d{4,9}
    )
    """,
    re.VERBOSE,
)

# Sri Lanka NIC: 9 digits + V/X, or 12 digits
_NIC_SL_PATTERN = _compile(
    r"\b(?:\d{9}[VXvx]|\d{12})\b"
)

# Generic TIN (US EIN, UK UTR, various formats)
_TIN_PATTERN = _compile(
    r"\b(?:\d{2}-\d{7}|\d{3}-\d{2}-\d{4}|\d{10})\b"
)

# VAT: country code + digits, e.g. GB123456789, LK1234567890
_VAT_PATTERN = _compile(
    r"\b[A-Z]{2}\d{8,12}\b"
)

# JWT: three base64url segments separated by dots
_JWT_PATTERN = _compile(
    r"\beyJ[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+\b"
)

# Bearer tokens / Authorization header values
_BEARER_PATTERN = _compile(
    r"(?:Bearer|Token|Basic|Digest)\s+([A-Za-z0-9\-_.~+/=]{20,})"
)

# Generic API key patterns: long alphanumeric strings in known key positions
_API_KEY_PATTERN = _compile(
    r"""
    (?:
        api[_\-]?key
        | secret[_\-]?key
        | private[_\-]?key
        | access[_\-]?key
        | client[_\-]?secret
        | app[_\-]?secret
        | auth[_\-]?token
        | app[_\-]?key
        | service[_\-]?key
    )
    \s*(?:=|:|\"|')\s*
    ['""]?([A-Za-z0-9\-_.+/]{20,})['""]?
    """,
    re.VERBOSE,
)

# OAuth / session tokens in variable assignments
_ACCESS_TOKEN_PATTERN = _compile(
    r"""
    (?:
        access[_\-]?token
        | refresh[_\-]?token
        | id[_\-]?token
        | session[_\-]?token
        | auth[_\-]?token
        | csrf[_\-]?token
    )
    \s*(?:=|:|\"|')\s*
    ['""]?([A-Za-z0-9\-_.+/]{20,})['""]?
    """,
    re.VERBOSE,
)

# AWS signed URLs, GCS signed URLs, Azure SAS tokens
_SIGNED_URL_PATTERN = _compile(
    r"""
    https?://[^\s"']+
    (?:
        [?&](?:X-Amz-Signature|x-goog-signature|sig|sv=)[^&\s"']+
        | [?&](?:X-Amz-Security-Token|X-Amz-Credential)[^&\s"']+
        | [?&]sas=[^&\s"']+
    )
    [^\s"']*
    """,
    re.VERBOSE,
)

# IPv4 addresses
_IP_V4_PATTERN = _compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
)

# Credit card numbers (Luhn-valid patterns: Visa, Mastercard, AmEx, etc.)
_CREDIT_CARD_PATTERN = _compile(
    r"""
    \b
    (?:
        4\d{12}(?:\d{3})?         # Visa
        | 5[1-5]\d{14}            # Mastercard
        | 3[47]\d{13}             # AmEx
        | 3(?:0[0-5]|[68]\d)\d{11}  # Diners Club
        | 6(?:011|5\d{2})\d{12}  # Discover
        | (?:2131|1800|35\d{3})\d{11}  # JCB
    )
    \b
    """,
    re.VERBOSE,
)


class PIIMasker:
    """
    Masks PII in source code text before embedding generation.

    Usage::

        masker = PIIMasker.from_config(config.pii_masking)
        masked_text, stats = masker.mask(original_text)
    """

    def __init__(
        self,
        mask_email: bool = True,
        mask_phone: bool = True,
        mask_nic: bool = True,
        mask_tin: bool = True,
        mask_vat: bool = True,
        mask_jwt: bool = True,
        mask_api_keys: bool = True,
        mask_access_tokens: bool = True,
        mask_signed_urls: bool = True,
        mask_ip_addresses: bool = False,
        mask_credit_cards: bool = True,
    ) -> None:
        self._rules: list[_Rule] = []

        if mask_signed_urls:
            # Must run before generic URL/email patterns
            self._rules.append(
                _Rule("signed_url", _SIGNED_URL_PATTERN, "[REDACTED_SIGNED_URL]", "signed_urls")
            )
        if mask_jwt:
            self._rules.append(_Rule("jwt", _JWT_PATTERN, "[REDACTED_JWT]", "jwts"))
        if mask_api_keys:
            self._rules.append(
                _Rule("api_key", _API_KEY_PATTERN, "[REDACTED_API_KEY]", "api_keys")
            )
        if mask_access_tokens:
            self._rules.append(
                _Rule("access_token", _ACCESS_TOKEN_PATTERN, "[REDACTED_TOKEN]", "access_tokens")
            )
        if mask_email:
            self._rules.append(_Rule("email", _EMAIL_PATTERN, "[REDACTED_EMAIL]", "emails"))
        if mask_phone:
            self._rules.append(_Rule("phone", _PHONE_PATTERN, "[REDACTED_PHONE]", "phones"))
        if mask_nic:
            self._rules.append(_Rule("nic", _NIC_SL_PATTERN, "[REDACTED_NIC]", "nics"))
        if mask_tin:
            self._rules.append(_Rule("tin", _TIN_PATTERN, "[REDACTED_TIN]", "tins"))
        if mask_vat:
            self._rules.append(_Rule("vat", _VAT_PATTERN, "[REDACTED_VAT]", "vats"))
        if mask_credit_cards:
            self._rules.append(
                _Rule("credit_card", _CREDIT_CARD_PATTERN, "[REDACTED_CC]", "credit_cards")
            )
        if mask_ip_addresses:
            self._rules.append(
                _Rule("ip_address", _IP_V4_PATTERN, "[REDACTED_IP]", "ip_addresses")
            )

    # ── Public API ────────────────────────────────────────────

    def mask(self, text: str) -> tuple[str, MaskingStats]:
        """
        Apply all active masking rules to ``text``.

        Args:
            text: Source code text to mask.

        Returns:
            Tuple of (masked_text, stats). The original ``text`` is never
            modified; a new string is returned.
        """
        stats = MaskingStats()
        result = text

        for rule in self._rules:
            def _replace(m: re.Match[str], _placeholder: str = rule.placeholder) -> str:
                return _placeholder

            new_result, count = rule.pattern.subn(_replace, result)
            result = new_result
            # Accumulate counts
            current = getattr(stats, rule.stat_field)
            setattr(stats, rule.stat_field, current + count)

        return result, stats

    def mask_text(self, text: str) -> str:
        """Convenience method — returns only the masked string."""
        masked, _ = self.mask(text)
        return masked

    def contains_pii(self, text: str) -> bool:
        """Return ``True`` if ``text`` contains any detectable PII."""
        for rule in self._rules:
            if rule.pattern.search(text):
                return True
        return False

    @classmethod
    def from_config(cls, config: "PIIMaskingConfig") -> "PIIMasker":  # type: ignore[name-defined]
        """Create a :class:`PIIMasker` from a :class:`PIIMaskingConfig`."""
        from local_context_engine.core.config import PIIMaskingConfig  # noqa: PLC0415

        assert isinstance(config, PIIMaskingConfig)
        return cls(
            mask_email=config.mask_email,
            mask_phone=config.mask_phone,
            mask_nic=config.mask_nic,
            mask_tin=config.mask_tin,
            mask_vat=config.mask_vat,
            mask_jwt=config.mask_jwt,
            mask_api_keys=config.mask_api_keys,
            mask_access_tokens=config.mask_access_tokens,
            mask_signed_urls=config.mask_signed_urls,
            mask_ip_addresses=config.mask_ip_addresses,
            mask_credit_cards=config.mask_credit_cards,
        )
