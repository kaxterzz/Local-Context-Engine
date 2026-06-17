"""
Security layer for Local Context Engine.

Enforces three guarantees:
  1. File blacklisting  — sensitive files are never scanned or indexed.
  2. PII masking        — personal data is replaced before embeddings are generated.
  3. Content redaction  — a secondary sweep over content before it leaves the engine.
"""

from local_context_engine.security.file_blacklist import FileBlacklist
from local_context_engine.security.pii_masker import PIIMasker
from local_context_engine.security.redactor import ContentRedactor

__all__ = ["FileBlacklist", "PIIMasker", "ContentRedactor"]
