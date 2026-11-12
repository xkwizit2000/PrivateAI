"""Redact secret-looking strings before they enter the archive."""

from __future__ import annotations

import re

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)", re.S)),
    ("telegram_token", re.compile(r"\b\d{8,10}:[A-Za-z0-9_-]{30,}\b")),
    ("api_key", re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}\b")),
    ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b")),
    ("aws_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("bearer", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{20,}")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")),
    (
        "mnemonic",
        re.compile(
            r"(?i)\b(?:mnemonic|seed phrase|recovery phrase|seed words?)\b[^\n]{0,20}?[:=]\s*"
            r"(?:[a-z]{3,9}\s+){11,23}[a-z]{3,9}\b"
        ),
    ),
    (
        "assignment",
        re.compile(
            r"(?i)\b(pass(?:word|wd)?|secret|token|api[_-]?key|private[_-]?key|auth)\b"
            r"(\s*[:=]\s*)(?!\[REDACTED)(?:\"[^\"\n]{4,}\"|'[^'\n]{4,}'|[^\s\"',;]{6,})"
        ),
    ),
)


def redact(text: str) -> tuple[str, int]:
    """Return (text with secrets replaced by [REDACTED:kind], number of replacements)."""
    count = 0
    for kind, pattern in _PATTERNS:
        if kind == "assignment":
            text, n = pattern.subn(lambda m: f"{m.group(1)}{m.group(2)}[REDACTED:secret]", text)
        else:
            text, n = pattern.subn(f"[REDACTED:{kind}]", text)
        count += n
    return text, count
