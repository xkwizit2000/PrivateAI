"""Safe session identifiers for Telegram chat IDs and Session account IDs."""

from __future__ import annotations

import re

# Telegram chat IDs are numeric; Session IDs are hex (typically 66 chars with 05 prefix).
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def safe_session_id(session_id: str | int) -> str:
    value = str(session_id).strip()
    if not value or ".." in value or "/" in value or "\\" in value:
        raise ValueError("Invalid session ID")
    if not _SESSION_ID_RE.fullmatch(value):
        raise ValueError("Invalid session ID")
    return value
