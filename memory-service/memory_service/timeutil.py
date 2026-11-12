"""Small date helpers (all UTC)."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

_REL = re.compile(r"^\s*(\d+)\s*([dwm])\s*$", re.I)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def parse_ts(value: str | float | int | datetime | None) -> datetime | None:
    """ISO-8601 (date or datetime, 'Z' ok) or epoch seconds -> aware UTC datetime."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), timezone.utc)
    s = str(value).strip()
    if s.endswith(("Z", "z")):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def parse_when(value: str | None, *, end: bool = False, now: datetime | None = None) -> float | None:
    """Parse a since/until bound to epoch seconds.

    Accepts ISO dates/datetimes or relative '7d' / '2w' / '1m' (ago from now).
    A bare date as an *end* bound means end of that day (inclusive).
    """
    if not value or not str(value).strip():
        return None
    now = now or utcnow()
    m = _REL.match(str(value))
    if m:
        n, unit = int(m.group(1)), m.group(2).lower()
        days = n * {"d": 1, "w": 7, "m": 30}[unit]
        return (now - timedelta(days=days)).timestamp()
    dt = parse_ts(value)
    if dt is None:
        return None
    if end and len(str(value).strip()) <= 10:
        dt = dt + timedelta(days=1) - timedelta(microseconds=1)
    return dt.timestamp()


def iso_week(day: str) -> str:
    y, w, _ = datetime.strptime(day, "%Y-%m-%d").isocalendar()
    return f"{y}-W{w:02d}"
