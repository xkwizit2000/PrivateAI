"""Decide per request whether to send MCP tool schemas to the model.

The tool schemas are a large share of every prompt, and the local model
reprocesses the whole prompt on each call, so plain chat turns ("hello",
"what is 2+2") should not carry them. In ``auto`` mode a cheap keyword/regex
check on the user's message decides; follow-ups to a file-access turn keep the
tools on for a short time.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic

from .config import TOOL_ROUTING_FOLLOWUP_SECONDS

ROUTING_MODES = ("auto", "always", "never")

# Words that suggest the user wants file-system access. Entries are regex
# fragments; _word_regex wraps them with word boundaries and common inflections.
_KEYWORD_STEMS = (
    "files?",
    "folders?",
    "director(?:y|ies)",
    "dirs?",
    "paths?",
    "read(?:s|ing)?",
    "open(?:s|ed|ing)?",
    "writ(?:e|es|ing|ten)|wrote",
    "sav(?:e|es|ed|ing)",
    "edit(?:s|ed|ing)?",
    "creat(?:e|es|ed|ing)",
    "delet(?:e|es|ed|ing)",
    "remov(?:e|es|ed|ing)",
    "mov(?:e|es|ed|ing)",
    "renam(?:e|es|ed|ing)",
    "cop(?:y|ies|ied|ying)",
    "search(?:es|ed|ing)?",
    "find(?:s|ing)?|found",
    "list(?:s|ed|ing)?",
    "ls",
    "tree",
    "contents",
    "cat",
    "grep",
    "mkdir",
    "append(?:s|ed|ing)?",
    "filesystem",
    "working\\s*dir(?:ectory)?",
    "workingdir",
)
_KEYWORD_RE = re.compile(r"\b(?:" + "|".join(_KEYWORD_STEMS) + r")\b", re.IGNORECASE)

_EXTENSIONS = (
    "txt|md|markdown|py|json|csv|tsv|yml|yaml|toml|ini|cfg|conf|log|js|ts|jsx|tsx|"
    "html|htm|css|pdf|xml|sh|sql|ipynb|rs|go|java|cpp|rb|php|env|docx|xlsx|zip|tar|gz"
)
_EXT_RE = re.compile(rf"(?<![\w.])[\w-]*\.(?:{_EXTENSIONS})\b", re.IGNORECASE)

# "and/or", "24/7", "1/2" and URLs are not file paths.
_NOT_PATH_RE = re.compile(r"\b(?:and/or|either/or|w/o|n/a|\d+/\d+)\b|https?://\S+", re.IGNORECASE)
_PATH_RE = re.compile(r"[\w.~-]*[/\\][\w.~/\\-]*|[/\\]")


@dataclass(frozen=True)
class RoutingDecision:
    """Whether to send tools for one request, and why (for logging)."""

    tools_on: bool
    reason: str
    # True when the user message itself looked like it needs file access.
    matched: bool = False


def match_file_access(text: str) -> str | None:
    """Return a short description of the first file-access cue in text, else None."""
    keyword = _KEYWORD_RE.search(text)
    if keyword:
        return f"keyword '{keyword.group(0).lower()}'"
    cleaned = _NOT_PATH_RE.sub(" ", text)
    if _PATH_RE.search(cleaned):
        return "path-like token"
    ext = _EXT_RE.search(text)
    if ext:
        return f"file extension '{ext.group(0).lower()}'"
    return None


class ToolRouter:
    """Per-session short-lived "tools stay on" state plus the routing decision."""

    def __init__(
        self,
        followup_seconds: float = TOOL_ROUTING_FOLLOWUP_SECONDS,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        """Create a router; followup_seconds is how long follow-ups keep tools on."""
        self.followup_seconds = followup_seconds
        self._clock = clock
        self._tools_until: dict[str, float] = {}

    def _key(self, session_id: str | int | None) -> str | None:
        """Normalize a session id for the in-memory map."""
        return None if session_id is None else str(session_id)

    def mark_active(self, session_id: str | int | None) -> None:
        """Keep tools on for follow-ups of this session for the follow-up window."""
        key = self._key(session_id)
        if key is not None and self.followup_seconds > 0:
            self._tools_until[key] = self._clock() + self.followup_seconds

    def followup_active(self, session_id: str | int | None) -> bool:
        """True while the session is inside its follow-up window."""
        key = self._key(session_id)
        if key is None:
            return False
        until = self._tools_until.get(key)
        if until is None:
            return False
        if self._clock() >= until:
            del self._tools_until[key]
            return False
        return True

    def decide(
        self, mode: str, session_id: str | int | None, text: str
    ) -> RoutingDecision:
        """Decide tools on/off for this request and refresh follow-up state."""
        if mode == "always":
            return RoutingDecision(True, "mode=always")
        if mode == "never":
            return RoutingDecision(False, "mode=never")

        cue = match_file_access(text)
        if cue is not None:
            self.mark_active(session_id)
            return RoutingDecision(True, f"matched {cue}", matched=True)
        if self.followup_active(session_id):
            return RoutingDecision(True, "follow-up after file-access turn")
        return RoutingDecision(False, "no file-access cue in message")


# Process-wide router used by the agent loop.
DEFAULT_ROUTER = ToolRouter()
