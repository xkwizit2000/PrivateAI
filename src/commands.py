"""Shared slash-command handlers for Telegram and Session front ends."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .config import OLLAMA_THINK, OLLAMA_TIMEOUT_SECONDS
from .ollama_status import fetch_ollama_status
from .session_settings import MAX_TIMEOUT_SECONDS, MIN_TIMEOUT_SECONDS, SessionSettings

_COMMAND_RE = re.compile(r"^/([A-Za-z0-9_]+)(?:@[A-Za-z0-9_]+)?(?:\s+(.*))?$", re.DOTALL)


@dataclass(frozen=True)
class ParsedCommand:
    name: str
    args: list[str]


def parse_slash_command(text: str) -> ParsedCommand | None:
    stripped = text.strip()
    match = _COMMAND_RE.match(stripped)
    if match is None:
        return None
    name = match.group(1).lower()
    rest = (match.group(2) or "").strip()
    args = rest.split() if rest else []
    return ParsedCommand(name=name, args=args)


async def handle_command(
    settings: SessionSettings,
    session_id: str | int,
    command: ParsedCommand,
) -> str | None:
    """Return a reply for a known command, or None if the command is unknown."""
    if command.name == "status":
        return await _status(settings, session_id)
    if command.name == "think":
        return _think(settings, session_id, command.args)
    if command.name == "timeout":
        return _timeout(settings, session_id, command.args)
    if command.name == "verbose":
        return _verbose(settings, session_id, command.args)
    return None


async def _status(settings: SessionSettings, session_id: str | int) -> str:
    session_text = settings.format_status(session_id)
    ollama_text = await fetch_ollama_status()
    return f"{session_text}\n\n{ollama_text}"


def _think(settings: SessionSettings, session_id: str | int, args: list[str]) -> str:
    current = settings.get_think(session_id)
    normalized = [arg.lower() for arg in args]

    if not normalized or normalized[0] in {"status", "show"}:
        return (
            f"Thinking mode is {'on' if current else 'off'} for this chat.\n"
            f"(Default from OLLAMA_THINK is {'on' if OLLAMA_THINK else 'off'}.)\n"
            "Use /think on or /think off to change it.\n"
            "Also: /verbose status, /timeout status, /status"
        )

    if normalized[0] in {"on", "true", "1", "enable"}:
        settings.set_think(session_id, True)
        return (
            "Thinking mode enabled for this chat. Replies may be slower but deeper.\n"
            "Consider raising the timeout, e.g. /timeout 600\n"
            "Status: think=on"
        )

    if normalized[0] in {"off", "false", "0", "disable"}:
        settings.set_think(session_id, False)
        return (
            "Thinking mode disabled for this chat. Replies should be faster.\n"
            "Status: think=off"
        )

    return "Usage: /think [on|off|status]"


def _timeout(settings: SessionSettings, session_id: str | int, args: list[str]) -> str:
    current = settings.get_timeout(session_id)
    normalized = [arg.lower() for arg in args]

    if not normalized or normalized[0] in {"status", "show"}:
        return (
            f"Timeout is {int(current)}s for this chat.\n"
            f"(Default from OLLAMA_TIMEOUT_SECONDS is {int(OLLAMA_TIMEOUT_SECONDS)}s.)\n"
            f"Allowed range: {int(MIN_TIMEOUT_SECONDS)}–{int(MAX_TIMEOUT_SECONDS)}s.\n"
            "Use /timeout <seconds> or /timeout reset to change it.\n"
            "Also: /think status, /verbose status, /status"
        )

    if normalized[0] in {"reset", "default", "clear"}:
        restored = settings.clear_timeout(session_id)
        return (
            f"Timeout reset to default ({int(restored)}s) for this chat.\n"
            f"Status: timeout={int(restored)}s"
        )

    try:
        applied = settings.set_timeout(session_id, float(normalized[0]))
    except ValueError as exc:
        return f"{exc}\nUsage: /timeout [status|<seconds>|reset]"

    return (
        f"Timeout set to {int(applied)}s for this chat.\n"
        f"Status: timeout={int(applied)}s"
    )


def _verbose(settings: SessionSettings, session_id: str | int, args: list[str]) -> str:
    current = settings.get_verbose(session_id)
    think = settings.get_think(session_id)
    normalized = [arg.lower() for arg in args]

    if not normalized or normalized[0] in {"status", "show"}:
        note = ""
        if current and not think:
            note = "\nNote: /think is off, so no reasoning text will stream yet."
        return (
            f"Verbose mode is {'on' if current else 'off'} for this chat.\n"
            "Use /verbose on or /verbose off to change it.\n"
            "With /verbose on and /think on, reasoning streams into the status message."
            f"{note}\n"
            "Also: /think status, /timeout status, /status"
        )

    if normalized[0] in {"on", "true", "1", "enable"}:
        settings.set_verbose(session_id, True)
        extra = ""
        if not think:
            extra = (
                "\n/think is currently off — enable it with /think on to stream reasoning."
            )
        return (
            "Verbose mode enabled. Reasoning will stream into the status when thinking is on."
            f"{extra}\n"
            "Status: verbose=on"
        )

    if normalized[0] in {"off", "false", "0", "disable"}:
        settings.set_verbose(session_id, False)
        return (
            "Verbose mode disabled. Status will show the simple progress indicator.\n"
            "Status: verbose=off"
        )

    return "Usage: /verbose [on|off|status]"
