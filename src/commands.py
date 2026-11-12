"""Shared slash-command handlers for Telegram and Session front ends."""

from __future__ import annotations

import re
from dataclasses import dataclass

from typing import Any

from .config import OLLAMA_THINK, OLLAMA_TIMEOUT_SECONDS
from .memory_service_client import ServiceMemory
from .status_report import build_status_report
from .session_settings import (
    MAX_TIMEOUT_SECONDS,
    MIN_TIMEOUT_SECONDS,
    TOOL_MODE_LABELS,
    SessionSettings,
)

_COMMAND_RE = re.compile(r"^/([A-Za-z0-9_]+)(?:@[A-Za-z0-9_]+)?(?:\s+(.*))?$", re.DOTALL)


@dataclass(frozen=True)
class ParsedCommand:
    """A slash command name plus whitespace-split arguments."""

    name: str
    args: list[str]


def parse_slash_command(text: str) -> ParsedCommand | None:
    """Parse a leading /command (optionally @bot) from message text, or return None."""
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
    memory_service: ServiceMemory | None = None,
    runtime: Any = None,
) -> str | None:
    """Return a reply for a known command, or None if the command is unknown.

    ``runtime`` (the AppRuntime) gives /status access to the hub, memory and RAG
    stores; without it /status reports those sections as n/a.
    """
    if command.name == "status":
        return await _status(settings, session_id, command.args, runtime, memory_service)
    if command.name == "think":
        return _think(settings, session_id, command.args)
    if command.name == "timeout":
        return _timeout(settings, session_id, command.args)
    if command.name == "verbose":
        return _verbose(settings, session_id, command.args)
    if command.name == "tools":
        return _tools(settings, session_id, command.args)
    if command.name == "memory":
        return await _memory(memory_service)
    if command.name == "recall":
        return await _recall(memory_service, command.args)
    return None


async def _memory(service: ServiceMemory | None) -> str:
    """Handle /memory: service reachability, archive counts and shadow stats since start."""
    if service is None:
        return (
            "Memory service: not configured (MEMORY_SERVICE_URL is empty).\n"
            "Using the legacy summary + RAG memory only."
        )
    stats = await service.stats()
    if stats is None:
        reach = "Memory service: UNREACHABLE (calls are bounded by short timeouts)."
    else:
        reach = (
            "Memory service: reachable. "
            f"turns={stats.get('turns')} chunks={stats.get('chunks')} "
            f"sessions={stats.get('sessions')} days={stats.get('days')} "
            f"facts={stats.get('facts_active')}"
        )
    return f"{reach}\n{service.status_text()}"


async def _recall(service: ServiceMemory | None, args: list[str]) -> str:
    """Handle /recall <query>: one explicit search of the archive (capped text)."""
    if service is None:
        return "Memory service is not configured (MEMORY_SERVICE_URL is empty)."
    query = " ".join(args).strip()
    if not query:
        return "Usage: /recall <what to look for in earlier conversations>"
    text = await service.search(query)
    if text is None:
        return "Memory service is unavailable right now; try again later."
    return text or "No matching memory."


async def _status(
    settings: SessionSettings,
    session_id: str | int,
    args: list[str],
    runtime: Any = None,
    memory_service: ServiceMemory | None = None,
) -> str:
    """Build the /status reply: /status = compact, /status full = detailed."""
    full = bool(args) and args[0].lower() in {"full", "all", "detail", "details", "verbose"}
    if args and not full:
        return "Usage: /status [full]"
    return await build_status_report(
        settings=settings,
        session_id=session_id,
        memory=getattr(runtime, "memory", None),
        rag=getattr(runtime, "rag", None),
        hub=getattr(runtime, "hub", None),
        memory_service=getattr(runtime, "memory_service", None) or memory_service,
        full=full,
    )


def _think(settings: SessionSettings, session_id: str | int, args: list[str]) -> str:
    """Handle /think to show or toggle model reasoning mode for this chat."""
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
    """Handle /timeout to show, set, or reset the Ollama chat timeout for this chat."""
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
    """Handle /verbose to show or toggle streaming of reasoning into progress UI."""
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


_TOOLS_WORDS = {
    "auto": "auto",
    "on": "always",
    "true": "always",
    "1": "always",
    "always": "always",
    "off": "never",
    "false": "never",
    "0": "never",
    "never": "never",
}


def _tools(settings: SessionSettings, session_id: str | int, args: list[str]) -> str:
    """Handle /tools to show or set tool routing (auto|on|off) for this chat."""
    current = settings.get_tool_mode(session_id)
    normalized = [arg.lower() for arg in args]

    if not normalized or normalized[0] in {"status", "show"}:
        return (
            f"Tools are {TOOL_MODE_LABELS[current]} for this chat "
            f"(default {TOOL_MODE_LABELS.get(settings.default_tools, settings.default_tools)}).\n"
            "auto = send file tools only when the message looks like it needs them "
            "(faster plain chat); on = always send them; off = never.\n"
            "Use /tools auto, /tools on, /tools off or /tools reset."
        )

    if normalized[0] in {"reset", "default", "clear"}:
        restored = settings.clear_tool_mode(session_id)
        return (
            f"Tools reset to default ({TOOL_MODE_LABELS.get(restored, restored)}) "
            f"for this chat.\nStatus: tools={TOOL_MODE_LABELS.get(restored, restored)}"
        )

    mode = _TOOLS_WORDS.get(normalized[0])
    if mode is None:
        return "Usage: /tools [auto|on|off|reset|status]"
    settings.set_tool_mode(session_id, mode)
    return f"Tool routing set for this chat.\nStatus: tools={TOOL_MODE_LABELS[mode]}"
