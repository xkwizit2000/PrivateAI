"""MCP client hub: connect to configured servers and call their tools."""

from __future__ import annotations

import copy
import json
import logging
import re
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .config import (
    MCP_PARAM_DESC_MAX_CHARS,
    MCP_READONLY,
    MCP_TOOL_ALLOWLIST,
    MCP_TOOL_DESC_MAX_CHARS,
)

logger = logging.getLogger(__name__)

# Heuristic blocklist used when MCP_READONLY is enabled.
_WRITE_TOOL_MARKERS = (
    "write",
    "edit",
    "create",
    "move",
    "delete",
    "remove",
    "mkdir",
    "unlink",
    "rename",
    "append",
)


# Conservative default exposure (smaller prompt): drops redundant/low-value tools
# (read_file = deprecated alias of read_text_file, read_media_file, append_file,
# copy_file, list_directory_with_sizes, get_file_info, list_allowed_directories).
DEFAULT_TOOL_ALLOWLIST: tuple[str, ...] = (
    "read_text_file",
    "read_multiple_files",
    "write_file",
    "edit_file",
    "create_directory",
    "list_directory",
    "directory_tree",
    "move_file",
    "search_files",
)

_ALLOW_ALL_VALUES = {"*", "all"}


def parse_tool_allowlist(raw: str | None) -> frozenset[str] | None:
    """Parse MCP_TOOL_ALLOWLIST; return None to expose every tool.

    Empty/blank means "use the default list"; ``*`` or ``all`` (anywhere in the
    list) means everything. Entries are lower-cased.
    """
    entries = [item.strip().lower() for item in (raw or "").split(",") if item.strip()]
    if not entries:
        return frozenset(name.lower() for name in DEFAULT_TOOL_ALLOWLIST)
    if any(item in _ALLOW_ALL_VALUES for item in entries):
        return None
    return frozenset(entries)


def tool_name_candidates(server_name: str, tool_name: str, qualified: str) -> set[str]:
    """Lower-cased names an allowlist entry may use to refer to one tool."""
    return {
        qualified.lower(),
        tool_name.lower(),
        f"{server_name}.{tool_name}".lower(),
    }


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[`])")


def trim_description(text: str, max_chars: int) -> str:
    """Shorten text to at most ~max_chars at a sentence (else word) boundary.

    ``max_chars <= 0`` means unlimited and returns the text unchanged. Text that
    already fits is returned with whitespace collapsed. Never cuts mid-word; a
    cut inside the first sentence ends with an ellipsis.
    """
    if max_chars <= 0 or not text:
        return text
    collapsed = " ".join(text.split())
    if len(collapsed) <= max_chars:
        return collapsed

    kept = ""
    for sentence in _SENTENCE_SPLIT_RE.split(collapsed):
        candidate = f"{kept} {sentence}".strip()
        if len(candidate) > max_chars:
            break
        kept = candidate
    if kept:
        return kept

    # The first sentence alone is too long: cut at the last word boundary.
    budget = max(max_chars - 3, 1)
    cut = collapsed[:budget]
    if collapsed[budget : budget + 1].strip():  # we landed mid-word
        space = cut.rfind(" ")
        if space > 0:
            cut = cut[:space]
    return cut.rstrip(" ,;:-") + "..."


_NESTED_SCHEMA_KEYS = ("items", "additionalProperties", "not", "if", "then", "else")
_NESTED_SCHEMA_LIST_KEYS = ("anyOf", "oneOf", "allOf", "prefixItems")
_NESTED_SCHEMA_MAP_KEYS = ("properties", "$defs", "definitions", "patternProperties")


def trim_schema_descriptions(schema: Any, max_chars: int) -> Any:
    """Return a deep copy of a JSON schema with ``description`` strings trimmed.

    Only the text of schema-node ``description`` values changes; names, types,
    ``required``, ``enum`` and the overall structure are untouched.
    """
    result = copy.deepcopy(schema)
    if max_chars > 0:
        _trim_schema_node(result, max_chars)
    return result


def _trim_schema_node(node: Any, max_chars: int) -> None:
    """In-place helper for trim_schema_descriptions (walks schema nodes only)."""
    if not isinstance(node, dict):
        return
    description = node.get("description")
    if isinstance(description, str):
        node["description"] = trim_description(description, max_chars)
    for key in _NESTED_SCHEMA_MAP_KEYS:
        mapping = node.get(key)
        if isinstance(mapping, dict):
            for child in mapping.values():
                _trim_schema_node(child, max_chars)
    for key in _NESTED_SCHEMA_KEYS:
        _trim_schema_node(node.get(key), max_chars)
    for key in _NESTED_SCHEMA_LIST_KEYS:
        children = node.get(key)
        if isinstance(children, list):
            for child in children:
                _trim_schema_node(child, max_chars)


def _sanitize_name(value: str) -> str:
    """Replace characters unsafe for Ollama tool names with underscores."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", value)


def _is_write_tool(name: str) -> bool:
    """Heuristic: True when a tool name looks like it can mutate data."""
    lowered = name.lower()
    return any(marker in lowered for marker in _WRITE_TOOL_MARKERS)


class McpHub:
    """Owns stdio MCP server subprocesses for the bot process lifetime."""

    def __init__(
        self,
        config_path: Path,
        readonly: bool | None = None,
        *,
        allowlist: str | None = None,
        desc_max_chars: int | None = None,
        param_desc_max_chars: int | None = None,
    ) -> None:
        """Load MCP config path plus readonly, allowlist and description-trim settings.

        ``allowlist`` is the raw MCP_TOOL_ALLOWLIST string (None = read from config).
        """
        self.config_path = config_path
        self.readonly = MCP_READONLY if readonly is None else readonly
        self.allowlist = parse_tool_allowlist(
            MCP_TOOL_ALLOWLIST if allowlist is None else allowlist
        )
        self.desc_max_chars = (
            MCP_TOOL_DESC_MAX_CHARS if desc_max_chars is None else desc_max_chars
        )
        self.param_desc_max_chars = (
            MCP_PARAM_DESC_MAX_CHARS if param_desc_max_chars is None else param_desc_max_chars
        )
        # Qualified names of tools hidden by readonly/allowlist (for friendly errors).
        self._hidden: set[str] = set()
        # Every name form seen from servers, to warn about unknown allowlist entries.
        self._seen_names: set[str] = set()
        self._exit_stack = AsyncExitStack()
        self._sessions: dict[str, ClientSession] = {}
        # qualified_name -> metadata for routing tool calls
        self._tools: dict[str, dict[str, Any]] = {}

    @property
    def tool_names(self) -> list[str]:
        """Sorted list of qualified tool names currently registered."""
        return sorted(self._tools)

    async def start(self) -> None:
        """Connect to every mcpServers entry in the config file."""
        if not self.config_path.exists():
            logger.warning("MCP config not found at %s; running without tools.", self.config_path)
            return

        raw = self.config_path.read_text(encoding="utf-8").strip()
        if not raw:
            logger.warning("MCP config %s is empty; running without tools.", self.config_path)
            return

        config = json.loads(raw)
        servers = config.get("mcpServers") or {}
        if not servers:
            logger.warning("No mcpServers defined in %s", self.config_path)
            return

        for server_name, server_cfg in servers.items():
            await self._connect_server(server_name, server_cfg)

        self.warn_unknown_allowlist_entries()
        logger.info(
            "MCP hub ready: %d server(s), %d tool(s) exposed, %d hidden",
            len(self._sessions),
            len(self._tools),
            len(self._hidden),
        )
        logger.info("Exposed tools (%d): %s", len(self._tools), ", ".join(self.tool_names))
        if self._hidden:
            logger.info("Hidden tools: %s", ", ".join(sorted(self._hidden)))
        full = len(json.dumps(self.as_ollama_tools(trimmed=False)))
        sent = len(json.dumps(self.as_ollama_tools()))
        logger.info("Tool schema size: %d chars sent (%d untrimmed)", sent, full)

    async def _connect_server(self, server_name: str, server_cfg: dict[str, Any]) -> None:
        """Spawn one stdio MCP server and register its exposed tools."""
        command = server_cfg.get("command")
        if not command:
            raise ValueError(f"MCP server '{server_name}' is missing 'command'")

        params = StdioServerParameters(
            command=command,
            args=server_cfg.get("args") or [],
            env=server_cfg.get("env"),
        )

        logger.info("Starting MCP server '%s': %s %s", server_name, command, params.args)
        read, write = await self._exit_stack.enter_async_context(stdio_client(params))
        session = await self._exit_stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self._sessions[server_name] = session

        listed = await session.list_tools()
        for tool in listed.tools:
            self._register_tool(server_name, session, tool)

    def _register_tool(self, server_name: str, session: Any, tool: Any) -> bool:
        """Register one server tool unless readonly/allowlist hides it; True if exposed."""
        qualified = _sanitize_name(f"{server_name}_{tool.name}")
        self._seen_names |= tool_name_candidates(server_name, tool.name, qualified)
        if self.readonly and _is_write_tool(tool.name):
            logger.info("Skipping write tool in readonly mode: %s.%s", server_name, tool.name)
            self._hidden.add(qualified)
            return False
        if self.allowlist is not None and not (
            tool_name_candidates(server_name, tool.name, qualified) & self.allowlist
        ):
            logger.info("Hiding tool not in MCP_TOOL_ALLOWLIST: %s.%s", server_name, tool.name)
            self._hidden.add(qualified)
            return False
        if qualified in self._tools:
            raise ValueError(f"Duplicate MCP tool name after sanitizing: {qualified}")
        self._tools[qualified] = {
            "server_name": server_name,
            "tool_name": tool.name,
            "session": session,
            "description": tool.description or "",
            "input_schema": tool.inputSchema or {"type": "object", "properties": {}},
        }
        logger.info("Registered tool %s -> %s.%s", qualified, server_name, tool.name)
        return True

    def warn_unknown_allowlist_entries(self) -> list[str]:
        """Log (and return) allowlist entries that match no tool any server offered."""
        if self.allowlist is None:
            return []
        unknown = sorted(name for name in self.allowlist if name not in self._seen_names)
        for name in unknown:
            logger.warning(
                "MCP_TOOL_ALLOWLIST entry '%s' does not match any tool offered by the MCP servers",
                name,
            )
        return unknown

    def as_ollama_tools(self, *, trimmed: bool = True) -> list[dict[str, Any]]:
        """Convert exposed MCP tools into Ollama function-calling tool schemas.

        Descriptions are shortened per MCP_TOOL_DESC_MAX_CHARS / MCP_PARAM_DESC_MAX_CHARS
        unless ``trimmed=False`` (untrimmed originals, for logging/debugging).
        """
        tools: list[dict[str, Any]] = []
        for qualified, meta in self._tools.items():
            if trimmed:
                description = trim_description(meta["description"], self.desc_max_chars)
                parameters = trim_schema_descriptions(
                    meta["input_schema"], self.param_desc_max_chars
                )
            else:
                description = meta["description"]
                parameters = meta["input_schema"]
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": qualified,
                        "description": description,
                        "parameters": parameters,
                    },
                }
            )
        return tools

    async def call_tool(self, qualified_name: str, arguments: dict[str, Any] | None = None) -> str:
        """Invoke a registered MCP tool and return its text result (or an error string)."""
        meta = self._tools.get(qualified_name)
        if meta is None:
            if qualified_name in self._hidden:
                logger.warning("Rejected call to hidden tool %s", qualified_name)
                return (
                    f"Error: tool '{qualified_name}' is not available. "
                    f"Available tools: {', '.join(self.tool_names) or 'none'}"
                )
            return f"Error: unknown tool '{qualified_name}'"

        session: ClientSession = meta["session"]
        tool_name: str = meta["tool_name"]
        args = arguments or {}

        logger.info("Calling MCP tool %s with args=%s", qualified_name, args)
        result = await session.call_tool(tool_name, arguments=args)

        parts: list[str] = []
        for block in result.content or []:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
            else:
                parts.append(str(block))

        output = "\n".join(parts).strip() or "(empty tool result)"
        if getattr(result, "isError", False):
            return f"Tool error from {qualified_name}:\n{output}"
        return output

    async def close(self) -> None:
        """Tear down all MCP stdio sessions and clear registered tools."""
        await self._exit_stack.aclose()
        self._sessions.clear()
        self._tools.clear()
        self._hidden.clear()
        logger.info("MCP hub shut down")
