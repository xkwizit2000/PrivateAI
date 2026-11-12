"""MCP client hub: connect to configured servers and call their tools."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import re
from collections.abc import Collection
from contextlib import AsyncExitStack
from pathlib import Path
from time import monotonic
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import McpError

from .config import (
    MCP_PARAM_DESC_MAX_CHARS,
    MCP_READONLY,
    MCP_TOOL_ALLOWLIST,
    MCP_TOOL_DESC_MAX_CHARS,
    MEMORY_MCP_SERVER,
    MEMORY_SSE_CONNECT_TIMEOUT,
    MEMORY_TOOL_TIMEOUT,
    MEMORY_TOOLS,
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


GROUP_FILES = "files"
GROUP_MEMORY = "memory"

_ENV_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

# After a failed (re)connect an SSE server is not retried for this long (seconds), so a
# dead memory service costs a tool call nothing but an immediate error string.
SSE_RETRY_COOLDOWN_SECONDS = 30.0


def expand_env(value: str) -> str:
    """Replace ``${VAR}`` with the environment value (unset = empty) so secrets stay out of mcp.json."""
    return _ENV_REF_RE.sub(lambda m: os.environ.get(m.group(1), ""), value)


class _SseConnection:
    """One MCP-over-SSE connection owned by a dedicated task.

    The anyio contexts inside ``sse_client`` / ``ClientSession`` must be exited by the
    task that entered them, so the connection lives in its own task: ``open()`` waits
    until it is initialised, ``close()`` asks the task to leave the contexts itself.
    """

    def __init__(self, name: str, url: str, headers: dict[str, str] | None) -> None:
        """Remember how to connect; nothing is opened yet."""
        self.name = name
        self.url = url
        self.headers = headers or None
        self.session: ClientSession | None = None
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    async def open(self, timeout: float) -> ClientSession:
        """Connect + initialize, or raise (TimeoutError/OSError/...) within ``timeout``."""
        ready: asyncio.Future[ClientSession] = asyncio.get_running_loop().create_future()
        self._stop = asyncio.Event()
        self._task = asyncio.create_task(self._run(ready), name=f"mcp-sse-{self.name}")
        try:
            self.session = await asyncio.wait_for(ready, timeout=timeout)
        except BaseException:
            await self.close(force=True)
            raise
        return self.session

    async def _run(self, ready: asyncio.Future[ClientSession]) -> None:
        """Task body: hold the SSE transport + session open until asked to stop."""
        try:
            async with sse_client(
                self.url, headers=self.headers, timeout=5, sse_read_timeout=300
            ) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    if not ready.done():
                        ready.set_result(session)
                    await self._stop.wait()
        except BaseException as exc:  # noqa: BLE001 - reported through the future / dead flag
            if not ready.done():
                ready.set_exception(exc if isinstance(exc, Exception) else RuntimeError(repr(exc)))
        finally:
            self.session = None
            if not ready.done():
                ready.set_exception(ConnectionError("SSE connection closed"))
            if not ready.cancelled():
                ready.exception()  # mark retrieved: no "never retrieved" warning after a timeout

    @property
    def alive(self) -> bool:
        """True while the owner task is running and the session is usable."""
        return self._task is not None and not self._task.done() and self.session is not None

    async def close(self, force: bool = False) -> None:
        """Leave the transport contexts (from the owning task); never raises.

        ``force`` cancels the owner task right away (used when a connect attempt
        timed out and the task is stuck inside the handshake).
        """
        task, self._task = self._task, None
        self.session = None
        if task is None:
            return
        self._stop.set()
        try:
            if force:
                raise asyncio.CancelledError
            await asyncio.wait_for(asyncio.shield(task), timeout=3)
        except BaseException:  # noqa: BLE001 - dead transport, best effort
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=2)
            except BaseException:  # noqa: BLE001
                pass


def _exc_name(exc: BaseException) -> str:
    """Exception type name, looking through anyio/ExceptionGroup wrappers (for log lines)."""
    inner = getattr(exc, "exceptions", None)
    while inner:
        exc = inner[0]
        inner = getattr(exc, "exceptions", None)
    return type(exc).__name__


def server_group(server_name: str) -> str:
    """Tool group of an MCP server: ``memory`` for the memory server, else ``files``."""
    return GROUP_MEMORY if server_name == MEMORY_MCP_SERVER else GROUP_FILES


class McpHub:
    """Owns stdio MCP server subprocesses and SSE connections for the bot process lifetime."""

    def __init__(
        self,
        config_path: Path,
        readonly: bool | None = None,
        *,
        allowlist: str | None = None,
        desc_max_chars: int | None = None,
        param_desc_max_chars: int | None = None,
        memory_tools: bool | None = None,
        call_timeout: float | None = None,
        connect_timeout: float | None = None,
    ) -> None:
        """Load MCP config path plus readonly, allowlist and description-trim settings.

        ``allowlist`` is the raw MCP_TOOL_ALLOWLIST string (None = read from config).
        ``memory_tools`` (default: MEMORY_TOOLS=on) decides whether the memory server's
        tools are registered at all; the memory group is governed only by that switch
        (it is exempt from MCP_TOOL_ALLOWLIST / MCP_READONLY, which are about files).
        ``call_timeout`` / ``connect_timeout`` bound SSE tool calls and (re)connects.
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
        self.memory_tools = (MEMORY_TOOLS == "on") if memory_tools is None else memory_tools
        self.call_timeout = MEMORY_TOOL_TIMEOUT if call_timeout is None else call_timeout
        self.connect_timeout = (
            MEMORY_SSE_CONNECT_TIMEOUT if connect_timeout is None else connect_timeout
        )
        self._exit_stack = AsyncExitStack()
        self._sessions: dict[str, ClientSession] = {}
        # SSE servers: per-server config, connection, reconnect lock and retry-after time.
        self._sse_cfg: dict[str, dict[str, Any]] = {}
        self._sse_conn: dict[str, _SseConnection] = {}
        self._sse_locks: dict[str, asyncio.Lock] = {}
        self._sse_down_until: dict[str, float] = {}
        self._bg_tasks: set[asyncio.Task[None]] = set()
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
            if server_cfg.get("url"):
                # Remote (SSE) servers are optional add-ons: never let one block startup.
                try:
                    await self._connect_server(server_name, server_cfg)
                except Exception as exc:  # noqa: BLE001
                    self._sse_down_until[server_name] = monotonic() + SSE_RETRY_COOLDOWN_SECONDS
                    logger.warning(
                        "MCP SSE server '%s' unavailable at startup (%s); continuing without it",
                        server_name,
                        _exc_name(exc),
                    )
                continue
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
        """Connect one MCP server (SSE if it has a ``url``, else spawn stdio) and register tools."""
        if server_cfg.get("url"):
            await self._connect_sse_server(server_name, server_cfg)
            return
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

    async def _connect_sse_server(self, server_name: str, server_cfg: dict[str, Any]) -> None:
        """Connect an MCP-over-SSE server (``{"url": ..., "headers": {...}}``) and register its tools."""
        if server_group(server_name) == GROUP_MEMORY and not self.memory_tools:
            logger.info(
                "MCP server '%s' is configured but MEMORY_TOOLS=off; not connecting", server_name
            )
            return
        headers = {
            str(key): expand_env(str(value))
            for key, value in (server_cfg.get("headers") or {}).items()
        }
        self._sse_cfg[server_name] = {
            "url": expand_env(str(server_cfg["url"])),
            "headers": headers,
            "timeout": float(server_cfg.get("timeout", self.call_timeout)),
        }
        logger.info("Connecting MCP SSE server '%s'", server_name)
        await self._open_sse(server_name)

    async def _open_sse(self, server_name: str) -> None:
        """(Re)open the SSE connection for a server and (re)register its tool sessions."""
        cfg = self._sse_cfg[server_name]
        old = self._sse_conn.pop(server_name, None)
        if old is not None:
            await old.close()
        conn = _SseConnection(server_name, cfg["url"], cfg["headers"])
        session = await conn.open(self.connect_timeout)
        try:
            listed = await asyncio.wait_for(session.list_tools(), timeout=self.connect_timeout)
        except BaseException:
            await conn.close()
            raise
        self._sse_conn[server_name] = conn
        self._sessions[server_name] = session
        for tool in listed.tools:
            qualified = _sanitize_name(f"{server_name}_{tool.name}")
            if qualified in self._tools:  # reconnect: only refresh the session handle
                self._tools[qualified]["session"] = session
            else:
                self._register_tool(server_name, session, tool)

    def tool_group(self, qualified_name: str) -> str | None:
        """Group (``files``/``memory``) of a registered tool, or None if unknown."""
        meta = self._tools.get(qualified_name)
        return None if meta is None else meta["group"]

    def _register_tool(self, server_name: str, session: Any, tool: Any) -> bool:
        """Register one server tool unless readonly/allowlist hides it; True if exposed."""
        qualified = _sanitize_name(f"{server_name}_{tool.name}")
        self._seen_names |= tool_name_candidates(server_name, tool.name, qualified)
        group = server_group(server_name)
        if group == GROUP_MEMORY:
            # Memory tools: only MEMORY_TOOLS decides (not the file allowlist/readonly lists).
            if not self.memory_tools:
                logger.info("Hiding memory tool (MEMORY_TOOLS=off): %s.%s", server_name, tool.name)
                self._hidden.add(qualified)
                return False
        elif self.readonly and _is_write_tool(tool.name):
            logger.info("Skipping write tool in readonly mode: %s.%s", server_name, tool.name)
            self._hidden.add(qualified)
            return False
        if (
            group != GROUP_MEMORY
            and self.allowlist is not None
            and not (tool_name_candidates(server_name, tool.name, qualified) & self.allowlist)
        ):
            logger.info("Hiding tool not in MCP_TOOL_ALLOWLIST: %s.%s", server_name, tool.name)
            self._hidden.add(qualified)
            return False
        if qualified in self._tools:
            raise ValueError(f"Duplicate MCP tool name after sanitizing: {qualified}")
        self._tools[qualified] = {
            "server_name": server_name,
            "group": group,
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

    def as_ollama_tools(
        self, *, trimmed: bool = True, groups: Collection[str] | None = None
    ) -> list[dict[str, Any]]:
        """Convert exposed MCP tools into Ollama function-calling tool schemas.

        Descriptions are shortened per MCP_TOOL_DESC_MAX_CHARS / MCP_PARAM_DESC_MAX_CHARS
        unless ``trimmed=False`` (untrimmed originals, for logging/debugging).
        ``groups`` (e.g. ``{"files"}``, ``{"memory"}``) restricts the result to tools of
        those groups; None means every registered tool.
        """
        tools: list[dict[str, Any]] = []
        for qualified, meta in self._tools.items():
            if groups is not None and meta["group"] not in groups:
                continue
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
        """Invoke a registered MCP tool and return its text result (or an error string).

        stdio tools behave as before. Tools of SSE servers are bounded by a per-call
        timeout, get one reconnect attempt, and never raise: a dead server yields an
        error string for the model instead of an exception out of the agent loop.
        """
        meta = self._tools.get(qualified_name)
        if meta is None:
            if qualified_name in self._hidden:
                logger.warning("Rejected call to hidden tool %s", qualified_name)
                return (
                    f"Error: tool '{qualified_name}' is not available. "
                    f"Available tools: {', '.join(self.tool_names) or 'none'}"
                )
            return f"Error: unknown tool '{qualified_name}'"

        tool_name: str = meta["tool_name"]
        args = arguments or {}

        if meta["group"] == GROUP_MEMORY:
            # Arguments of memory tools are user conversation text: log the keys only.
            logger.info("Calling MCP tool %s with arg keys=%s", qualified_name, sorted(args))
        else:
            logger.info("Calling MCP tool %s with args=%s", qualified_name, args)

        server_name: str = meta["server_name"]
        if server_name in self._sse_cfg:
            return await self._call_sse_tool(qualified_name, server_name, tool_name, args)

        session: ClientSession = meta["session"]
        result = await session.call_tool(tool_name, arguments=args)
        return self._format_result(qualified_name, result)

    @staticmethod
    def _format_result(qualified_name: str, result: Any) -> str:
        """Join a tool result's content blocks into text (prefixing tool-reported errors)."""
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

    async def _call_sse_tool(
        self, qualified_name: str, server_name: str, tool_name: str, args: dict[str, Any]
    ) -> str:
        """Call a tool on an SSE server with timeout + one reconnect; always returns a string."""
        timeout = self._sse_cfg[server_name]["timeout"]
        unavailable = f"Tool error: {server_name} service unavailable"
        if monotonic() < self._sse_down_until.get(server_name, 0.0):
            return unavailable
        session = self._tools[qualified_name]["session"]
        for attempt in (1, 2):
            try:
                result = await asyncio.wait_for(
                    session.call_tool(tool_name, arguments=args), timeout=timeout
                )
                return self._format_result(qualified_name, result)
            except McpError as exc:
                # The server answered with a protocol-level error: the transport is fine.
                return f"Tool error from {qualified_name}: {str(exc)[:200]}"
            except Exception as exc:  # noqa: BLE001 - includes timeouts and transport errors
                logger.warning(
                    "MCP SSE tool %s failed (%s) attempt %d/2",
                    qualified_name,
                    _exc_name(exc),
                    attempt,
                )
                if attempt == 2:
                    break
                reconnected = await self._reconnect_sse(server_name, session)
                if reconnected is None:
                    break
                session = reconnected
        return unavailable

    async def _reconnect_sse(
        self, server_name: str, failed_session: ClientSession
    ) -> ClientSession | None:
        """Reconnect once (shared by concurrent callers); None if it failed."""
        lock = self._sse_locks.setdefault(server_name, asyncio.Lock())
        async with lock:
            current = self._sessions.get(server_name)
            if current is not None and current is not failed_session:
                return current  # somebody else already reconnected
            try:
                await self._open_sse(server_name)
            except Exception as exc:  # noqa: BLE001
                self._sse_down_until[server_name] = monotonic() + SSE_RETRY_COOLDOWN_SECONDS
                logger.warning(
                    "MCP SSE server '%s' reconnect failed (%s); not retrying for %.0fs",
                    server_name,
                    _exc_name(exc),
                    SSE_RETRY_COOLDOWN_SECONDS,
                )
                return None
            self._sse_down_until.pop(server_name, None)
            logger.info("MCP SSE server '%s' reconnected", server_name)
            return self._sessions.get(server_name)

    def kick_reconnect(self) -> None:
        """Start a background reconnect for down SSE servers (never waits; rate-limited)."""
        for server_name in list(self._sse_cfg):
            conn = self._sse_conn.get(server_name)
            if conn is not None and conn.alive:
                continue
            lock = self._sse_locks.setdefault(server_name, asyncio.Lock())
            if lock.locked() or monotonic() < self._sse_down_until.get(server_name, 0.0):
                continue
            self._sse_down_until[server_name] = monotonic() + SSE_RETRY_COOLDOWN_SECONDS
            task = asyncio.create_task(self._background_reconnect(server_name))
            self._bg_tasks.add(task)
            task.add_done_callback(self._bg_tasks.discard)

    async def _background_reconnect(self, server_name: str) -> None:
        """Try to (re)connect a down SSE server; failure only logs a warning."""
        lock = self._sse_locks.setdefault(server_name, asyncio.Lock())
        async with lock:
            try:
                await self._open_sse(server_name)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "MCP SSE server '%s' still unavailable (%s)", server_name, _exc_name(exc)
                )
                return
            self._sse_down_until.pop(server_name, None)
            logger.info("MCP SSE server '%s' connected", server_name)

    async def close(self) -> None:
        """Tear down all MCP stdio sessions / SSE connections and clear registered tools."""
        for task in list(self._bg_tasks):
            task.cancel()
        for conn in list(self._sse_conn.values()):
            await conn.close()
        self._sse_conn.clear()
        self._sse_cfg.clear()
        await self._exit_stack.aclose()
        self._sessions.clear()
        self._tools.clear()
        self._hidden.clear()
        logger.info("MCP hub shut down")
