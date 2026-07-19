"""MCP client hub: connect to configured servers and call their tools."""

from __future__ import annotations

import json
import logging
import re
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .config import MCP_READONLY

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


def _sanitize_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", value)


def _is_write_tool(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in _WRITE_TOOL_MARKERS)


class McpHub:
    """Owns stdio MCP server subprocesses for the bot process lifetime."""

    def __init__(self, config_path: Path, readonly: bool | None = None) -> None:
        self.config_path = config_path
        self.readonly = MCP_READONLY if readonly is None else readonly
        self._exit_stack = AsyncExitStack()
        self._sessions: dict[str, ClientSession] = {}
        # qualified_name -> metadata for routing tool calls
        self._tools: dict[str, dict[str, Any]] = {}

    @property
    def tool_names(self) -> list[str]:
        return sorted(self._tools)

    async def start(self) -> None:
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

        logger.info(
            "MCP hub ready: %d server(s), %d tool(s)",
            len(self._sessions),
            len(self._tools),
        )

    async def _connect_server(self, server_name: str, server_cfg: dict[str, Any]) -> None:
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
            qualified = _sanitize_name(f"{server_name}_{tool.name}")
            if self.readonly and _is_write_tool(tool.name):
                logger.info("Skipping write tool in readonly mode: %s.%s", server_name, tool.name)
                continue
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

    def as_ollama_tools(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        for qualified, meta in self._tools.items():
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": qualified,
                        "description": meta["description"],
                        "parameters": meta["input_schema"],
                    },
                }
            )
        return tools

    async def call_tool(self, qualified_name: str, arguments: dict[str, Any] | None = None) -> str:
        meta = self._tools.get(qualified_name)
        if meta is None:
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
        await self._exit_stack.aclose()
        self._sessions.clear()
        self._tools.clear()
        logger.info("MCP hub shut down")
