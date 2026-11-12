"""Tests for the MCP-over-SSE transport in McpHub (memory tool group)."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.mcp_client import McpHub, expand_env
from tests.sse_fake_server import TOKEN, FakeSseServer, free_port

ROOT = Path(__file__).resolve().parent


def _write_config(servers: dict) -> Path:
    path = Path(tempfile.mkdtemp()) / "mcp.json"
    path.write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")
    return path


def _memory_cfg(url: str, token: str = TOKEN, **extra) -> dict:
    return {"memory": {"url": url, "headers": {"Authorization": f"Bearer {token}"}, **extra}}


def _hub(path: Path, **kwargs) -> McpHub:
    kwargs.setdefault("memory_tools", True)
    kwargs.setdefault("call_timeout", 2.0)
    kwargs.setdefault("connect_timeout", 3.0)
    return McpHub(path, allowlist="list_directory", **kwargs)


class ExpandEnvTests(unittest.TestCase):
    def test_expands_known_and_unknown(self):
        with patch.dict(os.environ, {"MEMORY_TOKEN": "abc"}, clear=False):
            self.assertEqual(expand_env("Bearer ${MEMORY_TOKEN}"), "Bearer abc")
            self.assertEqual(expand_env("${DEFINITELY_UNSET_VAR_X}"), "")
            self.assertEqual(expand_env("plain"), "plain")


class SseTransportTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.server = FakeSseServer()
        self.server.start()
        self.addCleanup(self.server.stop)

    async def _started_hub(self, **kwargs) -> McpHub:
        hub = _hub(_write_config(_memory_cfg(self.server.url)), **kwargs)
        await hub.start()
        self.addAsyncCleanup(hub.close)
        return hub

    async def test_registers_tools_and_calls_them(self):
        hub = await self._started_hub()
        self.assertIn("memory_search_memory", hub.tool_names)
        self.assertEqual(hub.tool_group("memory_search_memory"), "memory")
        out = await hub.call_tool("memory_search_memory", {"query": "hello", "k": 2})
        self.assertIn("canned hit for 5 chars (k=2)", out)
        self.assertEqual(await hub.call_tool("memory_memory_stats", {}), "turns=14 chunks=14 sessions=1")

    async def test_token_expanded_from_env_in_headers(self):
        cfg = {"memory": {"url": self.server.url, "headers": {"Authorization": "Bearer ${MEMORY_TOKEN}"}}}
        with patch.dict(os.environ, {"MEMORY_TOKEN": TOKEN}):
            hub = _hub(_write_config(cfg))
            await hub.start()
        self.addAsyncCleanup(hub.close)
        self.assertIn("memory_memory_stats", hub.tool_names)

    async def test_wrong_token_does_not_prevent_startup(self):
        hub = _hub(_write_config(_memory_cfg(self.server.url, token="nope")), connect_timeout=1.5)
        with self.assertLogs("src.mcp_client", level="WARNING") as logs:
            await hub.start()
        await hub.close()
        self.assertEqual(hub.tool_names, [])
        self.assertTrue(any("unavailable at startup" in line for line in logs.output))
        self.assertFalse(any("nope" in line for line in logs.output))

    async def test_tool_error_is_a_string_not_an_exception(self):
        hub = await self._started_hub()
        out = await hub.call_tool("memory_broken_tool", {})
        self.assertIsInstance(out, str)
        self.assertIn("Tool error", out)
        out = await hub.call_tool("memory_search_memory", {"wrong": 1})
        self.assertIsInstance(out, str)

    async def test_call_timeout_returns_error_string_quickly(self):
        hub = await self._started_hub(call_timeout=0.4)
        started = time.monotonic()
        out = await hub.call_tool("memory_slow_tool", {"seconds": 30})
        self.assertLess(time.monotonic() - started, 4.0)
        self.assertEqual(out, "Tool error: memory service unavailable")
        # Still healthy afterwards.
        self.assertIn("turns=", await hub.call_tool("memory_memory_stats", {}))

    async def test_reconnects_once_after_server_restart(self):
        hub = await self._started_hub(call_timeout=1.0)
        self.assertIn("turns=", await hub.call_tool("memory_memory_stats", {}))
        self.server.stop()
        self.server.start()
        out = await hub.call_tool("memory_memory_stats", {})
        self.assertIn("turns=", out)  # first attempt fails, reconnect, second attempt works
        self.assertIn("turns=", await hub.call_tool("memory_memory_stats", {}))

    async def test_dead_server_gives_error_then_fast_failures(self):
        hub = await self._started_hub(call_timeout=0.5, connect_timeout=1.0)
        self.server.stop()
        with self.assertLogs("src.mcp_client", level="WARNING"):
            out = await hub.call_tool("memory_memory_stats", {})
        self.assertEqual(out, "Tool error: memory service unavailable")
        started = time.monotonic()
        out = await hub.call_tool("memory_memory_stats", {})  # cooldown: no new attempt
        self.assertLess(time.monotonic() - started, 0.2)
        self.assertEqual(out, "Tool error: memory service unavailable")

    async def test_memory_tools_off_does_not_connect_or_expose(self):
        hub = _hub(_write_config(_memory_cfg(self.server.url)), memory_tools=False)
        await hub.start()
        self.addAsyncCleanup(hub.close)
        self.assertEqual(hub.tool_names, [])
        self.assertEqual(hub._sse_cfg, {})


class SseFailureIsolationTests(unittest.IsolatedAsyncioTestCase):
    async def test_unreachable_server_never_breaks_startup_or_file_tools(self):
        port = free_port()  # nothing listens here
        servers = {
            "filesystem": {"command": sys.executable, "args": [str(ROOT / "stdio_fake_server.py")]},
            **_memory_cfg(f"http://127.0.0.1:{port}/sse"),
        }
        hub = _hub(_write_config(servers), connect_timeout=1.5)
        try:
            with self.assertLogs("src.mcp_client", level="WARNING") as logs:
                await hub.start()
            self.assertTrue(any("unavailable at startup" in line for line in logs.output))
            self.assertEqual(hub.tool_names, ["filesystem_list_directory"])
            out = await hub.call_tool("filesystem_list_directory", {"path": "/x"})
            self.assertIn("a.txt", out)
        finally:
            await hub.close()  # same task as start(): anyio's stdio scope requires it

    async def test_black_hole_server_is_bounded_by_connect_timeout(self):
        with socket.socket() as blackhole:
            blackhole.bind(("127.0.0.1", 0))
            blackhole.listen(5)  # accepts TCP, never answers HTTP
            url = f"http://127.0.0.1:{blackhole.getsockname()[1]}/sse"
            hub = _hub(_write_config(_memory_cfg(url)), connect_timeout=0.7)
            started = time.monotonic()
            await hub.start()
            elapsed = time.monotonic() - started
            await hub.close()
        self.assertLess(elapsed, 3.0)
        self.assertEqual(hub.tool_names, [])

    async def test_background_reconnect_registers_tools_when_service_returns(self):
        server = FakeSseServer()
        hub = _hub(_write_config(_memory_cfg(server.url)), connect_timeout=1.0)
        await hub.start()  # server is not running yet
        self.addAsyncCleanup(hub.close)
        self.assertEqual(hub.tool_names, [])
        server.start()
        self.addCleanup(server.stop)
        hub._sse_down_until.clear()  # skip the cooldown instead of sleeping in the test
        hub.kick_reconnect()
        for _ in range(60):
            if hub.tool_names:
                break
            await asyncio.sleep(0.1)
        self.assertIn("memory_memory_stats", hub.tool_names)
        self.assertIn("turns=", await hub.call_tool("memory_memory_stats", {}))


class GroupFilteringTests(unittest.TestCase):
    def _tool(self, name):
        return SimpleNamespace(
            name=name, description="d", inputSchema={"type": "object", "properties": {}}
        )

    def _hub(self, **kwargs):
        hub = McpHub(Path("/nonexistent"), allowlist="list_directory", memory_tools=True, **kwargs)
        hub._register_tool("filesystem", MagicMock(), self._tool("list_directory"))
        hub._register_tool("filesystem", MagicMock(), self._tool("write_file"))  # not allowlisted
        hub._register_tool("memory", MagicMock(), self._tool("search_memory"))
        hub._register_tool("memory", MagicMock(), self._tool("remember_fact"))
        return hub

    @staticmethod
    def _names(tools):
        return sorted(t["function"]["name"] for t in tools)

    def test_groups_are_independent(self):
        hub = self._hub()
        self.assertEqual(
            self._names(hub.as_ollama_tools(groups={"files"})), ["filesystem_list_directory"]
        )
        self.assertEqual(
            self._names(hub.as_ollama_tools(groups={"memory"})),
            ["memory_remember_fact", "memory_search_memory"],
        )
        self.assertEqual(len(hub.as_ollama_tools()), 3)  # no filter = everything exposed
        self.assertEqual(hub.as_ollama_tools(groups=set()), [])

    def test_memory_group_ignores_file_allowlist_and_readonly(self):
        hub = McpHub(
            Path("/nonexistent"), readonly=True, allowlist="list_directory", memory_tools=True
        )
        self.assertTrue(hub._register_tool("memory", MagicMock(), self._tool("remember_fact")))
        self.assertFalse(hub._register_tool("filesystem", MagicMock(), self._tool("write_file")))

    def test_memory_tools_off_hides_memory_group(self):
        hub = McpHub(Path("/nonexistent"), allowlist="list_directory", memory_tools=False)
        self.assertFalse(hub._register_tool("memory", MagicMock(), self._tool("search_memory")))
        self.assertTrue(hub._register_tool("filesystem", MagicMock(), self._tool("list_directory")))
        self.assertEqual(hub.tool_names, ["filesystem_list_directory"])


class MemoryArgsNotLoggedTests(unittest.IsolatedAsyncioTestCase):
    async def test_memory_tool_call_logs_keys_not_values(self):
        server = FakeSseServer()
        server.start()
        self.addCleanup(server.stop)
        hub = _hub(_write_config(_memory_cfg(server.url)))
        await hub.start()
        self.addAsyncCleanup(hub.close)
        with self.assertLogs("src.mcp_client", level="INFO") as logs:
            await hub.call_tool("memory_search_memory", {"query": "SECRET-PRIVATE-TEXT"})
        joined = "\n".join(logs.output)
        self.assertIn("arg keys=['query']", joined)
        self.assertNotIn("SECRET-PRIVATE-TEXT", joined)


if __name__ == "__main__":
    unittest.main()
