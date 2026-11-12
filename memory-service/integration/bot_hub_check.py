"""Verify the memory service from the *bot's own* MCP stack (mcp==1.3.0 client + src.mcp_client.McpHub).

Run inside the bot image (does not touch the live container; it is a separate `docker run`):

  docker run --rm --network memnet --entrypoint "" \
     -v $PWD/integration:/it:ro -e MEM_URL=http://memsvc:8765 -e MEM_TOKEN=... \
     privateai:latest python /it/bot_hub_check.py [--idle-seconds 330] [--restart-cmd ...]

`SseCapableHub` below is the proposed change to src/mcp_client.py (url transport +
failure-tolerant call_tool + reconnect). It subclasses McpHub so nothing under src/ is modified.
"""

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, "/app")

from mcp import ClientSession  # noqa: E402
from mcp.client.sse import sse_client  # noqa: E402

from src.mcp_client import McpHub  # noqa: E402


CALL_TIMEOUT = 4.0  # memory calls normally take <100 ms; fail fast so a dead service can't stall a chat turn


class SseCapableHub(McpHub):
    """McpHub + `{"url": ..., "headers": {...}}` servers over SSE."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._sse_cfg: dict[str, dict] = {}
        self._sse_stacks: dict[str, object] = {}

    async def _connect_server(self, server_name, server_cfg):
        if not server_cfg.get("url"):
            return await super()._connect_server(server_name, server_cfg)
        self._sse_cfg[server_name] = server_cfg
        await self._connect_sse(server_name)

    async def _connect_sse(self, server_name):
        from contextlib import AsyncExitStack

        cfg = self._sse_cfg[server_name]
        old = self._sse_stacks.pop(server_name, None)
        if old is not None:
            try:
                await old.aclose()
            except BaseException:  # noqa: BLE001 - dead transport, best effort
                pass
        stack = AsyncExitStack()
        read, write = await stack.enter_async_context(
            sse_client(cfg["url"], headers=cfg.get("headers"), timeout=5, sse_read_timeout=300)
        )
        session = await stack.enter_async_context(ClientSession(read, write))
        await asyncio.wait_for(session.initialize(), timeout=10)
        self._sse_stacks[server_name] = stack
        self._sessions[server_name] = session
        for tool in (await session.list_tools()).tools:
            if f"{server_name}_{tool.name}" in self._tools:  # reconnect: refresh session handle only
                self._tools[f"{server_name}_{tool.name}"]["session"] = session
            else:
                self._register_tool(server_name, session, tool)

    async def close(self):
        for stack in list(self._sse_stacks.values()):
            try:
                await stack.aclose()
            except BaseException:  # noqa: BLE001
                pass
        self._sse_stacks.clear()
        await super().close()

    async def call_tool(self, qualified_name, arguments=None):
        meta = self._tools.get(qualified_name)
        if meta is None or meta["server_name"] not in self._sse_cfg:
            return await super().call_tool(qualified_name, arguments)
        for attempt in (1, 2):
            try:
                return await asyncio.wait_for(super().call_tool(qualified_name, arguments), timeout=CALL_TIMEOUT)
            except BaseException as exc:  # noqa: BLE001
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                if attempt == 1:
                    try:
                        await self._connect_sse(meta["server_name"])
                        continue
                    except BaseException:  # noqa: BLE001
                        pass
                return f"Tool error: memory service unavailable ({type(exc).__name__})"


def ok(name, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name} {extra}")
    return cond


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--idle-seconds", type=float, default=0)
    ap.add_argument("--wait-restart", action="store_true", help="pause so the caller can restart the service")
    args = ap.parse_args()
    url = os.environ["MEM_URL"].rstrip("/")
    token = os.environ.get("MEM_TOKEN", "")
    cfg = {"mcpServers": {"memory": {"url": url + "/sse", **({"headers": {"Authorization": f"Bearer {token}"}} if token else {})}}}
    path = Path(tempfile.mkdtemp()) / "mcp.json"
    path.write_text(json.dumps(cfg))
    import importlib.metadata as md
    print("client mcp version:", md.version("mcp"))

    results = []
    hub = SseCapableHub(path, allowlist="search_memory,get_conversation,remember_fact,list_facts,memory_stats")
    await hub.start()
    results.append(ok("tools listed", hub.tool_names == sorted(f"memory_{n}" for n in
                    ("search_memory", "get_conversation", "remember_fact", "list_facts", "memory_stats")), str(hub.tool_names)))
    trimmed = len(json.dumps(hub.as_ollama_tools()))
    full = len(json.dumps(hub.as_ollama_tools(trimmed=False)))
    results.append(ok("schema size < 2000 chars (as sent to Ollama, after the bot's own trimming)", trimmed < 2000,
                      f"sent={trimmed} untrimmed={full} chars"))
    print(json.dumps(hub.as_ollama_tools(), separators=(",", ":"))[:2400])

    out = await hub.call_tool("memory_memory_stats", {})
    results.append(ok("memory_stats", "turns=" in out, out[:120]))
    out = await hub.call_tool("memory_search_memory", {"query": "anything about the test", "k": 3})
    results.append(ok("search_memory returns text within cap", 0 < len(out) <= 1800, f"{len(out)} chars"))
    out = await hub.call_tool("memory_remember_fact", {"statement": "integration-test fact: prefers metric units", "category": "preference"})
    results.append(ok("remember_fact", out.startswith(("saved", "already")), out))
    out = await hub.call_tool("memory_list_facts", {})
    results.append(ok("list_facts shows fact", "metric units" in out, out[:100]))
    out = await hub.call_tool("memory_get_conversation", {"around_turn_id": 1, "window": 1})
    results.append(ok("get_conversation", out.startswith("[t") or out.startswith("No turn"), out[:80]))
    out = await hub.call_tool("memory_search_memory", {"query": "x", "k": 99, "since": "7d", "bogus_arg": 1})
    results.append(ok("bad/extra args handled without crashing", isinstance(out, str), out[:80]))

    if args.idle_seconds:
        print(f"idling {args.idle_seconds:.0f}s to test SSE keep-alive across the client's 300s read timeout ...")
        await asyncio.sleep(args.idle_seconds)
        t0 = time.perf_counter()
        out = await hub.call_tool("memory_memory_stats", {})
        results.append(ok("call after idle works (no reconnect needed)", "turns=" in out, f"{(time.perf_counter()-t0)*1000:.0f} ms"))

    if args.wait_restart:
        print("RESTART_NOW", flush=True)
        await asyncio.sleep(25)  # caller restarts the service container during this window
        t0 = time.perf_counter()
        out = await hub.call_tool("memory_memory_stats", {})
        results.append(ok("call after service restart (reconnect or graceful error)", "turns=" in out or out.startswith("Tool error"),
                          f"{(time.perf_counter()-t0):.1f}s -> {out[:70]}"))
        out = await hub.call_tool("memory_memory_stats", {})
        results.append(ok("subsequent call works after reconnect", "turns=" in out, out[:60]))
    await hub.close()
    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
