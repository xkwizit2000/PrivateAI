"""A tiny in-process MCP-over-SSE server (mcp 1.3.0 FastMCP) for the bot's SSE tests.

It mimics the memory service's tool surface (names/arguments) with canned answers,
requires ``Authorization: Bearer <token>`` and can be stopped/restarted on the same
port to exercise the client's reconnect logic. No real conversation data is involved.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time

import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse, Response
from starlette.routing import Mount, Route

TOKEN = "test-token"


def free_port() -> int:
    """Return an unused localhost TCP port."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def build_app(token: str = TOKEN) -> Starlette:
    """Starlette app serving a FastMCP 'memory' server at /sse (+ /messages/)."""
    mcp = FastMCP("memory", log_level="WARNING")

    @mcp.tool(description="Search past conversations (keyword+semantic). Returns dated snippets with turn ids.")
    async def search_memory(query: str, k: int = 5) -> str:
        """query: what to look for."""
        return f"[t1 2026-01-01 user] canned hit for {len(query)} chars (k={k})"

    @mcp.tool(description="Memory size/coverage summary.")
    async def memory_stats() -> str:
        return "turns=14 chunks=14 sessions=1"

    @mcp.tool(description="Save a durable fact.")
    async def remember_fact(statement: str, category: str = "note") -> str:
        return "saved fact#1"

    @mcp.tool(description="Sleeps for a while (to trigger client timeouts).")
    async def slow_tool(seconds: float = 30) -> str:
        await asyncio.sleep(seconds)
        return "late"

    @mcp.tool(description="Always fails.")
    async def broken_tool() -> str:
        raise RuntimeError("boom")

    sse = SseServerTransport("/messages/")
    server = mcp._mcp_server

    async def handle_sse(request):
        header = request.headers.get("authorization", "")
        if token and header != f"Bearer {token}":
            return PlainTextResponse("unauthorized", status_code=401)
        async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
            await server.run(streams[0], streams[1], server.create_initialization_options())
        return Response()

    return Starlette(
        routes=[Route("/sse", endpoint=handle_sse), Mount("/messages/", app=sse.handle_post_message)]
    )


class FakeSseServer:
    """Run/stop/restart the fake server in a background thread on a fixed port."""

    def __init__(self, token: str = TOKEN, port: int | None = None) -> None:
        """Pick a port (kept across restarts)."""
        self.token = token
        self.port = port or free_port()
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        """SSE endpoint URL."""
        return f"http://127.0.0.1:{self.port}/sse"

    def start(self) -> None:
        """Start serving (blocks until accepting connections)."""
        config = uvicorn.Config(
            build_app(self.token), host="127.0.0.1", port=self.port, log_level="critical"
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        for _ in range(200):
            if self._server.started:
                return
            time.sleep(0.025)
        raise RuntimeError("fake SSE server did not start")

    def stop(self) -> None:
        """Stop serving (open SSE streams are dropped)."""
        if self._server is not None:
            self._server.should_exit = True
            self._server.force_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._server = self._thread = None
