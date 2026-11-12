"""Real MCP-over-SSE round trip using the mcp 1.3.0 *client* (the version the bot pins)."""

import asyncio
import importlib.metadata
import json
import socket
import threading
import time

import pytest

pytest.importorskip("mcp")
uvicorn = pytest.importorskip("uvicorn")

from mcp import ClientSession  # noqa: E402
from mcp.client.sse import sse_client  # noqa: E402

from memory_service.http_api import build_app  # noqa: E402
from memory_service.mcp_server import build_mcp  # noqa: E402


@pytest.fixture()
def server(svc):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    app = build_app(svc, build_mcp(svc), auth_token="tok")
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    th.join(timeout=5)


def test_client_version_is_bots():
    assert importlib.metadata.version("mcp") == "1.3.0"


def test_sse_roundtrip(server):
    async def go():
        async with sse_client(server + "/sse", headers={"Authorization": "Bearer tok"}) as (r, w):
            async with ClientSession(r, w) as session:
                await session.initialize()
                tools = (await session.list_tools()).tools
                names = {t.name for t in tools}
                assert names == {"search_memory", "get_conversation", "remember_fact", "list_facts", "memory_stats"}
                size = len(json.dumps([{"name": t.name, "description": t.description, "inputSchema": t.inputSchema} for t in tools]))
                assert size < 2000, size
                res = await session.call_tool("search_memory", {"query": "Biscuit cat", "k": 2})
                assert not res.isError and "Biscuit" in res.content[0].text
                res = await session.call_tool("remember_fact", {"statement": "Uses a Ryzen mini PC", "category": "profile"})
                assert res.content[0].text.startswith("saved fact#")
                res = await session.call_tool("memory_stats", {})
                assert "turns=14" in res.content[0].text
                res = await session.call_tool("get_conversation", {"date": "2026-03-02"})
                assert "restic" in res.content[0].text

    asyncio.run(go())


def test_sse_requires_token(server):
    async def go():
        with pytest.raises(BaseException):
            async with sse_client(server + "/sse") as (r, w):
                async with ClientSession(r, w) as session:
                    await asyncio.wait_for(session.initialize(), 5)

    asyncio.run(go())
