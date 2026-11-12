"""MCP server (FastMCP from mcp==1.3.0, the same version the bot's client uses).

Tool schemas are deliberately tiny: short names/descriptions, no titles, no
anyOf/null unions, and hard output-size caps enforced in MemoryService.
Empty string / 0 mean "not set" so every parameter is a plain scalar.
"""

import anyio
from mcp.server.fastmcp import FastMCP

from .service import MemoryService


def _compact(node):
    """Strip schema noise (titles, null defaults, Optional anyOf) in place."""
    if isinstance(node, dict):
        node.pop("title", None)
        if node.get("default") is None and "default" in node:
            node.pop("default")
        any_of = node.get("anyOf")
        if any_of:
            non_null = [x for x in any_of if x.get("type") != "null"]
            if len(non_null) == 1:
                node.pop("anyOf")
                node.update(non_null[0])
        for v in list(node.values()):
            _compact(v)
    elif isinstance(node, list):
        for v in node:
            _compact(v)
    return node


def build_mcp(svc: MemoryService, host: str = "0.0.0.0", port: int = 8765) -> FastMCP:
    mcp = FastMCP("memory", host=host, port=port, log_level="WARNING")

    async def run(fn, *a, **kw):
        svc.touch()
        return await anyio.to_thread.run_sync(lambda: fn(*a, **kw))

    @mcp.tool(description="Search past conversations (keyword+semantic). Returns dated snippets with turn ids.")
    async def search_memory(query: str, k: int = 5, since: str = "", until: str = "", session: str = "") -> str:
        """query: what to look for. k: results (max 10). since/until: YYYY-MM-DD or 7d. session: optional."""
        def go():
            res = svc.search(query, k=k, since=since or None, until=until or None, session=session or None)
            return svc.format_search(res)
        return await run(go)

    @mcp.tool(description="Show the conversation around a turn id (t123) or on a date, with day summary.")
    async def get_conversation(around_turn_id: int = 0, date: str = "", window: int = 3) -> str:
        """around_turn_id: id from search results. date: YYYY-MM-DD. window: turns each side (max 8)."""
        return await run(svc.get_conversation, around_turn_id or None, date or None, window)

    @mcp.tool(description="Save a durable fact/preference/decision/task about the user for future chats.")
    async def remember_fact(statement: str, category: str = "note") -> str:
        """category: profile|preference|decision|task|note."""
        return await run(svc.remember_fact, statement, category)

    @mcp.tool(description="List saved durable facts, newest first.")
    async def list_facts(category: str = "") -> str:
        """category: optional filter."""
        return await run(svc.format_facts, category or None)

    @mcp.tool(description="Memory size/coverage summary.")
    async def memory_stats() -> str:
        return await run(svc.format_stats)

    # Prune schema noise on the registered tools (FastMCP generates verbose pydantic schemas).
    for tool in mcp._tool_manager.list_tools():
        _compact(tool.parameters)
    return mcp


def sse_routes(mcp: FastMCP) -> list:
    """Starlette routes (/sse, /messages/) serving `mcp` over SSE.

    mcp 1.3.0's FastMCP can only run its own uvicorn server (no sse_app()), so we
    mount the same transport objects it uses into our shared ASGI app.
    """
    from mcp.server.sse import SseServerTransport
    from starlette.responses import Response
    from starlette.routing import Mount, Route

    sse = SseServerTransport("/messages/")
    server = mcp._mcp_server

    async def handle_sse(request):
        async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
            await server.run(streams[0], streams[1], server.create_initialization_options())
        return Response()

    return [Route("/sse", endpoint=handle_sse), Mount("/messages/", app=sse.handle_post_message)]
