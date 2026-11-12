"""Thin HTTP API (+ MCP SSE mounted in the same ASGI app, one process, one embedder)."""

from __future__ import annotations

import hmac
import json

import anyio
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route
from starlette.types import ASGIApp, Receive, Scope, Send

from .service import MemoryService


class BearerAuth:
    """Optional shared-secret auth (Authorization: Bearer <token>); /healthz is open."""

    def __init__(self, app: ASGIApp, token: str) -> None:
        self.app, self.token = app, token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and self.token and scope["path"] != "/healthz":
            headers = dict(scope["headers"])
            supplied = headers.get(b"authorization", b"").decode()
            expected = f"Bearer {self.token}"
            if not hmac.compare_digest(supplied.encode(), expected.encode()):
                await PlainTextResponse("unauthorized", status_code=401)(scope, receive, send)
                return
        await self.app(scope, receive, send)


def build_app(svc: MemoryService, mcp=None, auth_token: str = "") -> Starlette:
    async def body(request) -> dict:
        try:
            data = await request.json()
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    async def work(fn, *a, **kw):
        return await anyio.to_thread.run_sync(lambda: fn(*a, **kw))

    async def healthz(request):
        return JSONResponse({"ok": True, "idle_s": round(svc.idle_for(), 1)})

    async def stats(request):
        return JSONResponse(await work(svc.store.stats))

    async def search(request):
        d = await body(request)
        svc.touch()

        def go():
            res = svc.search(str(d.get("query", "")), int(d.get("k", 5)), d.get("since"), d.get("until"),
                             d.get("session"), d.get("mode", "hybrid"))
            return {"text": svc.format_search(res, d.get("max_chars")), "top_cos": res.top_cos,
                    "ms": round(res.ms, 1),
                    "hits": [{"turn_id": h.turn_id, "day": h.day, "role": h.role, "score": h.score,
                              "vec_cos": h.vec_cos} for h in res.hits]}
        return JSONResponse(await work(go))

    async def recall(request):
        d = await body(request)
        svc.touch()
        return JSONResponse(await work(svc.recall, str(d.get("query", "")), d.get("session"),
                                       int(d.get("exclude_recent", 8)), int(d.get("max_hits", 2)),
                                       d.get("max_chars")))

    async def turns(request):
        """Bot -> service: append completed turn(s). {session_id, turns:[{role,content,timestamp?}]}"""
        d = await body(request)
        svc.touch()
        sid = str(d.get("session_id", ""))
        items = [{**t, "session_id": sid} for t in d.get("turns", []) if sid]
        res = await work(svc.store.add_turns, items)
        return JSONResponse({"turns_new": res.turns_new, "chunks_new": res.chunks_new})

    async def core(request):
        svc.touch()
        sid = request.query_params.get("session", "")
        return JSONResponse(await work(svc.core, sid, int(request.query_params.get("turns", 4))))

    async def conversation(request):
        d = await body(request)
        svc.touch()
        text = await work(svc.get_conversation, d.get("around_turn_id"), d.get("date"), int(d.get("window", 3)),
                          d.get("session"))
        return JSONResponse({"text": text})

    async def facts(request):
        svc.touch()
        if request.method == "POST":
            d = await body(request)
            msg = await work(svc.remember_fact, str(d.get("statement", "")), d.get("category", "note"),
                             d.get("source_turn_ids", []), d.get("session"))
            return JSONResponse({"result": msg})
        return JSONResponse({"text": await work(svc.format_facts, request.query_params.get("category"))})

    routes = [
        Route("/healthz", healthz),
        Route("/v1/stats", stats),
        Route("/v1/search", search, methods=["POST"]),
        Route("/v1/recall", recall, methods=["POST"]),
        Route("/v1/turns", turns, methods=["POST"]),
        Route("/v1/core", core),
        Route("/v1/conversation", conversation, methods=["POST"]),
        Route("/v1/facts", facts, methods=["GET", "POST"]),
    ]
    if mcp is not None:
        from .mcp_server import sse_routes

        routes.extend(sse_routes(mcp))  # /sse and /messages/
    middleware = [Middleware(BearerAuth, token=auth_token)] if auth_token else []
    return Starlette(routes=routes, middleware=middleware)
