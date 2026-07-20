"""Local HTTP bridge used by the Session.js adapter."""

from __future__ import annotations

import logging
from typing import Any

from aiohttp import web

from .config import (
    BRIDGE_HOST,
    BRIDGE_PORT,
    BRIDGE_TOKEN,
    SESSION_ALLOWED_IDS,
)
from .runtime import AppRuntime

logger = logging.getLogger(__name__)


def _authorized(request: web.Request, session_id: str) -> bool:
    if BRIDGE_TOKEN:
        auth = request.headers.get("Authorization", "")
        expected = f"Bearer {BRIDGE_TOKEN}"
        if auth != expected:
            return False
    if SESSION_ALLOWED_IDS and session_id not in SESSION_ALLOWED_IDS:
        return False
    return True


async def handle_health(_: web.Request) -> web.Response:
    return web.json_response({"ok": True, "service": "privateai-bridge"})


async def handle_message(request: web.Request) -> web.Response:
    runtime: AppRuntime = request.app["runtime"]
    try:
        payload: dict[str, Any] = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON body"}, status=400)

    session_id = str(payload.get("session_id") or payload.get("from") or "").strip()
    text = str(payload.get("text") or "").strip()
    if not session_id or not text:
        return web.json_response(
            {"error": "session_id (or from) and text are required"},
            status=400,
        )

    if not _authorized(request, session_id):
        return web.json_response({"error": "unauthorized"}, status=403)

    try:
        reply = await runtime.handle_text(session_id, text)
    except Exception:
        logger.exception("Bridge chat request failed for session %s", session_id)
        return web.json_response(
            {"error": "agent request failed"},
            status=500,
        )

    return web.json_response({"reply": reply, "session_id": session_id})


async def on_startup(app: web.Application) -> None:
    app["runtime"] = await AppRuntime.create()


async def on_cleanup(app: web.Application) -> None:
    runtime: AppRuntime | None = app.get("runtime")
    if runtime is not None:
        await runtime.close()


def create_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/health", handle_health)
    app.router.add_post("/v1/message", handle_message)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


def main() -> None:
    if SESSION_ALLOWED_IDS:
        logger.info(
            "Session bridge allowlist enabled (%d id(s))",
            len(SESSION_ALLOWED_IDS),
        )
    else:
        logger.warning(
            "SESSION_ALLOWED_ID is empty; bridge will accept any session_id "
            "(still protect with BRIDGE_TOKEN if exposed beyond localhost)"
        )

    app = create_app()
    logger.info(
        "Headless Bot Server Online via Session bridge on http://%s:%s ...",
        BRIDGE_HOST,
        BRIDGE_PORT,
    )
    web.run_app(app, host=BRIDGE_HOST, port=BRIDGE_PORT, print=None)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()
