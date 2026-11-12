"""CLI: serve | mcp-stdio | ingest | search | stats | facts | get | consolidate | reembed | purge."""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import time

from .config import Settings


def _service(settings: Settings):
    from .embedder import make_embedder
    from .service import MemoryService
    from .store import MemoryStore

    emb = make_embedder(settings)
    return MemoryService(MemoryStore(settings.db_path, emb, settings), settings)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="memory_service")
    p.add_argument("--db", help="override DB_PATH")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("serve", help="HTTP API + MCP over SSE")
    sub.add_parser("mcp-stdio", help="MCP over stdio (local use)")
    ing = sub.add_parser("ingest", help="backfill from bot transcript JSONL file(s)/dir")
    ing.add_argument("paths", nargs="+")
    se = sub.add_parser("search")
    se.add_argument("query")
    se.add_argument("-k", type=int, default=5)
    se.add_argument("--since"), se.add_argument("--until"), se.add_argument("--session")
    se.add_argument("--mode", default="hybrid", choices=["hybrid", "vector", "keyword"])
    se.add_argument("--json", action="store_true")
    sub.add_parser("stats")
    fa = sub.add_parser("facts")
    fa.add_argument("--add"), fa.add_argument("--category", default="note")
    ge = sub.add_parser("get")
    ge.add_argument("--turn", type=int), ge.add_argument("--date"), ge.add_argument("--window", type=int, default=3)
    co = sub.add_parser("consolidate")
    co.add_argument("--llm", choices=["ollama", "extractive"], default="extractive")
    co.add_argument("--max-units", type=int, default=3)
    co.add_argument("--force", action="store_true", help="ignore the idle check")
    sub.add_parser("reembed")
    pu = sub.add_parser("purge")
    pu.add_argument("turn_ids", type=int, nargs="+")
    a = p.parse_args(argv)

    settings = Settings.from_env()
    if a.db:
        settings.db_path = a.db
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if a.cmd in ("serve", "mcp-stdio"):
        from .mcp_server import build_mcp

        t0 = time.perf_counter()
        svc = _service(settings)
        logging.getLogger("memory").info("ready in %.1fs: %s", time.perf_counter() - t0, svc.format_stats())
        mcp = build_mcp(svc, settings.host, settings.port)
        if a.cmd == "mcp-stdio":
            mcp.run("stdio")
            return 0
        import uvicorn

        from .http_api import build_app

        uvicorn.run(build_app(svc, mcp, settings.auth_token), host=settings.host, port=settings.port,
                    log_level="warning")
        return 0

    svc = _service(settings)
    if a.cmd == "ingest":
        files = []
        for pth in a.paths:
            files += sorted(glob.glob(os.path.join(pth, "*.jsonl"))) if os.path.isdir(pth) else [pth]
        t0 = time.perf_counter()
        tot = {"turns_new": 0, "chunks_new": 0, "redactions": 0}
        for f in files:
            r = svc.store.ingest_jsonl(f)
            for k in tot:
                tot[k] += getattr(r, k)
            print(f"{os.path.basename(f)[:16]}: +{r.turns_new} turns +{r.chunks_new} chunks ({r.turns_seen} seen)")
        print(json.dumps({**tot, "seconds": round(time.perf_counter() - t0, 2)}))
    elif a.cmd == "search":
        res = svc.search(a.query, a.k, a.since, a.until, a.session, a.mode)
        if a.json:
            print(json.dumps([h.__dict__ for h in res.hits], default=str, indent=1))
        else:
            print(svc.format_search(res))
        print(f"-- {res.ms:.1f} ms (embed {res.embed_ms:.1f} ms), top_cos={res.top_cos}", file=sys.stderr)
    elif a.cmd == "stats":
        print(json.dumps(svc.store.stats(), indent=1))
    elif a.cmd == "facts":
        print(svc.remember_fact(a.add, a.category) if a.add else svc.format_facts(None))
    elif a.cmd == "get":
        print(svc.get_conversation(a.turn, a.date, a.window))
    elif a.cmd == "consolidate":
        from . import consolidate

        fn = (consolidate.ollama_summarizer(settings.ollama_host, settings.ollama_model)
              if a.llm == "ollama" else consolidate.extractive_summarizer)
        print(json.dumps(consolidate.run(svc, fn, max_units=a.max_units, force=a.force, model_name=a.llm)))
    elif a.cmd == "reembed":
        print(f"re-embedded {svc.store.reembed_all()} chunks")
    elif a.cmd == "purge":
        print(f"purged {svc.store.purge_turns(a.turn_ids)} turns")
    return 0
