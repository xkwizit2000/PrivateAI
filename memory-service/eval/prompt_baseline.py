"""Measure prompt size/latency of the LEGACY memory injection vs the proposed design (Ollama, tiny requests).

Builds, from the real backup, the exact kind of prompt the legacy bot sends (summary <=6000 chars + last 8
records within a 12,000-char budget + RAG top-4 <=4,000 chars from a copy of the legacy RAG sqlite using
nomic-embed-text via Ollama) and the proposed variants, then asks Ollama for ONE token (num_predict=1) to
read prompt_eval_count / prompt_eval_duration. Variants (all end with the same short user message):
  A legacy   : system + summary + RAG block + last 8 records
  B core     : system + last 4 records (no tools, no recall)           <- typical non-recall turn
  C core+tools: B + memory tool schemas                                 <- turn where the model may search
  D core+recall: B + gated recall block (<=700 chars)                   <- turn where the gate fired
Prints only sizes/timings.
"""

import argparse
import glob
import json
import math
import os
import sqlite3
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_service.config import Settings  # noqa: E402
from memory_service.embedder import FastEmbedder  # noqa: E402
from memory_service.mcp_server import build_mcp  # noqa: E402
from memory_service.service import MemoryService  # noqa: E402
from memory_service.store import MemoryStore  # noqa: E402

SYSTEM = ("You are PrivateAI, a personal Systems, Security, AI, Automation, Options trading, and strategy development assistant.\n\n"
          "You may use MCP tools when they help answer the user accurately.\nPrefer inspection and read-only tools before any write or destructive action.\n"
          "If a tool fails, explain the failure briefly and suggest a safer next step.\nKeep answers concise and practical.\nWhen no tool is needed, answer directly.\n")


def ollama(host, path, payload, timeout=180):
    req = urllib.request.Request(host + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def legacy_recent(records, turns=4, budget=12000, summary_len=0):
    recent = records[-(turns * 2):]
    remaining = budget - summary_len
    out = []
    for r in reversed(recent):
        c = str(r.get("content", ""))
        if not c:
            continue
        c = c[-remaining:]
        out.insert(0, {"role": r["role"], "content": c})
        remaining -= len(c)
        if remaining <= 0:
            break
    return out


def cosine(a, b):
    d = sum(x * y for x, y in zip(a, b))
    return d / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)) or 1)


def legacy_rag(host, rag_db, query, top_k=4, max_chars=4000, min_score=0.25):
    emb = ollama(host, "/api/embed", {"model": "nomic-embed-text", "input": [query], "options": {"num_gpu": 0}})["embeddings"][0]
    con = sqlite3.connect(f"file:{rag_db}?mode=ro", uri=True)
    scored = []
    for role, content, e in con.execute("SELECT role, content, embedding FROM chunks"):
        s = cosine(emb, json.loads(e))
        if s >= min_score:
            scored.append((s, role, content))
    scored.sort(reverse=True)
    used, lines = 0, ["Relevant past context:"]
    n = 0
    for s, role, content in scored:
        if used + len(content) > max_chars and n:
            break
        lines.append(f"- {role}: {content[:max_chars - used]}")
        used += len(content)
        n += 1
        if n >= top_k:
            break
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts", required=True)
    ap.add_argument("--session", required=True, help="transcript file stem, e.g. the Telegram chat id")
    ap.add_argument("--rag-db", required=True)
    ap.add_argument("--ollama", default="http://127.0.0.1:11434")
    ap.add_argument("--model", default="gemma4:e4b-jt-8k")
    ap.add_argument("--cache", default="/opt/models")
    ap.add_argument("--rag-queries", default="what car do I drive|which model fits my VRAM|why were replies truncated|what instrument am I learning|what is MCP")
    a = ap.parse_args()

    records = [json.loads(l) for l in open(os.path.join(a.transcripts, a.session + ".jsonl")) if l.strip()]
    summary = json.load(open(os.path.join(a.transcripts, a.session + ".summary.json")))["summary"][:6000]
    user_msg = "Thanks. What should I do next?"

    rag_chars = []
    for q in a.rag_queries.split("|"):
        rag_chars.append(len(legacy_rag(a.ollama, a.rag_db, q)))
    rag_block = legacy_rag(a.ollama, a.rag_db, a.rag_queries.split("|")[0])
    recent8 = legacy_recent(records, 4, 12000, len(summary))
    mem_msg = {"role": "system", "content": f"Durable memory from earlier conversations:\n{summary}"}
    rag_msg = {"role": "system", "content": rag_block}
    sysm = {"role": "system", "content": SYSTEM}
    usr = {"role": "user", "content": user_msg}

    s = Settings()
    emb = FastEmbedder(s.embed_model, a.cache, 2, local_only=True)
    store = MemoryStore(":memory:", emb, s)
    for f in sorted(glob.glob(os.path.join(a.transcripts, "*.jsonl"))):
        store.ingest_jsonl(f)
    svc = MemoryService(store, s)
    tools = [{"type": "function", "function": {"name": f"memory_{t.name}", "description": t.description, "parameters": t.parameters}}
             for t in build_mcp(svc)._tool_manager.list_tools()]
    recall = svc.recall("do you remember what car I was asking about earlier?", session=a.session, exclude_recent=8)
    recall_msg = {"role": "system", "content": "Possibly relevant earlier conversation:\n" + recall["text"]}
    recent4 = legacy_recent(records, 2, 12000, 0)

    variants = {
        "A legacy (summary+RAG+last8)": ([sysm, mem_msg, rag_msg, *recent8, usr], None),
        "B core (last4 only)": ([sysm, *recent4, usr], None),
        "C core + memory tools": ([sysm, *recent4, usr], tools),
        "D core + gated recall(<=700)": ([sysm, recall_msg, *recent4, usr], None),
    }
    print(json.dumps({"summary_chars": len(summary), "recent8_chars": sum(len(m["content"]) for m in recent8),
                      "rag_block_chars_for_5_queries": rag_chars, "recall_chars": len(recall["text"]),
                      "tool_schema_chars": len(json.dumps(tools))}))
    for name, (msgs, tl) in variants.items():
        chars = sum(len(m["content"]) for m in msgs) + (len(json.dumps(tl)) if tl else 0)
        t0 = time.perf_counter()
        r = ollama(a.ollama, "/api/chat", {"model": a.model, "messages": msgs, "tools": tl, "stream": False, "think": False,
                                           "options": {"num_ctx": 8192, "num_predict": 1}})
        wall = time.perf_counter() - t0
        pe, pd = r.get("prompt_eval_count"), (r.get("prompt_eval_duration") or 0) / 1e9
        print(json.dumps({"variant": name, "prompt_chars": chars, "prompt_tokens": pe, "prompt_eval_s": round(pd, 2),
                          "tok_per_s": round(pe / pd, 1) if pd else None, "wall_s": round(wall, 2),
                          "chars_per_token": round(chars / pe, 2) if pe else None}))


if __name__ == "__main__":
    main()
