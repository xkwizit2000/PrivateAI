"""STRETCH: does the local model call the memory tool sensibly? (standalone; not via the bot)

Sends the compact memory tool schemas (named like the bot would see them) to Ollama /api/chat
(num_ctx 8192, think false) with a bot-like system prompt, executes tool calls against the
in-process service, feeds results back and records latency / grounding. A handful of requests only.

  python eval/llm_tool_test.py --transcripts /eval --questions /eval/llm_q.json --ollama http://127.0.0.1:11434
  questions: [{"q": "...", "expect": "regex that a grounded answer should contain"}]
"""

import argparse
import glob
import json
import os
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_service.config import Settings  # noqa: E402
from memory_service.embedder import FastEmbedder  # noqa: E402
from memory_service.mcp_server import build_mcp  # noqa: E402
from memory_service.service import MemoryService  # noqa: E402
from memory_service.store import MemoryStore  # noqa: E402

SYSTEM = (
    "You are PrivateAI, a personal assistant. You have tools to search long-term memory of past conversations; "
    "use them when the user refers to earlier conversations or personal details you do not have in this chat. "
    "Keep answers short and base them on tool results."
)


def chat(host, model, messages, tools, timeout):
    body = json.dumps({"model": model, "messages": messages, "tools": tools, "stream": False, "think": False,
                       "options": {"num_ctx": 8192, "num_predict": 200, "temperature": 0.2}}).encode()
    req = urllib.request.Request(host.rstrip("/") + "/api/chat", data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    return data, time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--ollama", default="http://127.0.0.1:11434")
    ap.add_argument("--model", default="gemma4:e4b-jt-8k")
    ap.add_argument("--cache", default="/opt/models")
    ap.add_argument("--timeout", type=float, default=120)
    a = ap.parse_args()

    s = Settings()
    emb = FastEmbedder(s.embed_model, a.cache, 2, local_only=True)
    store = MemoryStore(":memory:", emb, s)
    for f in sorted(glob.glob(os.path.join(a.transcripts, "*.jsonl"))):
        store.ingest_jsonl(f)
    svc = MemoryService(store, s)
    mcp = build_mcp(svc)
    # Same shape/naming the bot would send (server key "memory" -> memory_<tool>).
    tools = [{"type": "function", "function": {"name": f"memory_{t.name}", "description": t.description, "parameters": t.parameters}}
             for t in mcp._tool_manager.list_tools()]
    print("tool schema chars:", len(json.dumps(tools)))
    dispatch = {
        "memory_search_memory": lambda **kw: svc.format_search(svc.search(kw["query"], int(kw.get("k", 5) or 5), kw.get("since") or None, kw.get("until") or None, kw.get("session") or None)),
        "memory_get_conversation": lambda **kw: svc.get_conversation(kw.get("around_turn_id") or None, kw.get("date") or None, int(kw.get("window", 3) or 3)),
        "memory_list_facts": lambda **kw: svc.format_facts(kw.get("category") or None),
        "memory_memory_stats": lambda **kw: svc.format_stats(),
        "memory_remember_fact": lambda **kw: svc.remember_fact(kw.get("statement", ""), kw.get("category", "note")),
    }
    results = []
    for item in json.load(open(a.questions)):
        q = item["q"]
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}]
        rec = {"q": q, "tool_called": None, "args": None, "latency_s": [], "prompt_tokens": [], "answer_ok": None}
        try:
            data, dt = chat(a.ollama, a.model, msgs, tools, a.timeout)
        except Exception as exc:  # noqa: BLE001
            rec["error"] = repr(exc)
            results.append(rec)
            print("ERROR", repr(exc)[:100])
            break
        rec["latency_s"].append(round(dt, 1)); rec["prompt_tokens"].append(data.get("prompt_eval_count"))
        msg = data["message"]
        calls = msg.get("tool_calls") or []
        answer = msg.get("content", "")
        if calls:
            msgs.append(msg)
            for c in calls:
                name, args = c["function"]["name"], c["function"].get("arguments") or {}
                rec["tool_called"], rec["args"] = name, args
                try:
                    out = dispatch[name](**args)
                except Exception as exc:  # noqa: BLE001
                    out = f"Tool error: {exc}"
                rec["tool_result_chars"] = len(out)
                msgs.append({"role": "tool", "content": out})
            data, dt = chat(a.ollama, a.model, msgs, tools, a.timeout)
            rec["latency_s"].append(round(dt, 1)); rec["prompt_tokens"].append(data.get("prompt_eval_count"))
            answer = data["message"].get("content", "")
        rec["answer"] = answer.strip()
        rec["answer_ok"] = bool(re.search(item["expect"], answer, re.I)) if item.get("expect") else None
        results.append(rec)
        print(json.dumps({k: v for k, v in rec.items() if k != "answer"}), "|", " ".join(answer.split())[:160])
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
