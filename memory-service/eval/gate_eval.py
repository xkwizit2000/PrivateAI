"""How often would the no-LLM gate auto-inject memory on a real transcript?

Replays each user message against ONLY the history that preceded it (until=its timestamp,
last 8 turns excluded, like the bot's own recent-turn window). Prints aggregate numbers only.
"""

import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_service.config import Settings  # noqa: E402
from memory_service.embedder import FastEmbedder  # noqa: E402
from memory_service.gate import refers_to_past, should_inject  # noqa: E402
from memory_service.service import MemoryService  # noqa: E402
from memory_service.store import MemoryStore  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts", required=True)
    ap.add_argument("--cache", default="/opt/models")
    ap.add_argument("--min-seq", type=int, default=20)
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    s = Settings()
    emb = FastEmbedder(s.embed_model, a.cache, 2, local_only=True)
    store = MemoryStore(":memory:", emb, s)
    for f in sorted(glob.glob(os.path.join(a.transcripts, "*.jsonl"))):
        store.ingest_jsonl(f)
    svc = MemoryService(store, s)
    rows = store.conn.execute("SELECT * FROM turns WHERE role='user' AND seq>=? ORDER BY id", (a.min_seq,)).fetchall()
    data = []
    lat = []
    for r in rows:
        until = (datetime.fromisoformat(r["ts"]) - timedelta(seconds=1)).isoformat()
        excl = [t for t in store.recent_turn_ids(r["session_id"], 0)]  # placeholder, replaced below
        recent = store.conn.execute(
            "SELECT id FROM turns WHERE session_id=? AND seq<? ORDER BY seq DESC LIMIT 8", (r["session_id"], r["seq"])
        ).fetchall()
        t0 = time.perf_counter()
        res = svc.search_chunks(r["content"][:400], k=2, until=until, exclude_turns=[x["id"] for x in recent], recency=False)
        lat.append((time.perf_counter() - t0) * 1000)
        data.append({"id": r["id"], "cue": refers_to_past(r["content"]), "cos": res.top_cos, "len": len(r["content"])})
    n = len(data)
    print(json.dumps({"user_messages": n, "with_past_cue": sum(d["cue"] for d in data),
                      "gate_latency_ms_p50": round(sorted(lat)[n // 2], 1), "p95": round(sorted(lat)[int(n * .95)], 1)}))
    print("threshold sensitivity (fire counts / fire rate):")
    for cue_min in (0.40, 0.45, 0.50, 0.55):
        for strong in (0.65, 0.72, 0.80, 0.90):
            fired = [should_inject(str(d["id"]) and "", None, 0, 0)[0] for d in data[:0]]  # noqa: F841
            cnt_cue = sum(1 for d in data if d["cue"] and d["cos"] is not None and d["cos"] >= cue_min and d["cos"] < strong)
            cnt_strong = sum(1 for d in data if d["cos"] is not None and d["cos"] >= strong)
            print(f"  cue_min={cue_min:.2f} strong={strong:.2f}: cue-fired={cnt_cue:3d} strong-fired={cnt_strong:3d} "
                  f"rate={(cnt_cue + cnt_strong) / n:.1%}")
    cos = sorted(d["cos"] for d in data if d["cos"] is not None)
    q = lambda p: round(cos[int(p * (len(cos) - 1))], 3)
    print("top_cos quantiles (p10,p50,p90):", q(0.1), q(0.5), q(0.9))
    if a.show:
        for d in data:
            if d["cue"] or (d["cos"] or 0) >= s.gate_strong_cos:
                t = store.get_turn(d["id"])
                print(f"  t{d['id']} {t['day']} cue={d['cue']} cos={d['cos']:.2f} :: {' '.join(t['content'].split())[:80]!r}")


if __name__ == "__main__":
    main()
