"""Score hand-written natural-language queries (precision@3 etc.) against a transcript set.

Queries live in a PRIVATE json file (never committed), format:
  [{"q": "what's wrong with my car", "rel": "audi|misfir|vacuum"}, ...]
A returned turn counts as relevant if the regex `rel` matches its full content
(case-insensitive). I additionally eyeball the top hits (`--show`) to catch
relevant results the regex misses; aggregate numbers are what gets reported.
"""

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_service.config import Settings  # noqa: E402
from memory_service.embedder import FastEmbedder  # noqa: E402
from memory_service.service import MemoryService  # noqa: E402
from memory_service.store import MemoryStore  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--cache", default="/opt/models")
    ap.add_argument("--online", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--maxshow", type=int, default=100)
    a = ap.parse_args()

    s = Settings()
    s.recency_weight = 0.0
    emb = FastEmbedder(a.model, a.cache, 2, local_only=not a.online)
    store = MemoryStore(":memory:", emb, s)
    for f in sorted(glob.glob(os.path.join(a.transcripts, "*.jsonl"))):
        store.ingest_jsonl(f)
    svc = MemoryService(store, s)
    queries = json.load(open(a.queries))

    variants = {
        "keyword": dict(mode="keyword"), "vector": dict(mode="vector"),
        "hybrid(k60,w1)": dict(mode="hybrid", rrf_k=60, fts_weight=1.0),
        "hybrid(k10,w1.5)": dict(mode="hybrid", rrf_k=10, fts_weight=1.5),
    }
    summary = {}
    for name, v in variants.items():
        s.rrf_k, s.fts_weight = v.get("rrf_k", 60), v.get("fts_weight", 1.0)
        p3, h5, first = [], [], []
        for item in queries:
            rx = re.compile(item["rel"], re.I)
            res = svc.search_chunks(item["q"], k=5, mode=v["mode"], recency=False)
            rel = []
            for h in res.hits:
                content = store.get_turn(h.turn_id)["content"]
                rel.append(bool(rx.search(content)))
            p3.append(sum(rel[:3]) / 3)
            h5.append(any(rel))
            first.append(next((i + 1 for i, r in enumerate(rel) if r), None))
            if a.show and name == "hybrid(k10,w1.5)":
                print(f"\nQ: {item['q']}  rel@5={rel}")
                for h, r in list(zip(res.hits, rel))[:3]:
                    print(f"   {'+' if r else '-'} t{h.turn_id} {h.day} {h.role} cos={h.vec_cos} :: {h.text[:a.maxshow]!r}")
        summary[name] = {
            "P@3": round(sum(p3) / len(p3), 3), "hit@5": round(sum(h5) / len(h5), 3),
            "MRR": round(sum(1 / f for f in first if f) / len(first), 3), "n": len(queries),
        }
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
