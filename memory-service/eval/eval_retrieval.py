"""Label-free retrieval-quality + embedder benchmark on real transcripts.

Usage (inside the service image):
  python eval/eval_retrieval.py --transcripts /eval --model BAAI/bge-small-en-v1.5 --n 100 --out res.json

Nothing from the transcripts is printed except aggregate metrics. Query types:
  keyword  - 3-4 most distinctive (highest-IDF) terms of a sampled turn
  fragment - a 6-10 word contiguous fragment from the middle of a sampled turn
  pair     - the *other* half of a user/assistant exchange (<=200 chars) used as the query;
             target = its partner turn, query turn excluded from the results
Each is scored against the source turn for keyword / vector / hybrid modes
(recall@1/5/10 and MRR). Hybrid is also scored with the recency boost on.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import re
import resource
import statistics
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_service.config import Settings  # noqa: E402
from memory_service.service import STOPWORDS, MemoryService  # noqa: E402
from memory_service.store import MemoryStore  # noqa: E402

WORD = re.compile(r"[A-Za-z0-9_]+")


def rss_mb() -> float:
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024
    return 0.0


def build_queries(svc: MemoryService, n: int, rng: random.Random):
    rows = svc.store.conn.execute("SELECT id, session_id, seq, role, content FROM turns ORDER BY id").fetchall()
    toks = {r["id"]: [t.lower() for t in WORD.findall(r["content"])] for r in rows}
    df = Counter()
    for t in toks.values():
        df.update(set(t))
    N = len(rows)
    by_key = {(r["session_id"], r["seq"]): r for r in rows}
    eligible = [r for r in rows[:-16] if len(r["content"]) >= 60]

    def distinct(r):
        cands = {t for t in toks[r["id"]] if len(t) >= 4 and t not in STOPWORDS and not t.isdigit() and df[t] <= max(3, N * 0.15)}
        return sorted(cands, key=lambda t: (-math.log(N / df[t]), t))[:4]

    out = {"keyword": [], "fragment": [], "pair": []}
    pool = eligible[:]
    rng.shuffle(pool)
    for r in pool:
        d = distinct(r)
        if len(d) >= 3 and len(out["keyword"]) < n:
            out["keyword"].append((" ".join(d), r["id"], None))
    rng.shuffle(pool)
    for r in pool:
        words = r["content"].split()
        if len(words) >= 14 and len(out["fragment"]) < n:
            ln = rng.randint(6, 10)
            st = rng.randint(2, len(words) - ln - 1)
            out["fragment"].append((" ".join(words[st : st + ln]), r["id"], None))
    rng.shuffle(pool)
    for r in pool:
        partner_seq = r["seq"] + (1 if r["role"] == "user" else -1)
        p = by_key.get((r["session_id"], partner_seq))
        if p is not None and p["role"] != r["role"] and len(p["content"]) >= 60 and len(out["pair"]) < n:
            out["pair"].append((" ".join(p["content"].split())[:200], r["id"], p["id"]))  # query from p, target r
    return out


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def rank_of(svc, q, target, exclude, mode, recency):
    """Rank of the target turn; a hit with identical text (a retried duplicate) counts as the target."""
    res = svc.search_chunks(q, k=10, mode=mode, exclude_turns=exclude, recency=recency)
    cache = svc.__dict__.setdefault("_eval_norm", {})
    if target not in cache:
        cache[target] = _norm(svc.store.get_turn(target)["content"])
    for i, h in enumerate(res.hits):
        if h.turn_id == target:
            return i + 1
        if h.turn_id not in cache:
            cache[h.turn_id] = _norm(svc.store.get_turn(h.turn_id)["content"])
        if cache[h.turn_id] == cache[target]:
            return i + 1
    return None


def summarize(ranks):
    n = len(ranks)
    r = lambda k: round(sum(1 for x in ranks if x is not None and x <= k) / n, 3)
    mrr = round(sum(1 / x for x in ranks if x) / n, 3)
    return {"n": n, "r@1": r(1), "r@5": r(5), "r@10": r(10), "mrr": mrr}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts", required=True)
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--backend", default="fastembed")
    ap.add_argument("--cache", default=os.getenv("EMBED_CACHE_DIR", "/opt/models"))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--online", action="store_true", help="allow model download")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out")
    ap.add_argument("--grid", action="store_true", help="also grid-search fusion weights")
    a = ap.parse_args()

    s = Settings()
    s.embed_backend, s.embed_model, s.embed_threads = a.backend, a.model, a.threads
    s.embed_cache_dir, s.embed_local_only = a.cache, not a.online
    rss0 = rss_mb()
    t0 = time.perf_counter()
    if a.backend == "hash":
        from memory_service.embedder import HashEmbedder
        emb = HashEmbedder()
    else:
        from memory_service.embedder import FastEmbedder
        emb = FastEmbedder(a.model, a.cache, a.threads, local_only=not a.online)
    load_s = time.perf_counter() - t0
    rss_loaded = rss_mb()

    store = MemoryStore(":memory:", emb, s)
    t0 = time.perf_counter()
    tot_chunks = 0
    for f in sorted(glob.glob(os.path.join(a.transcripts, "*.jsonl"))):
        tot_chunks += store.ingest_jsonl(f).chunks_new
    ingest_s = time.perf_counter() - t0
    svc = MemoryService(store, s)
    rss_ingested = rss_mb()

    qs = build_queries(svc, a.n, random.Random(a.seed))
    result = {
        "model": a.model if a.backend != "hash" else "hash", "dim": emb.dim, "threads": a.threads,
        "load_s": round(load_s, 2), "rss_before_mb": round(rss0), "rss_loaded_mb": round(rss_loaded),
        "rss_after_ingest_mb": round(rss_ingested), "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        "chunks": tot_chunks, "ingest_s": round(ingest_s, 1), "chunks_per_s": round(tot_chunks / ingest_s, 1),
        "quality": {},
    }
    lat = []
    for qtype, items in qs.items():
        result["quality"][qtype] = {}
        for mode, recency in (("keyword", False), ("vector", False), ("hybrid", False), ("hybrid", True)):
            ranks = []
            for q, target, exclude_turn in items:
                t = time.perf_counter()
                ranks.append(rank_of(svc, q, target, [exclude_turn] if exclude_turn else None, mode, recency))
                if mode == "hybrid":
                    lat.append((time.perf_counter() - t) * 1000)
            name = mode + ("+recency" if recency else "")
            result["quality"][qtype][name] = summarize(ranks) if ranks else {"n": 0}
    if a.grid:
        grid = []
        for k in (5, 10, 30, 60):
            for wf in (1.0, 1.5, 2.0, 3.0):
                s.rrf_k, s.fts_weight = k, wf
                row = {"rrf_k": k, "fts_w": wf}
                for qtype, items in qs.items():
                    ranks = [rank_of(svc, q, t, [x] if x else None, "hybrid", False) for q, t, x in items]
                    row[qtype] = summarize(ranks)["mrr"]
                row["avg_mrr"] = round(sum(row[t] for t in qs) / len(qs), 3)
                grid.append(row)
        result["grid"] = grid
        s.rrf_k, s.fts_weight = Settings().rrf_k, Settings().fts_weight
    lat.sort()
    result["hybrid_latency_ms"] = {"p50": round(statistics.median(lat), 1), "p95": round(lat[int(0.95 * len(lat))], 1), "n": len(lat)}
    text = json.dumps(result, indent=1)
    print(text)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text)


if __name__ == "__main__":
    main()
