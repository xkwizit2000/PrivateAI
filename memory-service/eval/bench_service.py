"""Latency benchmark against a running service over HTTP (stdlib only).

Queries are random 6-10 word fragments sampled from the transcript files (never printed).
  python eval/bench_service.py --url http://memsvc:8765 --token T --transcripts /eval --n 150
"""

import argparse
import glob
import json
import os
import random
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def post(url, path, token, payload):
    req = urllib.request.Request(url + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=30) as r:
        out = json.loads(r.read())
    return (time.perf_counter() - t) * 1000, out


def stats(xs):
    xs = sorted(xs)
    return {"n": len(xs), "p50": round(statistics.median(xs), 1), "p95": round(xs[int(0.95 * (len(xs) - 1))], 1),
            "max": round(xs[-1], 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--token", default="")
    ap.add_argument("--transcripts", required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--session", default="", help="session id (transcript file stem) for /v1/recall")
    a = ap.parse_args()
    rng = random.Random(3)
    texts = []
    for f in glob.glob(os.path.join(a.transcripts, "*.jsonl")):
        for line in open(f):
            if line.strip():
                texts.append(json.loads(line)["content"])
    qs = []
    while len(qs) < a.n:
        w = rng.choice(texts).split()
        if len(w) >= 14:
            ln = rng.randint(6, 10)
            st = rng.randint(0, len(w) - ln)
            qs.append(" ".join(w[st:st + ln]))
    for q in qs[:5]:  # warm-up
        post(a.url, "/v1/search", a.token, {"query": q})
    out = {}
    for mode in ("hybrid", "vector", "keyword"):
        lat, server_ms, chars = [], [], []
        for q in qs:
            ms, r = post(a.url, "/v1/search", a.token, {"query": q, "k": 5, "mode": mode})
            lat.append(ms); server_ms.append(r["ms"]); chars.append(len(r["text"]))
        out[f"search_{mode}_http_ms"] = stats(lat)
        out[f"search_{mode}_server_ms"] = stats(server_ms)
        if mode == "hybrid":
            out["search_hybrid_output_chars"] = {"p50": int(statistics.median(chars)), "max": max(chars)}
    lat = [post(a.url, "/v1/recall", a.token, {"query": q, "session": a.session or None})[0] for q in qs[:100]]
    out["recall_http_ms"] = stats(lat)
    with ThreadPoolExecutor(4) as ex:
        t = time.perf_counter()
        list(ex.map(lambda q: post(a.url, "/v1/search", a.token, {"query": q}), qs[:60]))
        out["parallel4_60_queries_total_s"] = round(time.perf_counter() - t, 2)
    lat = []
    for i in range(10):
        ms, _ = post(a.url, "/v1/turns", a.token, {"session_id": "bench", "turns": [
            {"role": "user", "content": f"synthetic benchmark question number {i} about widgets"},
            {"role": "assistant", "content": f"synthetic benchmark answer number {i} about gadgets and widgets"}]})
        lat.append(ms)
    out["append_turn_pair_http_ms"] = stats(lat)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
