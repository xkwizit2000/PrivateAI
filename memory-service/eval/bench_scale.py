"""Brute-force vector scan cost at larger corpus sizes (random unit vectors; no personal data)."""

import json
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from memory_service.store import VectorIndex  # noqa: E402

DIM = int(os.getenv("DIM", "384"))
rng = np.random.default_rng(0)
res = {}
for n in (10_000, 50_000, 100_000, 250_000):
    idx = VectorIndex(DIM)
    v = rng.standard_normal((n, DIM)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    idx.add(np.arange(n), np.arange(n), np.zeros(n), ["s"] * n, v)
    q = v[rng.integers(0, n)]
    lat = []
    for _ in range(30):
        t = time.perf_counter()
        idx.top(q, 50, idx.mask(since=-1.0))
        lat.append((time.perf_counter() - t) * 1000)
    res[n] = {"matrix_mb": round(idx.mat[:n].nbytes / 1e6, 1), "scan_ms_p50": round(statistics.median(lat), 2),
              "scan_ms_max": round(max(lat), 2)}
print(json.dumps(res, indent=1))
