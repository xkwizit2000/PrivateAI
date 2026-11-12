"""Backfill the *running* memory service from bot transcripts over its HTTP API (stdlib only).

Why HTTP instead of `python -m memory_service ingest`: the live service keeps its vector
index in memory, so rows written by a second process would stay invisible until a restart.
Posting to /v1/turns updates the index of the running process.

The transcripts are only read (mount the directory read-only). Output is metadata only
(file stem hash, record/turn/chunk counts) - never any message text.

  docker run --rm --network <compose-network> --user 10001 --entrypoint "" \
     -v ~/PrivateAI/data/sessions:/in:ro -v $PWD/integration:/it:ro \
     -e MEMORY_TOKEN -e MEM_URL=http://memory-service:8765 \
     privateai:latest python /it/backfill_http.py /in [--count-only] [--sleep 0.2]

Idempotent: turns are keyed on (session, sha256(timestamp|role|content)), so re-running
(or a later dual-write of the same record) never duplicates anything.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
import urllib.request

ROLES = {"user", "assistant", "system", "tool"}


def read_records(path: str) -> tuple[list[dict], int]:
    """Return (usable records, malformed line count) of one JSONL transcript."""
    records: list[dict] = []
    bad = 0
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if isinstance(record, dict):
                records.append(record)
            else:
                bad += 1
    return records, bad


def post(url: str, token: str, payload: dict, timeout: float) -> dict:
    """POST JSON with the bearer token; returns the parsed JSON answer."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {token}"} if token else {})},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - internal URL
        return json.loads(response.read())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("paths", nargs="+", help="transcript directories or *.jsonl files")
    parser.add_argument("--url", default=os.environ.get("MEM_URL", "http://memory-service:8765"))
    parser.add_argument("--batch", type=int, default=32, help="records per request")
    parser.add_argument("--sleep", type=float, default=0.2, help="pause between requests (keeps CPU use gentle)")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--count-only", action="store_true", help="only count records, send nothing")
    args = parser.parse_args()

    token = os.environ.get("MEMORY_TOKEN", "")
    files: list[str] = []
    for path in args.paths:
        files += sorted(glob.glob(os.path.join(path, "*.jsonl"))) if os.path.isdir(path) else [path]

    totals = {"files": 0, "records": 0, "usable": 0, "turns_new": 0, "chunks_new": 0, "malformed": 0}
    started = time.perf_counter()
    for path in files:
        session_id = os.path.splitext(os.path.basename(path))[0]
        label = hashlib.sha256(session_id.encode()).hexdigest()[:8]
        records, bad = read_records(path)
        usable = [
            {"role": str(r.get("role", "user")), "content": str(r.get("content", "")), "timestamp": r.get("timestamp")}
            for r in records
            if str(r.get("content", "") or "").strip() and str(r.get("role", "user")) in ROLES
        ]
        file_new = file_chunks = 0
        if not args.count_only:
            for i in range(0, len(usable), args.batch):
                answer = post(
                    args.url.rstrip("/") + "/v1/turns",
                    token,
                    {"session_id": session_id, "turns": usable[i : i + args.batch]},
                    args.timeout,
                )
                file_new += int(answer.get("turns_new", 0))
                file_chunks += int(answer.get("chunks_new", 0))
                time.sleep(args.sleep)
        print(
            f"session#{label}: records={len(records)} usable={len(usable)} malformed={bad} "
            f"+turns={file_new} +chunks={file_chunks}"
        )
        totals["files"] += 1
        totals["records"] += len(records)
        totals["usable"] += len(usable)
        totals["malformed"] += bad
        totals["turns_new"] += file_new
        totals["chunks_new"] += file_chunks
    totals["seconds"] = round(time.perf_counter() - started, 1)
    print(json.dumps(totals))
    return 0


if __name__ == "__main__":
    sys.exit(main())
