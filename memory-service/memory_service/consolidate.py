"""Hierarchical dated summaries: per-day, then per-ISO-week (derived from day summaries).

Design points
  * Summaries are *derived* data: raw turns stay the source of truth, so a bad
    summary never loses information (unlike the legacy rolling blob).
  * The LLM is behind a pluggable `SummarizeFn(prompt) -> str`. The default
    `ollama_summarizer` calls Ollama; `extractive_summarizer` needs no LLM.
  * Work is planned as small units. Each unit's input is capped at
    `max_input_chars`; bigger days are map-reduced (chunk summaries -> merge).
  * Idle-only: `run()` does nothing unless the service has been quiet for
    `idle_seconds`, and it re-checks before every unit and stops as soon as
    activity appears, so it never competes with a chat turn for the model.
  * A day is only summarised once it is complete (before today UTC) or has gone
    quiet; stale summaries (input_sha changed) are rebuilt.
"""

from __future__ import annotations

import hashlib
import json
import logging
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from .service import MemoryService
from .timeutil import iso_week, utcnow

log = logging.getLogger(__name__)

SummarizeFn = Callable[[str], str]

DAY_PROMPT = """Summarize this day of conversation between a user and their AI assistant for long-term memory.
Keep: decisions, durable preferences, goals, work done and its current state, open tasks, key names/numbers.
Drop: greetings, filler, secrets/credentials. Max {limit} characters, plain sentences, no preamble.

CONVERSATION ({day}):
{body}
"""
MERGE_PROMPT = """Merge these partial summaries of one {unit} into a single summary of at most {limit} characters.
Keep decisions, preferences, state of work, open tasks. No preamble.

{body}
"""


def extractive_summarizer(prompt: str) -> str:
    """LLM-free fallback: the first ~2 lines of each user message, joined."""
    lines = [ln[2:].strip() if ln[:2] in ("u:", "a:") else ln.strip() for ln in prompt.splitlines()]
    picked = [ln[:160] for ln in lines if ln and not ln.startswith(("Summarize", "Merge", "Keep", "Drop", "CONVERSATION"))]
    return " | ".join(picked[:8])[:700] or "(no content)"


def ollama_summarizer(host: str, model: str, num_ctx: int = 8192, timeout: float = 240.0) -> SummarizeFn:
    """Summarizer using Ollama /api/chat (non-streaming, think off, small output)."""

    def run(prompt: str) -> str:
        body = json.dumps({
            "model": model, "stream": False, "think": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"num_ctx": num_ctx, "num_predict": 300, "temperature": 0.2},
        }).encode()
        req = urllib.request.Request(host.rstrip("/") + "/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        return (data.get("message", {}).get("content") or "").strip()

    return run


@dataclass
class Unit:
    level: str      # day | week
    period: str
    session_id: str
    input_sha: str
    n_turns: int
    first_turn_id: int
    last_turn_id: int
    text: str       # source text (transcript or day summaries)


def _day_text(rows, per_turn: int = 600) -> str:
    return "\n".join(f"{'u' if r['role'] == 'user' else 'a'}: {' '.join(r['content'].split())[:per_turn]}" for r in rows)


def plan(svc: MemoryService, now: datetime | None = None, min_turns: int = 2) -> list[Unit]:
    """List stale/missing day and week summaries (days first, oldest first)."""
    now = now or utcnow()
    today = now.astimezone(timezone.utc).strftime("%Y-%m-%d")
    st = svc.store
    units: list[Unit] = []
    with st.lock:
        days = st.conn.execute(
            "SELECT session_id, day, COUNT(*) n, MIN(id) a, MAX(id) b FROM turns GROUP BY session_id, day ORDER BY day"
        ).fetchall()
        have = {(r["level"], r["period"], r["session_id"]): r["input_sha"] for r in st.conn.execute(
            "SELECT level, period, session_id, input_sha FROM summaries")}
    for d in days:
        if d["day"] >= today or d["n"] < min_turns:
            continue  # today's still open (re-planned tomorrow); skip trivial days
        rows = st.turns_on_day(d["day"], d["session_id"])
        sha = hashlib.sha256("".join(r["sha"] for r in rows).encode()).hexdigest()
        if have.get(("day", d["day"], d["session_id"])) != sha:
            units.append(Unit("day", d["day"], d["session_id"], sha, d["n"], d["a"], d["b"], _day_text(rows)))

    # Weeks: built from *day summaries* of completed weeks (only after all days are summarised).
    pending_days = {(u.session_id, iso_week(u.period)) for u in units}
    with st.lock:
        ds = st.conn.execute(
            "SELECT session_id, period, text, input_sha, n_turns, first_turn_id, last_turn_id "
            "FROM summaries WHERE level='day' ORDER BY period").fetchall()
    weeks: dict[tuple[str, str], list] = {}
    for r in ds:
        weeks.setdefault((r["session_id"], iso_week(r["period"])), []).append(r)
    this_week = iso_week(today)
    for (sid, wk), items in weeks.items():
        if wk >= this_week or (sid, wk) in pending_days or len(items) < 2:
            continue
        sha = hashlib.sha256("".join(i["input_sha"] for i in items).encode()).hexdigest()
        if have.get(("week", wk, sid)) != sha:
            text = "\n".join(f"{i['period']}: {i['text']}" for i in items)
            units.append(Unit("week", wk, sid, sha, sum(i["n_turns"] for i in items),
                              min(i["first_turn_id"] for i in items), max(i["last_turn_id"] for i in items), text))
    return units


def summarize_unit(unit: Unit, fn: SummarizeFn, max_input_chars: int = 6000, out_chars: int = 700) -> str:
    """Summarise one unit; map-reduce when the input exceeds max_input_chars."""
    def one(body: str, label: str) -> str:
        if unit.level == "day":
            return fn(DAY_PROMPT.format(limit=out_chars, day=label, body=body))
        return fn(MERGE_PROMPT.format(unit="week", limit=out_chars, body=body))

    text = unit.text
    if len(text) <= max_input_chars:
        return one(text, unit.period).strip()[:out_chars * 2]
    parts, cur = [], ""
    for line in text.splitlines():
        if cur and len(cur) + len(line) + 1 > max_input_chars:
            parts.append(cur)
            cur = ""
        cur += line[:max_input_chars] + "\n"
    if cur:
        parts.append(cur)
    partial = [one(p, f"{unit.period} part {i + 1}").strip() for i, p in enumerate(parts)]
    merged = "\n".join(f"- {p}" for p in partial)
    return fn(MERGE_PROMPT.format(unit=unit.level, limit=out_chars, body=merged[:max_input_chars])).strip()[:out_chars * 2]


def run(svc: MemoryService, fn: SummarizeFn, *, now: datetime | None = None, idle_seconds: float | None = None,
        max_units: int = 3, max_input_chars: int | None = None, model_name: str = "custom",
        force: bool = False) -> dict:
    """Run up to `max_units` pending consolidations if (and only while) the service is idle."""
    s = svc.s
    idle_needed = s.idle_seconds if idle_seconds is None else idle_seconds
    cap = max_input_chars or s.consolidate_max_input_chars
    done, skipped = [], None
    todo = plan(svc, now)
    for unit in todo[:max_units] if max_units else todo:
        if not force and svc.idle_for() < idle_needed:
            skipped = f"not idle (idle_for={svc.idle_for():.0f}s < {idle_needed:.0f}s)"
            break
        text = summarize_unit(unit, fn, cap)
        if not text:
            continue
        emb = None
        with svc.store.embed_lock:
            emb = svc.embedder.embed_passages([text])[0].tobytes()
        ts = utcnow().isoformat()
        with svc.store.lock:
            svc.store.conn.execute(
                "INSERT INTO summaries(level,period,session_id,text,emb,first_turn_id,last_turn_id,n_turns,input_sha,model,created_at,updated_at)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(level,period,session_id) DO UPDATE SET "
                "text=excluded.text, emb=excluded.emb, first_turn_id=excluded.first_turn_id, last_turn_id=excluded.last_turn_id,"
                "n_turns=excluded.n_turns, input_sha=excluded.input_sha, model=excluded.model, updated_at=excluded.updated_at",
                (unit.level, unit.period, unit.session_id, text, emb, unit.first_turn_id, unit.last_turn_id,
                 unit.n_turns, unit.input_sha, model_name, ts, ts),
            )
        done.append(f"{unit.level}:{unit.period}")
    return {"planned": len(todo), "done": done, "skipped": skipped}
