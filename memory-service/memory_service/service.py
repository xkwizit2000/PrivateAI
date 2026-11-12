"""Retrieval + tool-level operations (shared by MCP, HTTP and CLI front ends)."""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from .config import Settings
from .gate import should_inject
from .store import MemoryStore
from .timeutil import parse_when, utcnow

STOPWORDS = frozenset(
    """a an the and or but if then so of to in on at by for with about as is are was were be been being am do does did
    have has had i me my we our you your he she it its they them their this that these those what which who whom when where
    why how can could would should will shall may might not no yes just also than too very from into over under again
    there here out up down off any some all more most other such only own same s t don now tell said say know remember
    earlier before ago did""".split()
)
_WORD = re.compile(r"[A-Za-z0-9_]+")
FACT_CATEGORIES = ("profile", "preference", "decision", "task", "note")


def keyword_tokens(query: str) -> list[str]:
    toks = [t.lower() for t in _WORD.findall(query)]
    keep = [t for t in toks if t not in STOPWORDS and len(t) > 1]
    return keep or [t for t in toks if len(t) > 1]


def fts_match_expr(query: str) -> str | None:
    toks = keyword_tokens(query)
    if not toks:
        return None
    return " OR ".join(f'"{t}"' for t in dict.fromkeys(toks))


def make_snippet(text: str, terms: list[str], limit: int) -> str:
    """Collapse whitespace; if too long, window around the first query-term hit."""
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    low = flat.lower()
    pos = min((p for p in (low.find(t) for t in terms) if p >= 0), default=0)
    start = max(0, min(pos - limit // 4, len(flat) - limit))
    piece = flat[start : start + limit].strip()
    return ("…" if start > 0 else "") + piece + ("…" if start + limit < len(flat) else "")


@dataclass
class Hit:
    turn_id: int
    chunk_id: int
    session_id: str
    day: str
    role: str
    text: str
    score: float  # final fused score
    vec_cos: float | None = None
    vec_rank: int | None = None
    fts_rank: int | None = None


@dataclass
class SearchResult:
    hits: list[Hit]
    top_cos: float | None
    facts: list[dict] = field(default_factory=list)
    summaries: list[dict] = field(default_factory=list)
    ms: float = 0.0
    embed_ms: float = 0.0


class MemoryService:
    def __init__(self, store: MemoryStore, settings: Settings | None = None) -> None:
        self.store = store
        self.s = settings or store.settings
        self.embedder = store.embedder
        self.last_activity = time.monotonic()

    def touch(self) -> None:
        self.last_activity = time.monotonic()

    def idle_for(self) -> float:
        return time.monotonic() - self.last_activity

    # --------------------------------------------------------------- search
    def search_chunks(
        self,
        query: str,
        k: int = 5,
        since: str | None = None,
        until: str | None = None,
        session: str | None = None,
        mode: str = "hybrid",
        exclude_turns=None,
        now: datetime | None = None,
        recency: bool = True,
    ) -> SearchResult:
        t0 = time.perf_counter()
        s = self.s
        now = now or utcnow()
        k = max(1, min(int(k or 5), 10))
        since_e = parse_when(since, now=now)
        until_e = parse_when(until, end=True, now=now)
        sess = session or None
        excl = set(exclude_turns or [])
        ncand = max(s.candidates, k * 10)

        vec_hits: list[tuple[int, float]] = []  # (index row, cosine)
        top_cos: float | None = None
        embed_ms = 0.0
        if mode in ("hybrid", "vector"):
            te = time.perf_counter()
            with self.store.embed_lock:
                q = self.embedder.embed_query(query)
            embed_ms = (time.perf_counter() - te) * 1000
            with self.store.lock:
                idx = self.store.index
                mask = idx.mask(since_e, until_e, sess, excl or None)
                rows, scores = idx.top(q, ncand, mask)
                vec_hits = [(int(idx.chunk_id[r]), float(sc)) for r, sc in zip(rows, scores)]
            top_cos = vec_hits[0][1] if vec_hits else None

        fts_hits: list[tuple[int, float]] = []
        if mode in ("hybrid", "keyword"):
            expr = fts_match_expr(query)
            if expr:
                fts_hits = self.store.fts_search(expr, ncand, since_e, until_e, sess, excl or None)

        # Reciprocal rank fusion over chunk ids.
        fused: dict[int, float] = {}
        vec_rank: dict[int, int] = {}
        fts_rank: dict[int, int] = {}
        vec_cos: dict[int, float] = {}
        for rank, (cid, cos) in enumerate(vec_hits):
            fused[cid] = fused.get(cid, 0.0) + s.vec_weight / (s.rrf_k + rank + 1)
            vec_rank[cid], vec_cos[cid] = rank + 1, cos
        for rank, (cid, _bm) in enumerate(fts_hits):
            fused[cid] = fused.get(cid, 0.0) + s.fts_weight / (s.rrf_k + rank + 1)
            fts_rank[cid] = rank + 1
        if not fused:
            return SearchResult([], top_cos, ms=(time.perf_counter() - t0) * 1000, embed_ms=embed_ms)

        chunks = self.store.chunks_by_ids(list(fused))
        terms = keyword_tokens(query)
        best_per_turn: dict[int, Hit] = {}
        for cid, sc in fused.items():
            row = chunks.get(cid)
            if row is None:
                continue
            if recency and s.recency_weight > 0:
                ts = datetime.fromisoformat(row["ts"])
                age_days = max(0.0, (now - ts).total_seconds() / 86400.0)
                sc *= 1.0 + s.recency_weight * math.exp(-age_days / s.recency_tau_days)
            cur = best_per_turn.get(row["turn_id"])
            if cur is None or sc > cur.score:
                best_per_turn[row["turn_id"]] = Hit(
                    turn_id=row["turn_id"], chunk_id=cid, session_id=row["session_id"], day=row["day"],
                    role=row["role"], text=row["text"], score=sc, vec_cos=vec_cos.get(cid),
                    vec_rank=vec_rank.get(cid), fts_rank=fts_rank.get(cid),
                )
        # Drop exact-duplicate texts (e.g. a retried message) - they only waste output budget.
        hits, seen = [], set()
        for h in sorted(best_per_turn.values(), key=lambda h: -h.score):
            key = " ".join(h.text.lower().split())
            if key in seen:
                continue
            seen.add(key)
            hits.append(h)
            if len(hits) >= k:
                break
        for h in hits:
            h.text = make_snippet(h.text, terms, s.search_snippet_chars)
        return SearchResult(hits, top_cos, ms=(time.perf_counter() - t0) * 1000, embed_ms=embed_ms)

    def _small_rank(self, table: str, qvec: np.ndarray, where: str, args: tuple, min_cos: float, limit: int):
        with self.store.lock:
            rows = self.store.conn.execute(
                f"SELECT * FROM {table} WHERE emb IS NOT NULL AND {where}", args
            ).fetchall()
        scored = []
        for r in rows:
            c = float(np.frombuffer(r["emb"], dtype=np.float32) @ qvec)
            if c >= min_cos:
                scored.append((c, r))
        scored.sort(key=lambda x: -x[0])
        return scored[:limit]

    def search(self, query: str, k: int = 5, since=None, until=None, session=None, mode: str = "hybrid",
               exclude_turns=None, include_facts: bool = True, now: datetime | None = None) -> SearchResult:
        """Full search: turn chunks (hybrid) + a couple of matching facts/summaries."""
        res = self.search_chunks(query, k, since, until, session, mode, exclude_turns, now)
        if include_facts and mode != "keyword" and query.strip():
            with self.store.embed_lock:
                q = self.embedder.embed_query(query)
            for c, r in self._small_rank("facts", q, "status='active'", (), self.s.small_min_cos, 2):
                res.facts.append({"id": r["id"], "category": r["category"], "statement": r["statement"], "cos": c})
            if not since and not until:
                for c, r in self._small_rank("summaries", q, "1=1", (), self.s.small_min_cos, 1):
                    res.summaries.append({"level": r["level"], "period": r["period"], "text": r["text"], "cos": c})
        return res

    # ------------------------------------------------------------ formatting
    def format_search(self, res: SearchResult, max_chars: int | None = None) -> str:
        cap = max_chars or self.s.search_max_chars
        lines: list[str] = []
        used = 0

        def add(line: str) -> bool:
            nonlocal used
            if used + len(line) + 1 > cap:
                return False
            lines.append(line)
            used += len(line) + 1
            return True

        for f in res.facts:
            add(f"[fact#{f['id']} {f['category']}] {f['statement'][:300]}")
        for sm in res.summaries:
            add(f"[{sm['level']} summary {sm['period']}] {sm['text'][:400]}")
        omitted = 0
        for h in res.hits:
            head = f"[t{h.turn_id} {h.day} {h.role}] "
            room = cap - used - len(head) - 1
            if room < 60:
                omitted += 1
                continue
            body = h.text if len(h.text) <= room else h.text[: room - 1] + "…"
            add(head + body)
        if omitted:
            add(f"(+{omitted} more omitted; refine query)")
        return "\n".join(lines) if lines else "No matching memory."

    # ------------------------------------------------------- get_conversation
    def get_conversation(self, around_turn_id: int | None = None, date: str | None = None,
                         window: int = 3, session: str | None = None, max_chars: int | None = None) -> str:
        cap = max_chars or self.s.conversation_max_chars
        window = max(0, min(int(window or 3), 8))
        out: list[str] = []
        if around_turn_id:
            rows = self.store.turns_around(int(around_turn_id), window)
            if not rows:
                return f"No turn t{around_turn_id}."
            focus = int(around_turn_id)
        elif date:
            day = (date or "").strip()[:10]
            rows = self.store.turns_on_day(day, session or None)
            with self.store.lock:
                sm = self.store.conn.execute(
                    "SELECT level, period, text FROM summaries WHERE level='day' AND period=? ORDER BY id LIMIT 1", (day,)
                ).fetchone()
            if sm:
                out.append(f"[day summary {day}] {sm['text'][:900]}")
            if not rows and not sm:
                return f"No conversation on {day}."
            focus = -1
        else:
            return "Give around_turn_id or date."
        per = max(120, (cap - sum(len(x) + 1 for x in out)) // max(1, len(rows)) - 20)
        if focus > 0:
            per = max(per, 600)
        used = sum(len(x) + 1 for x in out)
        shown = 0
        for r in rows:
            lim = 900 if r["id"] == focus else per
            body = " ".join(r["content"].split())
            if len(body) > lim:
                body = body[: lim - 1] + "…"
            line = f"[t{r['id']} {r['ts'][:16]} {r['role']}] {body}"
            if used + len(line) + 1 > cap:
                out.append(f"(+{len(rows) - shown} more turns; narrow the date or use around_turn_id)")
                break
            out.append(line)
            used += len(line) + 1
            shown += 1
        return "\n".join(out)

    # -------------------------------------------------------------- facts
    def remember_fact(self, statement: str, category: str = "note", source_turn_ids=(), session: str | None = None) -> str:
        statement = (statement or "").strip()
        if len(statement) < 3:
            return "Error: empty statement."
        statement = statement[:500]
        category = (category or "note").strip().lower()
        if category not in FACT_CATEGORIES:
            category = "note"
        r = self.store.add_fact(statement, category, source_turn_ids, session)
        if r["action"] == "exists":
            return f"already known: fact#{r['id']}"
        if r["action"] == "superseded":
            return f"saved fact#{r['id']} (replaces fact#{r['superseded']})"
        return f"saved fact#{r['id']}"

    def format_facts(self, category: str | None = None, max_chars: int | None = None) -> str:
        cap = max_chars or self.s.facts_max_chars
        cat = (category or "").strip().lower() or None
        rows = self.store.list_facts(cat)
        if not rows:
            return "No facts stored." if not cat else f"No {cat} facts."
        lines, used = [], 0
        for r in rows:
            line = f"fact#{r['id']} [{r['category']}] {r['statement']} ({r['updated_at'][:10]})"
            if used + len(line) + 1 > cap:
                lines.append(f"(+{len(rows) - len(lines)} more; pass a category)")
                break
            lines.append(line)
            used += len(line) + 1
        return "\n".join(lines)

    def profile_text(self, max_chars: int = 600) -> str:
        """Compact always-on fact profile (for the bot's tiny core prompt)."""
        rows = self.store.list_facts(None)
        order = {c: i for i, c in enumerate(FACT_CATEGORIES)}
        rows = sorted(rows, key=lambda r: order.get(r["category"], 99))
        parts, used = [], 0
        for r in rows:
            piece = f"- {r['statement']}"
            if used + len(piece) + 1 > max_chars:
                break
            parts.append(piece)
            used += len(piece) + 1
        return "\n".join(parts)

    # --------------------------------------------------------------- stats
    def format_stats(self) -> str:
        st = self.store.stats()
        return (
            f"turns={st['turns']} chunks={st['chunks']} sessions={st['sessions']} days={st['days']} "
            f"facts={st['facts_active']} summaries={st['summaries']} "
            f"range={str(st['first_ts'] or '-')[:10]}..{str(st['last_ts'] or '-')[:10]} "
            f"model={st['embed_model']} db={st.get('db_bytes', 0) // 1024}KB"
        )

    # -------------------------------------------------------------- recall
    def recall(self, query: str, session: str | None = None, exclude_recent: int = 8, max_hits: int = 2,
               max_chars: int | None = None, now: datetime | None = None) -> dict:
        """Gated auto-injection: returns inject=False (empty text) unless the gate fires."""
        s = self.s
        cap = max_chars or s.recall_max_chars
        excl = self.store.recent_turn_ids(session, exclude_recent) if exclude_recent else []
        res = self.search_chunks(query, k=max_hits, session=session, exclude_turns=excl, now=now)
        ok, reason = should_inject(query, res.top_cos, s.gate_cue_min_cos, s.gate_strong_cos)
        text = ""
        if ok and res.hits:
            per = max(100, cap // max_hits - 30)
            parts = []
            for h in res.hits[:max_hits]:
                parts.append(f"[t{h.turn_id} {h.day} {h.role}] {h.text[:per]}")
            text = "\n".join(parts)[:cap]
        return {
            "inject": bool(text), "reason": reason, "text": text, "top_cos": res.top_cos,
            "hits": [{"turn_id": h.turn_id, "day": h.day, "role": h.role, "score": round(h.score, 5)} for h in res.hits],
            "ms": round(res.ms, 1),
        }

    def core(self, session: str, recent_turns: int = 4, max_chars: int = 2400) -> dict:
        """Tiny always-on core: fact profile + last few raw turns (compact)."""
        profile = self.profile_text(600)
        ids = self.store.recent_turn_ids(session, recent_turns * 2)[::-1]
        turns, used = [], len(profile)
        for tid in ids:
            r = self.store.get_turn(tid)
            body = r["content"]
            if used + len(body) > max_chars:
                body = body[-max(0, max_chars - used):]
            if not body:
                continue
            turns.append({"id": r["id"], "role": r["role"], "content": body})
            used += len(body)
        return {"profile": profile, "turns": turns}
