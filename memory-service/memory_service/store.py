"""SQLite-backed storage: immutable turn archive, chunk/FTS/vector indexes, facts."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .chunking import chunk_text
from .config import Settings
from .embedder import Embedder
from .redact import redact
from .schema import DDL, SCHEMA_VERSION
from .timeutil import parse_ts, utcnow


class EmbedModelMismatch(RuntimeError):
    """The DB was built with a different embedding model; run `reembed`."""


class VectorIndex:
    """Brute-force cosine index over a contiguous float32 matrix (numpy).

    ~1.5 KB per chunk at 384 dims: 50k chunks = 77 MB; a full scan is a single
    matrix-vector product (sub-millisecond per 10k chunks).
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.n = 0
        self._cap = 1024
        self.mat = np.zeros((self._cap, dim), dtype=np.float32)
        self.chunk_id = np.zeros(self._cap, dtype=np.int64)
        self.turn_id = np.zeros(self._cap, dtype=np.int64)
        self.ts = np.zeros(self._cap, dtype=np.float64)
        self.sess = np.zeros(self._cap, dtype=np.int32)
        self.sessions: dict[str, int] = {}

    def _grow(self, need: int) -> None:
        if self.n + need <= self._cap:
            return
        cap = max(self._cap * 2, self.n + need)
        for name in ("mat", "chunk_id", "turn_id", "ts", "sess"):
            old = getattr(self, name)
            new = np.zeros((cap,) + old.shape[1:], dtype=old.dtype)
            new[: self.n] = old[: self.n]
            setattr(self, name, new)
        self._cap = cap

    def session_code(self, session: str) -> int:
        return self.sessions.setdefault(session, len(self.sessions))

    def add(self, chunk_ids, turn_ids, ts, sessions, vecs: np.ndarray) -> None:
        k = len(chunk_ids)
        if k == 0:
            return
        self._grow(k)
        s = slice(self.n, self.n + k)
        self.mat[s] = vecs
        self.chunk_id[s] = chunk_ids
        self.turn_id[s] = turn_ids
        self.ts[s] = ts
        self.sess[s] = [self.session_code(x) for x in sessions]
        self.n += k

    def mask(self, since=None, until=None, session=None, exclude_turns=None) -> np.ndarray | None:
        n = self.n
        m = None

        def _and(cur, new):
            return new if cur is None else (cur & new)

        if since is not None:
            m = _and(m, self.ts[:n] >= since)
        if until is not None:
            m = _and(m, self.ts[:n] <= until)
        if session is not None:
            code = self.sessions.get(session)
            m = _and(m, (self.sess[:n] == code) if code is not None else np.zeros(n, dtype=bool))
        if exclude_turns:
            m = _and(m, ~np.isin(self.turn_id[:n], list(exclude_turns)))
        return m

    def top(self, q: np.ndarray, limit: int, mask: np.ndarray | None = None):
        """Return (indices, cosine scores) of the best `limit` rows, best first."""
        if self.n == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
        scores = self.mat[: self.n] @ q.astype(np.float32)
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        k = min(limit, self.n)
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        keep = np.isfinite(scores[idx])
        return idx[keep], scores[idx][keep]


def _sha(ts: str, role: str, content: str) -> str:
    return hashlib.sha256(f"{ts}\0{role}\0{content}".encode()).hexdigest()


@dataclass
class IngestResult:
    turns_seen: int = 0
    turns_new: int = 0
    chunks_new: int = 0
    redactions: int = 0


class MemoryStore:
    """Owns the SQLite connection, the in-memory vector index and all writes."""

    def __init__(self, path: str | Path, embedder: Embedder, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.embedder = embedder
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self.embed_lock = threading.Lock()  # ONNX session: one call at a time
        with self.lock:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self.conn.execute("PRAGMA foreign_keys=ON")
            self.conn.executescript(DDL)
            self._check_model()
        self.index = VectorIndex(embedder.dim)
        self._load_index()

    # ------------------------------------------------------------------ meta
    def _meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def _set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def _check_model(self) -> None:
        self._set_meta("schema_version", str(SCHEMA_VERSION)) if self._meta("schema_version") is None else None
        have = self._meta("embed_model")
        n = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        if have is None or n == 0:
            self._set_meta("embed_model", self.embedder.name)
            self._set_meta("embed_dim", str(self.embedder.dim))
        elif have != self.embedder.name:
            raise EmbedModelMismatch(
                f"DB was built with {have!r}, service configured with {self.embedder.name!r}; run `reembed`"
            )

    def _load_index(self) -> None:
        dim = self.embedder.dim
        with self.lock:
            rows = self.conn.execute(
                "SELECT id, turn_id, ts_epoch, session_id, emb FROM chunks ORDER BY id"
            ).fetchall()
        if not rows:
            return
        vecs = np.frombuffer(b"".join(r["emb"] for r in rows), dtype=np.float32).reshape(len(rows), dim)
        self.index.add(
            [r["id"] for r in rows], [r["turn_id"] for r in rows],
            [r["ts_epoch"] for r in rows], [r["session_id"] for r in rows], vecs,
        )

    # ---------------------------------------------------------------- ingest
    def add_turns(self, turns: Iterable[dict], source: str = "bot") -> IngestResult:
        """Idempotently archive + index turns.

        Each turn: {session_id, role, content, timestamp?}. Order is preserved.
        """
        res = IngestResult()
        prepared: list[dict] = []
        for t in turns:
            res.turns_seen += 1
            content = str(t.get("content", "") or "").strip()
            role = str(t.get("role", "user"))
            if not content or role not in ("user", "assistant", "system", "tool"):
                continue
            n_red = 0
            if self.settings.redact:
                content, n_red = redact(content)
            dt = parse_ts(t.get("timestamp")) or utcnow()
            ts = dt.astimezone(timezone.utc).isoformat()
            prepared.append(
                dict(session_id=str(t["session_id"]), role=role, content=content, ts=ts,
                     epoch=dt.timestamp(), day=dt.astimezone(timezone.utc).strftime("%Y-%m-%d"),
                     sha=_sha(ts, role, content), redactions=n_red)
            )
        if not prepared:
            return res

        # Skip already-archived turns before paying for embeddings.
        with self.lock:
            fresh = []
            for p in prepared:
                exists = self.conn.execute(
                    "SELECT 1 FROM turns WHERE session_id=? AND sha=?", (p["session_id"], p["sha"])
                ).fetchone()
                if not exists:
                    fresh.append(p)
        if not fresh:
            return res

        pieces: list[tuple[int, int, str]] = []  # (turn position, idx, text)
        for pos, p in enumerate(fresh):
            for idx, text in enumerate(chunk_text(p["content"], self.settings.chunk_chars)):
                pieces.append((pos, idx, text))
        with self.embed_lock:
            vecs = self.embedder.embed_passages([x[2] for x in pieces])

        by_turn: dict[int, list[int]] = {}
        for i, (pos, _idx, _t) in enumerate(pieces):
            by_turn.setdefault(pos, []).append(i)

        new_chunk_ids, new_turn_ids, new_ts, new_sess, new_rows = [], [], [], [], []
        with self.lock:
            self.conn.execute("BEGIN")
            try:
                for pos, p in enumerate(fresh):
                    seq = self.conn.execute(
                        "SELECT COALESCE(MAX(seq),0)+1 FROM turns WHERE session_id=?", (p["session_id"],)
                    ).fetchone()[0]
                    cur = self.conn.execute(
                        "INSERT OR IGNORE INTO turns(session_id,seq,ts,ts_epoch,day,role,content,sha,redactions,source)"
                        " VALUES(?,?,?,?,?,?,?,?,?,?)",
                        (p["session_id"], seq, p["ts"], p["epoch"], p["day"], p["role"], p["content"],
                         p["sha"], p["redactions"], source),
                    )
                    if cur.rowcount == 0:
                        continue
                    turn_id = cur.lastrowid
                    res.turns_new += 1
                    res.redactions += p["redactions"]
                    for i in by_turn.get(pos, []):
                        _pos, idx, text = pieces[i]
                        c = self.conn.execute(
                            "INSERT INTO chunks(turn_id,idx,text,emb,session_id,role,ts_epoch) VALUES(?,?,?,?,?,?,?)",
                            (turn_id, idx, text, vecs[i].astype(np.float32).tobytes(),
                             p["session_id"], p["role"], p["epoch"]),
                        )
                        new_chunk_ids.append(c.lastrowid)
                        new_turn_ids.append(turn_id)
                        new_ts.append(p["epoch"])
                        new_sess.append(p["session_id"])
                        new_rows.append(i)
                        res.chunks_new += 1
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise
            if new_rows:
                self.index.add(new_chunk_ids, new_turn_ids, new_ts, new_sess, vecs[new_rows])
        return res

    def ingest_jsonl(self, path: str | Path, session_id: str | None = None) -> IngestResult:
        """Ingest one bot transcript file ({timestamp, role, content} per line)."""
        path = Path(path)
        sid = session_id or path.stem
        records = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                r["session_id"] = sid
                records.append(r)
        total = IngestResult()
        for i in range(0, len(records), 32):  # bounded memory/CPU bursts
            part = self.add_turns(records[i : i + 32], source=f"jsonl:{path.name[:24]}")
            total.turns_seen += part.turns_seen
            total.turns_new += part.turns_new
            total.chunks_new += part.chunks_new
            total.redactions += part.redactions
        return total

    # ----------------------------------------------------------------- reads
    def get_turn(self, turn_id: int) -> sqlite3.Row | None:
        with self.lock:
            return self.conn.execute("SELECT * FROM turns WHERE id=?", (turn_id,)).fetchone()

    def turns_around(self, turn_id: int, window: int) -> list[sqlite3.Row]:
        with self.lock:
            t = self.conn.execute("SELECT session_id, seq FROM turns WHERE id=?", (turn_id,)).fetchone()
            if t is None:
                return []
            return self.conn.execute(
                "SELECT * FROM turns WHERE session_id=? AND seq BETWEEN ? AND ? ORDER BY seq",
                (t["session_id"], t["seq"] - window, t["seq"] + window),
            ).fetchall()

    def turns_on_day(self, day: str, session: str | None = None) -> list[sqlite3.Row]:
        with self.lock:
            if session:
                return self.conn.execute(
                    "SELECT * FROM turns WHERE day=? AND session_id=? ORDER BY ts_epoch, id", (day, session)
                ).fetchall()
            return self.conn.execute("SELECT * FROM turns WHERE day=? ORDER BY ts_epoch, id", (day,)).fetchall()

    def recent_turn_ids(self, session: str | None, n: int) -> list[int]:
        with self.lock:
            if session:
                rows = self.conn.execute(
                    "SELECT id FROM turns WHERE session_id=? ORDER BY seq DESC LIMIT ?", (session, n)
                ).fetchall()
            else:
                rows = self.conn.execute("SELECT id FROM turns ORDER BY id DESC LIMIT ?", (n,)).fetchall()
        return [r["id"] for r in rows]

    def chunks_by_ids(self, ids: Sequence[int]) -> dict[int, sqlite3.Row]:
        if not ids:
            return {}
        with self.lock:
            q = ",".join("?" * len(ids))
            rows = self.conn.execute(
                f"SELECT c.id, c.turn_id, c.idx, c.text, t.ts, t.day, t.role, t.session_id, t.seq "
                f"FROM chunks c JOIN turns t ON t.id=c.turn_id WHERE c.id IN ({q})", list(ids)
            ).fetchall()
        return {r["id"]: r for r in rows}

    def fts_search(self, match: str, limit: int, since=None, until=None, session=None, exclude_turns=None):
        """Return [(chunk_id, bm25)] best first (bm25 is lower=better, we return rank order)."""
        sql = ("SELECT c.id AS id, bm25(chunks_fts) AS s FROM chunks_fts "
               "JOIN chunks c ON c.id = chunks_fts.rowid WHERE chunks_fts MATCH ?")
        args: list = [match]
        if since is not None:
            sql += " AND c.ts_epoch >= ?"; args.append(since)
        if until is not None:
            sql += " AND c.ts_epoch <= ?"; args.append(until)
        if session is not None:
            sql += " AND c.session_id = ?"; args.append(session)
        if exclude_turns:
            sql += f" AND c.turn_id NOT IN ({','.join('?' * len(exclude_turns))})"; args.extend(exclude_turns)
        sql += " ORDER BY s LIMIT ?"; args.append(limit)
        with self.lock:
            try:
                return [(r["id"], r["s"]) for r in self.conn.execute(sql, args).fetchall()]
            except sqlite3.OperationalError:
                return []

    # ----------------------------------------------------------------- facts
    def _fact_rows(self, status: str = "active", category: str | None = None):
        sql = "SELECT * FROM facts WHERE status=?"
        args: list = [status]
        if category:
            sql += " AND category=?"; args.append(category)
        sql += " ORDER BY updated_at DESC, id DESC"
        with self.lock:
            return self.conn.execute(sql, args).fetchall()

    def add_fact(self, statement: str, category: str = "note", source_turn_ids: Sequence[int] = (),
                 session_id: str | None = None, confidence: float = 1.0) -> dict:
        """Insert a fact; dedupe near-identical, supersede near-conflicting (same category)."""
        s = self.settings
        statement = " ".join(statement.split())
        if s.redact:
            statement, _ = redact(statement)
        norm = statement.lower()
        with self.embed_lock:
            vec = self.embedder.embed_passages([statement])[0]
        now = utcnow().isoformat()
        with self.lock:
            same = self._fact_rows("active", category)
            best, best_cos = None, -1.0
            for r in same:
                if r["norm"] == norm:
                    best, best_cos = r, 1.0
                    break
                if r["emb"] is not None:
                    c = float(np.frombuffer(r["emb"], dtype=np.float32) @ vec)
                    if c > best_cos:
                        best, best_cos = r, c
            if best is not None and best_cos >= s.fact_dedupe_cos:
                self.conn.execute("UPDATE facts SET updated_at=? WHERE id=?", (now, best["id"]))
                return {"id": best["id"], "action": "exists", "superseded": None}
            cur = self.conn.execute(
                "INSERT INTO facts(category,statement,norm,emb,confidence,source_turn_ids,session_id,created_at,updated_at)"
                " VALUES(?,?,?,?,?,?,?,?,?)",
                (category, statement, norm, vec.astype(np.float32).tobytes(), confidence,
                 json.dumps(list(source_turn_ids)), session_id, now, now),
            )
            new_id = cur.lastrowid
            superseded = None
            if best is not None and best_cos >= s.fact_supersede_cos:
                self.conn.execute(
                    "UPDATE facts SET status='superseded', superseded_by=?, updated_at=? WHERE id=?",
                    (new_id, now, best["id"]),
                )
                superseded = best["id"]
            return {"id": new_id, "action": "superseded" if superseded else "added", "superseded": superseded}

    def list_facts(self, category: str | None = None, status: str = "active"):
        return self._fact_rows(status, category)

    def retract_fact(self, fact_id: int) -> bool:
        with self.lock:
            cur = self.conn.execute(
                "UPDATE facts SET status='retracted', updated_at=? WHERE id=? AND status='active'",
                (utcnow().isoformat(), fact_id),
            )
            return cur.rowcount > 0

    # ----------------------------------------------------------------- admin
    def purge_turns(self, turn_ids: Sequence[int]) -> int:
        """Audited hard delete (e.g. a secret slipped through redaction). Drops derived rows too."""
        if not turn_ids:
            return 0
        q = ",".join("?" * len(turn_ids))
        with self.lock:
            self.conn.execute("BEGIN")
            try:
                self.conn.execute("DROP TRIGGER turns_no_delete")
                self.conn.execute(f"DELETE FROM chunks WHERE turn_id IN ({q})", list(turn_ids))
                cur = self.conn.execute(f"DELETE FROM turns WHERE id IN ({q})", list(turn_ids))
                self.conn.execute(
                    "CREATE TRIGGER turns_no_delete BEFORE DELETE ON turns "
                    "BEGIN SELECT RAISE(ABORT, 'turns are immutable'); END"
                )
                self._set_meta("last_purge", utcnow().isoformat())
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise
        # Rebuild the in-memory index (rare operation).
        self.index = VectorIndex(self.embedder.dim)
        self._load_index()
        return cur.rowcount

    def reembed_all(self, batch: int = 64) -> int:
        """Re-embed every chunk with the current model (after changing EMBED_MODEL)."""
        with self.lock:
            rows = self.conn.execute("SELECT id, text FROM chunks ORDER BY id").fetchall()
        for i in range(0, len(rows), batch):
            part = rows[i : i + batch]
            with self.embed_lock:
                vecs = self.embedder.embed_passages([r["text"] for r in part])
            with self.lock:
                self.conn.execute("BEGIN")
                for r, v in zip(part, vecs):
                    self.conn.execute("UPDATE chunks SET emb=? WHERE id=?", (v.astype(np.float32).tobytes(), r["id"]))
                self.conn.execute("COMMIT")
        with self.lock:
            self._set_meta("embed_model", self.embedder.name)
            self._set_meta("embed_dim", str(self.embedder.dim))
            for tbl in ("facts", "summaries"):
                for r in self.conn.execute(f"SELECT id, {'statement' if tbl == 'facts' else 'text'} AS t FROM {tbl}").fetchall():
                    v = self.embedder.embed_passages([r["t"]])[0]
                    self.conn.execute(f"UPDATE {tbl} SET emb=? WHERE id=?", (v.astype(np.float32).tobytes(), r["id"]))
        self.index = VectorIndex(self.embedder.dim)
        self._load_index()
        return len(rows)

    def stats(self) -> dict:
        with self.lock:
            c = self.conn
            one = lambda sql: c.execute(sql).fetchone()[0]
            rng = c.execute("SELECT MIN(ts), MAX(ts) FROM turns").fetchone()
            out = {
                "turns": one("SELECT COUNT(*) FROM turns"),
                "chunks": one("SELECT COUNT(*) FROM chunks"),
                "sessions": one("SELECT COUNT(DISTINCT session_id) FROM turns"),
                "days": one("SELECT COUNT(DISTINCT day) FROM turns"),
                "facts_active": one("SELECT COUNT(*) FROM facts WHERE status='active'"),
                "facts_total": one("SELECT COUNT(*) FROM facts"),
                "summaries": one("SELECT COUNT(*) FROM summaries"),
                "first_ts": rng[0], "last_ts": rng[1],
                "redactions": one("SELECT COALESCE(SUM(redactions),0) FROM turns"),
                "embed_model": self._meta("embed_model"),
                "embed_dim": self.embedder.dim,
            }
        if self.path != ":memory:":
            p = Path(self.path)
            out["db_bytes"] = p.stat().st_size + sum(
                q.stat().st_size for q in (Path(self.path + "-wal"),) if q.exists()
            )
        out["index_mb"] = round(self.index.mat[: self.index.n].nbytes / 1e6, 2)
        return out

    def close(self) -> None:
        with self.lock:
            self.conn.close()
