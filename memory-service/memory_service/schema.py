"""SQLite schema (single file DB, WAL). See docs/memory-service-design.md."""

SCHEMA_VERSION = 1

DDL = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

-- Immutable raw archive: one row per message ("turn"). Never updated or deleted
-- (triggers enforce it; purge_turns() is the audited escape hatch).
CREATE TABLE IF NOT EXISTS turns (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,      -- the public "turn id" (t<id>)
    session_id TEXT    NOT NULL,
    seq        INTEGER NOT NULL,                       -- order within the session
    ts         TEXT    NOT NULL,                       -- ISO-8601 UTC
    ts_epoch   REAL    NOT NULL,
    day        TEXT    NOT NULL,                       -- YYYY-MM-DD (UTC)
    role       TEXT    NOT NULL CHECK (role IN ('user','assistant','system','tool')),
    content    TEXT    NOT NULL,                       -- secrets already redacted
    sha        TEXT    NOT NULL,                       -- sha256(ts|role|content): idempotent ingest
    redactions INTEGER NOT NULL DEFAULT 0,
    source     TEXT    NOT NULL DEFAULT 'bot',
    UNIQUE (session_id, sha)
);
CREATE INDEX IF NOT EXISTS turns_session_seq ON turns (session_id, seq);
CREATE INDEX IF NOT EXISTS turns_day ON turns (day);
CREATE INDEX IF NOT EXISTS turns_ts ON turns (ts_epoch);

CREATE TRIGGER IF NOT EXISTS turns_no_update BEFORE UPDATE ON turns
BEGIN SELECT RAISE(ABORT, 'turns are immutable'); END;
CREATE TRIGGER IF NOT EXISTS turns_no_delete BEFORE DELETE ON turns
BEGIN SELECT RAISE(ABORT, 'turns are immutable'); END;

-- Retrieval units: a turn is split into >=1 chunks; each chunk has a float32 embedding.
CREATE TABLE IF NOT EXISTS chunks (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    turn_id    INTEGER NOT NULL REFERENCES turns(id),
    idx        INTEGER NOT NULL,
    text       TEXT    NOT NULL,
    emb        BLOB    NOT NULL,                       -- dim * 4 bytes, L2-normalised
    session_id TEXT    NOT NULL,
    role       TEXT    NOT NULL,
    ts_epoch   REAL    NOT NULL,
    UNIQUE (turn_id, idx)
);
CREATE INDEX IF NOT EXISTS chunks_turn ON chunks (turn_id);

-- Keyword index (external-content FTS5, porter stemming).
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, content='chunks', content_rowid='id', tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;

-- Curated durable facts (mutable, versioned by supersession; nothing is deleted).
CREATE TABLE IF NOT EXISTS facts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    category        TEXT    NOT NULL,                  -- profile|preference|decision|task|note
    statement       TEXT    NOT NULL,
    norm            TEXT    NOT NULL,                  -- lower-cased, whitespace-collapsed
    emb             BLOB,
    confidence      REAL    NOT NULL DEFAULT 1.0,
    status          TEXT    NOT NULL DEFAULT 'active'  -- active|superseded|retracted
                    CHECK (status IN ('active','superseded','retracted')),
    superseded_by   INTEGER REFERENCES facts(id),
    source_turn_ids TEXT    NOT NULL DEFAULT '[]',     -- JSON array of turn ids
    session_id      TEXT,
    created_at      TEXT    NOT NULL,
    updated_at      TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS facts_cat_status ON facts (category, status);

-- Hierarchical dated summaries: day -> week (-> month). Derived data, always
-- rebuildable from `turns`; input_sha detects stale summaries.
CREATE TABLE IF NOT EXISTS summaries (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    level         TEXT    NOT NULL CHECK (level IN ('day','week','month')),
    period        TEXT    NOT NULL,                    -- 2026-08-04 | 2026-W32 | 2026-08
    session_id    TEXT    NOT NULL,
    text          TEXT    NOT NULL,
    emb           BLOB,
    first_turn_id INTEGER,
    last_turn_id  INTEGER,
    n_turns       INTEGER NOT NULL DEFAULT 0,
    input_sha     TEXT    NOT NULL,
    model         TEXT    NOT NULL,
    created_at    TEXT    NOT NULL,
    updated_at    TEXT    NOT NULL,
    UNIQUE (level, period, session_id)
);
"""
