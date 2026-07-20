"""Conversation-history RAG: Ollama embeddings + local SQLite store."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import sqlite3
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ollama import AsyncClient

from .config import (
    MEMORY_RECENT_TURNS,
    OLLAMA_HOST,
    RAG_EMBED_MODEL,
    RAG_MAX_CONTEXT_CHARS,
    RAG_MIN_SCORE,
    RAG_TOP_K,
)

logger = logging.getLogger(__name__)

CHUNK_SIZE = 800
EmbedFn = Callable[[Sequence[str]], Awaitable[list[list[float]]]]


def _safe_session_id(session_id: str | int) -> str:
    value = str(session_id)
    if not value or not value.lstrip("-").isdigit():
        raise ValueError("Session ID must be numeric")
    return value


def _chunk_text(text: str, size: int = CHUNK_SIZE) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    if len(cleaned) <= size:
        return [cleaned]
    return [cleaned[i : i + size] for i in range(0, len(cleaned), size)]


def _content_hash(session_id: str, role: str, content: str) -> str:
    payload = f"{session_id}\0{role}\0{content}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


class RagStore:
    """Per-session SQLite vector store for conversation chunks."""

    def __init__(
        self,
        data_dir: Path,
        *,
        embed_model: str = RAG_EMBED_MODEL,
        top_k: int = RAG_TOP_K,
        max_context_chars: int = RAG_MAX_CONTEXT_CHARS,
        min_score: float = RAG_MIN_SCORE,
        ollama_host: str = OLLAMA_HOST,
        recent_turns: int = MEMORY_RECENT_TURNS,
        embed_fn: EmbedFn | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.embed_model = embed_model
        self.top_k = top_k
        self.max_context_chars = max_context_chars
        self.min_score = min_score
        self.ollama_host = ollama_host
        self.recent_turns = recent_turns
        self._embed_fn = embed_fn
        self._locks: dict[str, asyncio.Lock] = {}
        self.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    def _db_path(self, session_id: str | int) -> Path:
        return self.data_dir / f"{_safe_session_id(session_id)}.sqlite3"

    def _lock(self, session_id: str | int) -> asyncio.Lock:
        safe_id = _safe_session_id(session_id)
        return self._locks.setdefault(safe_id, asyncio.Lock())

    def _connect(self, session_id: str | int) -> sqlite3.Connection:
        path = self._db_path(session_id)
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                embedding TEXT NOT NULL
            )
            """
        )
        conn.commit()
        os.chmod(path, 0o600)
        return conn

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._embed_fn is not None:
            return await self._embed_fn(texts)

        client = AsyncClient(host=self.ollama_host)
        response = await client.embed(model=self.embed_model, input=list(texts))
        embeddings = getattr(response, "embeddings", None) or []
        return [list(vector) for vector in embeddings]

    async def index_turn(
        self,
        session_id: str | int,
        user_message: str,
        assistant_message: str,
    ) -> int:
        """Chunk, embed, and store a completed user/assistant turn. Returns rows inserted."""
        safe_id = _safe_session_id(session_id)
        pieces: list[tuple[str, str]] = []
        for role, message in (("user", user_message), ("assistant", assistant_message)):
            for chunk in _chunk_text(message):
                pieces.append((role, chunk))
        if not pieces:
            return 0

        async with self._lock(session_id):
            conn = self._connect(session_id)
            try:
                to_insert: list[tuple[str, str, str]] = []
                for role, content in pieces:
                    digest = _content_hash(safe_id, role, content)
                    exists = conn.execute(
                        "SELECT 1 FROM chunks WHERE content_hash = ?",
                        (digest,),
                    ).fetchone()
                    if exists:
                        continue
                    to_insert.append((role, content, digest))
                if not to_insert:
                    return 0

                try:
                    vectors = await self.embed([content for _, content, _ in to_insert])
                except Exception:
                    logger.exception("RAG embed failed while indexing session %s", safe_id)
                    return 0

                if len(vectors) != len(to_insert):
                    logger.warning(
                        "RAG embed returned %d vectors for %d chunks",
                        len(vectors),
                        len(to_insert),
                    )
                    return 0

                created_at = datetime.now(timezone.utc).isoformat()
                for (role, content, digest), vector in zip(to_insert, vectors):
                    conn.execute(
                        """
                        INSERT INTO chunks
                            (session_id, role, content, content_hash, created_at, embedding)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            safe_id,
                            role,
                            content,
                            digest,
                            created_at,
                            json.dumps(vector),
                        ),
                    )
                conn.commit()
                logger.info(
                    "Indexed %d RAG chunk(s) for session %s",
                    len(to_insert),
                    safe_id,
                )
                return len(to_insert)
            finally:
                conn.close()

    async def retrieve(
        self,
        session_id: str | int,
        query: str,
        *,
        exclude_texts: Sequence[str] | None = None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return top-k relevant past chunks for the query."""
        cleaned_query = query.strip()
        if not cleaned_query:
            return []

        limit = self.top_k if top_k is None else top_k
        excluded = {text.strip() for text in (exclude_texts or []) if text and text.strip()}

        async with self._lock(session_id):
            conn = self._connect(session_id)
            try:
                rows = conn.execute(
                    "SELECT role, content, embedding FROM chunks WHERE session_id = ?",
                    (_safe_session_id(session_id),),
                ).fetchall()
            finally:
                conn.close()

        if not rows:
            return []

        try:
            query_vectors = await self.embed([cleaned_query])
        except Exception:
            logger.exception("RAG embed failed while retrieving for session %s", session_id)
            return []
        if not query_vectors:
            return []
        query_vector = query_vectors[0]

        scored: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            content = str(row["content"])
            if content in excluded:
                continue
            try:
                embedding = json.loads(row["embedding"])
            except json.JSONDecodeError:
                continue
            score = _cosine(query_vector, embedding)
            if score < self.min_score:
                continue
            scored.append(
                (
                    score,
                    {
                        "role": str(row["role"]),
                        "content": content,
                        "score": score,
                    },
                )
            )

        scored.sort(key=lambda item: item[0], reverse=True)
        selected: list[dict[str, Any]] = []
        used = 0
        for _, item in scored:
            chunk = item["content"]
            if used + len(chunk) > self.max_context_chars and selected:
                break
            remaining = self.max_context_chars - used
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                item = {**item, "content": chunk[:remaining]}
            selected.append(item)
            used += len(item["content"])
            if len(selected) >= limit:
                break
        return selected

    def format_context(self, snippets: Sequence[dict[str, Any]]) -> str:
        if not snippets:
            return ""
        lines = ["Relevant past context:"]
        for snippet in snippets:
            role = snippet.get("role", "unknown")
            content = str(snippet.get("content", "")).strip()
            if not content:
                continue
            lines.append(f"- {role}: {content}")
        return "\n".join(lines)

    async def backfill_from_transcript(
        self,
        session_id: str | int,
        records: Sequence[dict[str, Any]],
    ) -> int:
        """Index transcript records that are not already stored."""
        inserted = 0
        pending_user: str | None = None
        for record in records:
            role = str(record.get("role", ""))
            content = str(record.get("content", ""))
            if role == "user":
                pending_user = content
                continue
            if role == "assistant" and pending_user is not None:
                inserted += await self.index_turn(session_id, pending_user, content)
                pending_user = None
        return inserted

    def recent_exclude_texts(
        self,
        records: Sequence[dict[str, Any]],
        recent_turns: int | None = None,
    ) -> list[str]:
        """Contents already injected as recent turns (skip in retrieve)."""
        turns = self.recent_turns if recent_turns is None else recent_turns
        recent = list(records[-(turns * 2) :])
        return [str(record.get("content", "")) for record in recent if record.get("content")]
