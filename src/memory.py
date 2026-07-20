"""Durable, per-session conversation memory."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ollama import AsyncClient

logger = logging.getLogger(__name__)


class MemoryStore:
    """Persist transcripts and maintain a compact rolling summary."""

    def __init__(
        self,
        data_dir: Path,
        recent_turns: int = 4,
        summarize_every: int = 6,
        max_context_chars: int = 12_000,
    ) -> None:
        self.data_dir = data_dir
        self.recent_turns = recent_turns
        self.summarize_every = summarize_every
        self.max_context_chars = max_context_chars
        self._locks: dict[str, asyncio.Lock] = {}
        self.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    def _safe_session_id(self, session_id: str | int) -> str:
        value = str(session_id)
        if not value or not value.lstrip("-").isdigit():
            raise ValueError("Session ID must be numeric")
        return value

    def _paths(self, session_id: str | int) -> tuple[Path, Path]:
        safe_id = self._safe_session_id(session_id)
        return (
            self.data_dir / f"{safe_id}.jsonl",
            self.data_dir / f"{safe_id}.summary.json",
        )

    def _lock(self, session_id: str | int) -> asyncio.Lock:
        safe_id = self._safe_session_id(session_id)
        return self._locks.setdefault(safe_id, asyncio.Lock())

    @staticmethod
    def _read_summary(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {"summary": "", "summarized_messages": 0}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.exception("Unable to read memory summary at %s", path)
            return {"summary": "", "summarized_messages": 0}

    @staticmethod
    def _read_records(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []

        records: list[dict[str, Any]] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed memory record in %s", path)
        except OSError:
            logger.exception("Unable to read transcript at %s", path)
        return records

    def list_session_ids(self) -> list[str]:
        """Return numeric session IDs that have transcript files."""
        ids: list[str] = []
        for path in sorted(self.data_dir.glob("*.jsonl")):
            stem = path.stem
            if stem.lstrip("-").isdigit():
                ids.append(stem)
        return ids

    async def load_transcript(self, session_id: str | int) -> list[dict[str, Any]]:
        """Load the append-only transcript for a session."""
        transcript_path, _ = self._paths(session_id)
        async with self._lock(session_id):
            return self._read_records(transcript_path)

    async def context_messages(self, session_id: str | int) -> list[dict[str, str]]:
        """Return rolling summary and recent turns for the next model prompt."""
        transcript_path, summary_path = self._paths(session_id)
        async with self._lock(session_id):
            summary_data = self._read_summary(summary_path)
            records = self._read_records(transcript_path)

        messages: list[dict[str, str]] = []
        summary = str(summary_data.get("summary", "")).strip()
        summary_budget = self.max_context_chars // 2
        summary = summary[:summary_budget]
        if summary:
            messages.append(
                {
                    "role": "system",
                    "content": f"Durable memory from earlier conversations:\n{summary}",
                }
            )

        recent = records[-(self.recent_turns * 2) :]
        remaining = self.max_context_chars - len(summary)
        selected: deque[dict[str, str]] = deque()
        for record in reversed(recent):
            content = str(record.get("content", ""))
            if not content:
                continue
            content = content[-remaining:]
            if not content:
                break
            selected.appendleft(
                {
                    "role": str(record.get("role", "user")),
                    "content": content,
                }
            )
            remaining -= len(content)
            if remaining <= 0:
                break
        messages.extend(selected)
        return messages

    async def remember_turn(
        self,
        session_id: str | int,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """Append a completed turn to the durable transcript."""
        transcript_path, _ = self._paths(session_id)
        timestamp = datetime.now(timezone.utc).isoformat()
        records = (
            {"timestamp": timestamp, "role": "user", "content": user_message},
            {"timestamp": timestamp, "role": "assistant", "content": assistant_message},
        )

        async with self._lock(session_id):
            with transcript_path.open("a", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            os.chmod(transcript_path, 0o600)

    async def summarize_if_needed(
        self,
        session_id: str | int,
        client: AsyncClient,
        model: str,
    ) -> bool:
        """Refresh the rolling summary after enough new completed turns."""
        transcript_path, summary_path = self._paths(session_id)

        async with self._lock(session_id):
            records = self._read_records(transcript_path)
            summary_data = self._read_summary(summary_path)
            summarized_messages = int(summary_data.get("summarized_messages", 0))
            new_records = records[summarized_messages:]

            if len(new_records) < self.summarize_every * 2:
                return False

            prior_summary = str(summary_data.get("summary", "")).strip()
            summary_batch = new_records[: self.summarize_every * 2]
            transcript = "\n".join(
                f"{record.get('role', 'unknown')}: "
                f"{str(record.get('content', ''))[:4000]}"
                for record in summary_batch
            )

            prompt = f"""Update the durable memory for a personal AI assistant.

Keep only information useful in future conversations:
- user preferences and stable facts
- goals, decisions, constraints, and architecture
- work completed, current state, and unresolved tasks
- important identifiers that are not secrets

Do not store passwords, tokens, credentials, or unnecessary chat.
Write a compact factual summary. Preserve important facts from the prior summary.

PRIOR SUMMARY:
{prior_summary or "(none)"}

NEW CONVERSATION:
{transcript}
"""
            response = await client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            updated_summary = (response.message.content or "").strip()
            if not updated_summary:
                logger.warning("Model returned an empty memory summary")
                return False

            payload = {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "summarized_messages": summarized_messages + len(summary_batch),
                "summary": updated_summary,
            }
            temporary = summary_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.chmod(temporary, 0o600)
            temporary.replace(summary_path)
            logger.info("Refreshed durable memory for session %s", session_id)
            return True
