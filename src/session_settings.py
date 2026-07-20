"""Per-chat session settings persisted under data/."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from .config import OLLAMA_THINK, OLLAMA_TIMEOUT_SECONDS
from .ids import safe_session_id

logger = logging.getLogger(__name__)

MIN_TIMEOUT_SECONDS = 30.0
MAX_TIMEOUT_SECONDS = 3600.0


class SessionSettings:
    """Store per-session overrides such as think mode and timeout."""

    def __init__(
        self,
        path: Path,
        *,
        default_think: bool = OLLAMA_THINK,
        default_timeout: float = OLLAMA_TIMEOUT_SECONDS,
    ) -> None:
        self.path = path
        self.default_think = default_think
        self.default_timeout = float(default_timeout)
        self._data: dict[str, dict[str, Any]] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._load()

    def _safe_id(self, session_id: str | int) -> str:
        return safe_session_id(session_id)

    def _load(self) -> None:
        if not self.path.exists():
            self._data = {}
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                self._data = {
                    str(key): value
                    for key, value in raw.items()
                    if isinstance(value, dict)
                }
            else:
                self._data = {}
        except (OSError, json.JSONDecodeError):
            logger.exception("Unable to load session settings from %s", self.path)
            self._data = {}

    def _save(self) -> None:
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        temporary.replace(self.path)

    def _entry(self, session_id: str | int) -> dict[str, Any]:
        return dict(self._data.get(self._safe_id(session_id), {}))

    def get_think(self, session_id: str | int) -> bool:
        value = self._entry(session_id).get("think")
        if isinstance(value, bool):
            return value
        return self.default_think

    def set_think(self, session_id: str | int, enabled: bool) -> bool:
        safe_id = self._safe_id(session_id)
        entry = self._entry(session_id)
        entry["think"] = enabled
        self._data[safe_id] = entry
        self._save()
        return enabled

    def get_verbose(self, session_id: str | int) -> bool:
        value = self._entry(session_id).get("verbose")
        if isinstance(value, bool):
            return value
        return False

    def set_verbose(self, session_id: str | int, enabled: bool) -> bool:
        safe_id = self._safe_id(session_id)
        entry = self._entry(session_id)
        entry["verbose"] = enabled
        self._data[safe_id] = entry
        self._save()
        return enabled

    def get_timeout(self, session_id: str | int) -> float:
        value = self._entry(session_id).get("timeout_seconds")
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
        return self.default_timeout

    def set_timeout(self, session_id: str | int, seconds: float) -> float:
        if seconds < MIN_TIMEOUT_SECONDS or seconds > MAX_TIMEOUT_SECONDS:
            raise ValueError(
                f"Timeout must be between {int(MIN_TIMEOUT_SECONDS)} "
                f"and {int(MAX_TIMEOUT_SECONDS)} seconds"
            )
        safe_id = self._safe_id(session_id)
        entry = self._entry(session_id)
        entry["timeout_seconds"] = float(seconds)
        self._data[safe_id] = entry
        self._save()
        return float(seconds)

    def clear_timeout(self, session_id: str | int) -> float:
        safe_id = self._safe_id(session_id)
        entry = self._entry(session_id)
        entry.pop("timeout_seconds", None)
        if entry:
            self._data[safe_id] = entry
        else:
            self._data.pop(safe_id, None)
        self._save()
        return self.default_timeout

    def status_items(self, session_id: str | int) -> list[tuple[str, str]]:
        """Return (name, display_value) for every session setting.

        Add new settings here so /status stays complete.
        """
        think = self.get_think(session_id)
        verbose = self.get_verbose(session_id)
        timeout = self.get_timeout(session_id)
        return [
            (
                "think",
                f"{'on' if think else 'off'} "
                f"(default {'on' if self.default_think else 'off'})",
            ),
            ("verbose", "on" if verbose else "off"),
            (
                "timeout",
                f"{int(timeout)}s (default {int(self.default_timeout)}s)",
            ),
        ]

    def format_status(self, session_id: str | int) -> str:
        """Human-readable status of all settings for a chat."""
        items = self.status_items(session_id)
        lines = [f"• {name}: {value}" for name, value in items]
        commands = ", ".join(f"/{name}" for name, _ in items)
        return "Session settings:\n" + "\n".join(lines) + f"\n\nChange with: {commands}"
