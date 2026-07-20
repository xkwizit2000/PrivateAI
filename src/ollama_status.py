"""Ollama host/status helpers for /status."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Sequence

from ollama import AsyncClient

from .config import OLLAMA_HOST, OLLAMA_MODEL

logger = logging.getLogger(__name__)


def _format_bytes(value: Any) -> str | None:
    try:
        size = float(value)
    except (TypeError, ValueError):
        return None
    if size < 0:
        return None
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    index = 0
    while size >= 1024 and index < len(units) - 1:
        size /= 1024
        index += 1
    if index == 0:
        return f"{int(size)} {units[index]}"
    return f"{size:.1f} {units[index]}"


def _format_expires(expires_at: Any, *, now: datetime | None = None) -> str | None:
    if expires_at is None:
        return None
    if not isinstance(expires_at, datetime):
        return f"expires {expires_at}"

    current = now or datetime.now(expires_at.tzinfo or timezone.utc)
    if expires_at.tzinfo is None and current.tzinfo is not None:
        current = current.replace(tzinfo=None)
    elif expires_at.tzinfo is not None and current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)

    seconds = int((expires_at - current).total_seconds())
    if seconds <= 0:
        return "expiring"
    if seconds < 60:
        return f"expires in {seconds}s"
    if seconds < 3600:
        return f"expires in {seconds // 60}m"
    return f"expires in {seconds // 3600}h"


def _model_line(model: Any, *, now: datetime | None = None) -> str:
    name = getattr(model, "name", None) or getattr(model, "model", None) or "unknown"
    bits: list[str] = []

    size = _format_bytes(getattr(model, "size", None))
    if size:
        bits.append(size)
    vram = _format_bytes(getattr(model, "size_vram", None))
    if vram:
        bits.append(f"{vram} VRAM")

    context = getattr(model, "context_length", None)
    if isinstance(context, int) and context > 0:
        bits.append(f"ctx {context}")

    details = getattr(model, "details", None)
    parameter_size = getattr(details, "parameter_size", None) if details else None
    quantization = getattr(details, "quantization_level", None) if details else None
    if parameter_size:
        bits.append(str(parameter_size))
    if quantization:
        bits.append(str(quantization))

    expires = _format_expires(getattr(model, "expires_at", None), now=now)
    if expires:
        bits.append(expires)

    if not bits:
        return f"  - {name}"
    return f"  - {name} — {', '.join(bits)}"


def format_ollama_status(
    models: Sequence[Any],
    *,
    host: str = OLLAMA_HOST,
    chat_model: str = OLLAMA_MODEL,
    now: datetime | None = None,
) -> str:
    """Format configured host/model plus currently loaded models."""
    lines = [
        f"Ollama ({host}):",
        f"• chat model: {chat_model}",
    ]
    if not models:
        lines.append("• running: none loaded")
        return "\n".join(lines)

    lines.append("• running:")
    lines.extend(_model_line(model, now=now) for model in models)
    return "\n".join(lines)


async def fetch_ollama_status(
    *,
    host: str = OLLAMA_HOST,
    chat_model: str = OLLAMA_MODEL,
    timeout_seconds: float = 10.0,
) -> str:
    """Call Ollama /api/ps and return a status block for Telegram."""
    try:
        client = AsyncClient(host=host, timeout=timeout_seconds)
        response = await client.ps()
        models = list(getattr(response, "models", None) or [])
        return format_ollama_status(models, host=host, chat_model=chat_model)
    except Exception as exc:
        logger.warning("Unable to query Ollama running models at %s: %s", host, exc)
        return (
            f"Ollama ({host}):\n"
            f"• chat model: {chat_model}\n"
            f"• running: unavailable ({exc})"
        )
