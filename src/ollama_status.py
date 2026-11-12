"""Ollama host/status helpers for /status."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from time import monotonic
from typing import Any, Sequence

import httpx
from ollama import AsyncClient

from .config import OLLAMA_HOST, OLLAMA_MODEL

logger = logging.getLogger(__name__)


def _format_bytes(value: Any) -> str | None:
    """Format a byte count as a human-readable size string."""
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
    """Format an Ollama model expiry timestamp as a short relative string."""
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
    """Format one running Ollama model as a bullet line for /status."""
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


# --- read-only probe used by the /status report (GET /api/version + GET /api/ps) -------

_FOREVER_YEARS = 5  # keep-alive beyond this many years is shown as "forever"
_FRACTION_RE = re.compile(r"(\.\d{6})\d+")


@dataclass(frozen=True)
class LoadedModel:
    """One entry of Ollama's /api/ps, reduced to what /status shows."""

    name: str
    size: int
    size_vram: int
    context_length: int | None
    expires_at: datetime | None
    parameter_size: str | None
    quantization: str | None

    @property
    def gpu_percent(self) -> int:
        """Share of the model held in GPU memory (0-100)."""
        if self.size <= 0:
            return 0
        return max(0, min(100, round(100 * self.size_vram / self.size)))

    @property
    def placement(self) -> str:
        """Human label of where the weights live: GPU, CPU or a split."""
        if self.size_vram <= 0:
            return "CPU"
        if self.size_vram >= self.size:
            return "100% GPU"
        return f"{self.gpu_percent}% GPU/{100 - self.gpu_percent}% CPU"


@dataclass(frozen=True)
class OllamaProbe:
    """Result of one status probe; ``error`` is set when Ollama is unreachable."""

    version: str | None
    models: list[LoadedModel]
    ms: float
    error: str | None = None
    ps_error: str | None = None

    @property
    def reachable(self) -> bool:
        """True when at least /api/version answered."""
        return self.error is None or self.ps_error is None


def _parse_time(value: Any) -> datetime | None:
    """Parse an RFC3339 timestamp with up to nanosecond precision."""
    if not isinstance(value, str) or not value:
        return None
    text = _FRACTION_RE.sub(r"\1", value.strip()).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def parse_ps(payload: Any) -> list[LoadedModel]:
    """Parse the JSON body of GET /api/ps; malformed entries are skipped."""
    models: list[LoadedModel] = []
    raw_models = payload.get("models") if isinstance(payload, dict) else None
    for entry in raw_models if isinstance(raw_models, list) else []:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name") or entry.get("model")
        if not isinstance(name, str) or not name:
            continue
        details = entry.get("details") if isinstance(entry.get("details"), dict) else {}
        context = entry.get("context_length")
        models.append(
            LoadedModel(
                name=name,
                size=int(entry.get("size") or 0),
                size_vram=int(entry.get("size_vram") or 0),
                context_length=context if isinstance(context, int) and context > 0 else None,
                expires_at=_parse_time(entry.get("expires_at")),
                parameter_size=details.get("parameter_size") or None,
                quantization=details.get("quantization_level") or None,
            )
        )
    return models


def format_keep_alive(expires_at: datetime | None, *, now: datetime | None = None) -> str:
    """Keep-alive remaining as text; far-future expiry (OLLAMA_KEEP_ALIVE=-1) is 'forever'."""
    if expires_at is None:
        return "n/a"
    current = now or datetime.now(timezone.utc)
    seconds = (expires_at - current).total_seconds()
    if seconds > _FOREVER_YEARS * 365 * 86400:
        return "forever"
    return _format_expires(expires_at, now=current) or "n/a"


def model_matches(loaded_name: str, configured: str) -> bool:
    """True if a loaded model name is the configured one (tolerates an implicit :latest)."""

    def norm(name: str) -> str:
        """Add the implicit :latest tag so 'x' and 'x:latest' compare equal."""
        return name if ":" in name else f"{name}:latest"

    return norm(loaded_name) == norm(configured)


async def probe_ollama(host: str = OLLAMA_HOST, timeout: float = 2.0) -> OllamaProbe:
    """Read-only probe: GET /api/version and GET /api/ps run concurrently.

    Never loads, unloads or generates with a model and never raises. Failures are
    reported by category name only (no URLs/bodies).
    """
    base = host.rstrip("/")
    started = monotonic()

    async def get(client: httpx.AsyncClient, path: str) -> Any:
        """GET one endpoint and decode JSON."""
        response = await client.get(f"{base}{path}")
        response.raise_for_status()
        return response.json()

    async with httpx.AsyncClient(timeout=timeout) as client:
        version_r, ps_r = await asyncio.gather(
            get(client, "/api/version"), get(client, "/api/ps"), return_exceptions=True
        )

    def category(exc: BaseException) -> str:
        """Short, URL-free description of a probe failure."""
        if isinstance(exc, httpx.TimeoutException):
            return "timeout"
        if isinstance(exc, httpx.HTTPStatusError):
            return f"HTTP {exc.response.status_code}"
        if isinstance(exc, (httpx.ConnectError, ConnectionError)):
            return "connection refused/unreachable"
        return type(exc).__name__

    version: str | None = None
    error: str | None = None
    if isinstance(version_r, BaseException):
        error = category(version_r)
    elif isinstance(version_r, dict):
        version = str(version_r.get("version") or "unknown")
    models: list[LoadedModel] = []
    ps_error: str | None = None
    if isinstance(ps_r, BaseException):
        ps_error = category(ps_r)
    else:
        models = parse_ps(ps_r)
    return OllamaProbe(
        version=version,
        models=models,
        ms=(monotonic() - started) * 1000,
        error=error,
        ps_error=ps_error,
    )
