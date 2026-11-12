"""/status report: health + configuration of every component the agent needs.

Layout
------
* ``collect_*``  async collectors, one per section. Each reads one thing (Ollama's
  HTTP API, the MCP hub, the memory service, the legacy memory files, the host
  exporter's JSON, the cgroup files ...) and returns a :class:`Section` made of
  :class:`Item` lines carrying a level (ok/info/na/warn/fail).
* ``build_status_report``  runs the collectors concurrently under per-collector
  timeouts (``asyncio.gather(return_exceptions=True)``), so one failing probe never
  breaks /status: it shows ``n/a (reason)`` and counts as a WARN.
* ``render_report``  pure formatting + verdict (``Overall: OK|DEGRADED|DOWN``).

Rules: read-only (never loads/unloads/generates with a model, never writes), plain text,
no secrets and no identifiers (secret env values and token-like strings are scrubbed
from the final text as defence in depth; failures are shown by category only).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from . import config as app_config
from .ollama_status import (
    LoadedModel,
    OllamaProbe,
    _format_bytes,
    format_keep_alive,
    model_matches,
    probe_ollama,
)
from .runtime_stats import MODEL_STATS, PROCESS_STARTED_EPOCH, PROCESS_STARTED_MONOTONIC

logger = logging.getLogger(__name__)

OK, INFO, NA, WARN, FAIL = "ok", "info", "na", "warn", "fail"
_RANK = {OK: 0, INFO: 0, NA: 0, WARN: 1, FAIL: 2}
COMPACT_TARGET_CHARS = 3000
_VERSION_FILE = app_config.PROJECT_ROOT / "VERSION"


# ----------------------------------------------------------------------- data model
@dataclass
class Item:
    """One line of a section."""

    text: str
    level: str = INFO
    full: bool = False  # shown only in /status full
    issue: str | list[str] | None = None  # short text(s) for the top verdict (default ``text``)
    critical: bool = False  # a FAIL that means the agent cannot work => verdict DOWN


@dataclass
class Section:
    """A titled group of items; its marker is the worst item level."""

    name: str
    head: str = ""
    items: list[Item] = field(default_factory=list)

    @property
    def level(self) -> str:
        """Worst level among the items (ok when nothing is wrong)."""
        worst = OK
        for item in self.items:
            if _RANK[item.level] > _RANK[worst]:
                worst = item.level
        return worst


@dataclass
class StatusConfig:
    """Everything /status needs from configuration (defaults come from src.config)."""

    ollama_host: str = app_config.OLLAMA_HOST
    ollama_model: str = app_config.OLLAMA_MODEL
    num_ctx: int | None = app_config.OLLAMA_NUM_CTX
    stuck_warn_seconds: float = app_config.OLLAMA_STUCK_WARN_SECONDS
    chat_backend: str = app_config.CHAT_BACKEND or "telegram"
    memory_backend: str = app_config.MEMORY_BACKEND
    memory_recall: str = app_config.MEMORY_RECALL
    memory_dual_write: bool = app_config.MEMORY_DUAL_WRITE
    memory_tools: str = app_config.MEMORY_TOOLS
    memory_service_url: str = app_config.MEMORY_SERVICE_URL
    rag_enabled: bool = app_config.RAG_ENABLED
    rag_dir: Path = app_config.RAG_DIR
    rag_embed_model: str = app_config.RAG_EMBED_MODEL
    rag_embed_num_gpu: int | None = app_config.RAG_EMBED_NUM_GPU
    tool_routing: str = app_config.TOOL_ROUTING
    tool_followup_seconds: float = app_config.TOOL_ROUTING_FOLLOWUP_SECONDS
    markers: str = app_config.STATUS_MARKERS
    http_timeout: float = app_config.STATUS_HTTP_TIMEOUT
    collector_timeout: float = app_config.STATUS_COLLECTOR_TIMEOUT
    host_file: Path = app_config.STATUS_HOST_FILE
    host_stale_seconds: float = app_config.STATUS_HOST_STALE_SECONDS
    load_warn_ratio: float = app_config.STATUS_LOAD_WARN_RATIO
    cpu_warn_percent: float = app_config.STATUS_CPU_WARN_PERCENT
    mem_available_warn_percent: float = app_config.STATUS_MEM_AVAILABLE_WARN_PERCENT
    swap_warn_percent: float = app_config.STATUS_SWAP_WARN_PERCENT
    disk_warn_percent: float = app_config.STATUS_DISK_WARN_PERCENT
    disk_fail_percent: float = app_config.STATUS_DISK_FAIL_PERCENT
    vram_warn_percent: float = app_config.STATUS_VRAM_WARN_PERCENT
    cpu_temp_warn_c: float = app_config.STATUS_CPU_TEMP_WARN_C
    gpu_temp_warn_c: float = app_config.STATUS_GPU_TEMP_WARN_C
    watchdog_restarts_warn: int = app_config.STATUS_WATCHDOG_RESTARTS_WARN
    container_mem_warn_percent: float = app_config.STATUS_CONTAINER_MEM_WARN_PERCENT
    loop_lag_warn_ms: float = app_config.STATUS_LOOP_LAG_WARN_MS
    model_error_window_seconds: float = app_config.STATUS_MODEL_ERROR_WINDOW_SECONDS
    memory_backlog_warn: int = app_config.STATUS_MEMORY_BACKLOG_WARN
    summary_stale_days: float = app_config.STATUS_SUMMARY_STALE_DAYS
    dual_write_lag_seconds: float = app_config.STATUS_DUAL_WRITE_LAG_SECONDS
    dual_write_missing_warn: int = app_config.STATUS_DUAL_WRITE_MISSING_WARN
    cgroup_dir: Path = Path("/sys/fs/cgroup")
    version_file: Path = _VERSION_FILE
    cpu_sample_seconds: float = 0.4


@dataclass
class StatusContext:
    """Live components handed to the collectors (any of them may be None)."""

    cfg: StatusConfig
    settings: Any = None  # SessionSettings
    session_id: str | int = ""
    memory: Any = None  # MemoryStore | CoreMemoryStore
    rag: Any = None  # RagStore
    hub: Any = None  # McpHub
    memory_service: Any = None  # ServiceMemory
    full: bool = False
    now: float = field(default_factory=time.time)
    legacy_task: "asyncio.Future[LegacyScan] | None" = None


# ------------------------------------------------------------------------ formatting
def fmt_duration(seconds: float | None) -> str:
    """Compact duration: 45s, 12m, 1h12m, 3d4h."""
    if seconds is None or seconds < 0:
        return "n/a"
    total = int(seconds)
    if total < 60:
        return f"{total}s"
    if total < 3600:
        return f"{total // 60}m"
    if total < 86400:
        return f"{total // 3600}h{(total % 3600) // 60:02d}m"
    return f"{total // 86400}d{(total % 86400) // 3600}h"


def fmt_age(epoch: float | None, now: float) -> str:
    """'3m ago' style age of a wall-clock time."""
    if epoch is None:
        return "never"
    return f"{fmt_duration(max(0.0, now - epoch))} ago"


def fmt_bytes(value: Any) -> str:
    """Byte count (reuses the Ollama status helper), 'n/a' when unknown."""
    return _format_bytes(value) or "n/a"


def fmt_pct(value: float | None, digits: int = 0) -> str:
    """Percent string or 'n/a'."""
    return "n/a" if value is None else f"{value:.{digits}f}%"


def safe_host(url: str) -> str:
    """scheme-less ``host:port`` of a URL: no credentials, path or query."""
    if not url:
        return "n/a"
    try:
        parts = urlsplit(url if "//" in url else f"//{url}")
        host = parts.hostname or "n/a"
        return f"{host}:{parts.port}" if parts.port else host
    except ValueError:
        return "n/a"


def epoch_from_iso(value: Any) -> float | None:
    """Parse an ISO timestamp (Z or offset, ns precision tolerated) to epoch seconds."""
    if not isinstance(value, str) or not value:
        return None
    text = re.sub(r"(\.\d{6})\d+", r"\1", value.strip()).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _num(value: Any) -> float | None:
    """Float for real numbers (bool excluded), else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


# ------------------------------------------------------------------- secret hygiene
_TOKEN_PATTERNS = (
    re.compile(r"\b\d{8,12}:[A-Za-z0-9_-]{30,}\b"),  # Telegram bot token
    re.compile(r"\b05[0-9a-fA-F]{64}\b"),  # Session ID
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\b[A-Fa-f0-9]{40,}\b"),  # long hex (keys/hashes)
    re.compile(r"\b[A-Za-z0-9_-]{48,}\b"),  # long opaque tokens
    re.compile(r"(?i)://[^/\s:@]+:[^/\s@]+@"),  # URL credentials
)


def _secret_values() -> list[str]:
    """Secret config values (len >= 8) to scrub from the output."""
    candidates: list[Any] = [
        app_config.TELEGRAM_TOKEN,
        app_config.SESSION_MNEMONIC,
        app_config.MEMORY_TOKEN,
        app_config.BRIDGE_TOKEN,
        app_config.ALLOWED_USER_ID,
        *app_config.SESSION_ALLOWED_IDS,
    ]
    values = {str(item).strip() for item in candidates if item}
    return sorted((v for v in values if len(v) >= 8), key=len, reverse=True)


def redact_secrets(text: str, extra: list[str] | None = None) -> str:
    """Scrub secret values and token-like strings from the final text."""
    for value in [*_secret_values(), *(extra or [])]:
        if value and len(value) >= 8:
            text = text.replace(value, "[redacted]")
    for pattern in _TOKEN_PATTERNS:
        text = pattern.sub("[redacted]", text)
    return text


# ------------------------------------------------------------------- legacy memory
@dataclass
class SessionScan:
    """Counts for one legacy session (no text, no id)."""

    records: int = 0
    summarized: int = 0
    summary_chars: int = 0
    summary_epoch: float | None = None


@dataclass
class LegacyScan:
    """Aggregate scan of the legacy transcript/summary files."""

    sessions: dict[str, SessionScan] = field(default_factory=dict)
    newest_record_epoch: float | None = None
    error: str | None = None

    @property
    def records(self) -> int:
        """Total transcript records over all sessions."""
        return sum(s.records for s in self.sessions.values())

    @property
    def summarized(self) -> int:
        """Total records already folded into summaries."""
        return sum(min(s.summarized, s.records) for s in self.sessions.values())

    @property
    def backlog_max(self) -> int:
        """Largest unsummarized backlog of any session."""
        return max((max(0, s.records - s.summarized) for s in self.sessions.values()), default=0)


def scan_legacy_memory(data_dir: Path | None) -> LegacyScan:
    """Blocking scan of ``*.jsonl`` / ``*.summary.json`` (run it in a thread)."""
    scan = LegacyScan()
    if data_dir is None:
        scan.error = "no memory store"
        return scan
    try:
        paths = sorted(Path(data_dir).glob("*.jsonl"))
    except OSError as exc:
        scan.error = type(exc).__name__
        return scan
    for path in paths:
        stem = path.name[: -len(".jsonl")]
        entry = SessionScan()
        last_line = ""
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if line.strip():
                        entry.records += 1
                        last_line = line
        except OSError:
            continue
        try:
            stamp = epoch_from_iso(json.loads(last_line).get("timestamp"))
            if stamp is not None and (
                scan.newest_record_epoch is None or stamp > scan.newest_record_epoch
            ):
                scan.newest_record_epoch = stamp
        except (ValueError, AttributeError):
            pass
        summary_path = path.with_name(f"{stem}.summary.json")
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            entry.summarized = int(data.get("summarized_messages") or 0)
            entry.summary_chars = len(str(data.get("summary") or ""))
            entry.summary_epoch = epoch_from_iso(data.get("updated_at"))
        except (OSError, ValueError, AttributeError, TypeError):
            pass
        scan.sessions[stem] = entry
    return scan


def scan_rag(rag_dir: Path) -> tuple[int, int, int]:
    """Blocking (databases, chunks, bytes) of the RAG stores, opened read-only."""
    databases = chunks = size = 0
    for path in sorted(Path(rag_dir).glob("*.sqlite3")):
        try:
            size += path.stat().st_size
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=0.5)
            try:
                chunks += conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            finally:
                conn.close()
            databases += 1
        except (OSError, sqlite3.Error):
            continue
    return databases, chunks, size


# ------------------------------------------------------------------------ collectors
def _raise(level: str, issues: list[str], new_level: str, text: str) -> str:
    """Record an issue on a multi-check line; returns the worse of the two levels."""
    issues.append(text)
    return new_level if _RANK[new_level] > _RANK[level] else level


def _read_text(path: Path) -> str | None:
    """Small file -> stripped text, None when unreadable."""
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def read_version(path: Path) -> tuple[str, str]:
    """(commit, branch) from the build-time VERSION file; 'unknown' fallbacks."""
    commit = branch = "unknown"
    text = _read_text(path)
    for line in (text or "").splitlines():
        key, _, value = line.partition("=")
        value = value.strip()
        if key.strip() == "commit" and value:
            commit = value[:12]
        elif key.strip() == "branch" and value:
            branch = value[:60]
    return commit, branch


@dataclass(frozen=True)
class CgroupSample:
    """Raw cgroup v2 numbers of this container."""

    mem_current: int | None
    mem_max: int | None  # None = unlimited
    cpu_usage_usec: int | None
    pids: int | None
    oom_kills: int | None
    cpu_quota_cores: float | None  # None = unlimited


def read_cgroup(directory: Path) -> CgroupSample | None:
    """Parse cgroup v2 files; None when the cgroup v2 hierarchy is not visible."""
    if not (directory / "memory.current").exists() and not (directory / "cpu.stat").exists():
        return None

    def int_of(name: str) -> int | None:
        """First integer of a cgroup file; 'max' and unreadable become None."""
        text = _read_text(directory / name)
        if not text:
            return None
        token = text.split()[0]
        return int(token) if token.isdigit() else None

    usage = None
    for line in (_read_text(directory / "cpu.stat") or "").splitlines():
        key, _, value = line.partition(" ")
        if key == "usage_usec" and value.strip().isdigit():
            usage = int(value)
    oom = None
    for line in (_read_text(directory / "memory.events") or "").splitlines():
        key, _, value = line.partition(" ")
        if key == "oom_kill" and value.strip().isdigit():
            oom = int(value)
    quota = None
    cpu_max = (_read_text(directory / "cpu.max") or "").split()
    if len(cpu_max) == 2 and cpu_max[0].isdigit() and cpu_max[1].isdigit() and int(cpu_max[1]):
        quota = int(cpu_max[0]) / int(cpu_max[1])
    return CgroupSample(
        mem_current=int_of("memory.current"),
        mem_max=int_of("memory.max"),
        cpu_usage_usec=usage,
        pids=int_of("pids.current"),
        oom_kills=oom,
        cpu_quota_cores=quota,
    )


async def measure_loop_lag(samples: int = 4, interval: float = 0.05) -> float:
    """Worst scheduling overshoot (ms) of short sleeps = event-loop responsiveness."""
    worst = 0.0
    for _ in range(samples):
        started = time.monotonic()
        await asyncio.sleep(interval)
        worst = max(worst, (time.monotonic() - started - interval) * 1000)
    return worst


def _proc_self() -> dict[str, int | None]:
    """RSS (bytes), thread and open-fd counts of this process (full view)."""
    out: dict[str, int | None] = {"rss": None, "threads": None, "fds": None}
    status = _read_text(Path("/proc/self/status")) or ""
    for line in status.splitlines():
        key, _, value = line.partition(":")
        parts = value.split()
        if key == "VmRSS" and parts and parts[0].isdigit():
            out["rss"] = int(parts[0]) * 1024
        elif key == "Threads" and parts and parts[0].isdigit():
            out["threads"] = int(parts[0])
    try:
        out["fds"] = len(list(Path("/proc/self/fd").iterdir()))
    except OSError:
        pass
    return out


async def collect_agent(ctx: StatusContext) -> Section:
    """AGENT: uptime, version, backend, container memory/CPU, loop lag, background work."""
    cfg = ctx.cfg
    uptime = time.monotonic() - PROCESS_STARTED_MONOTONIC
    commit, branch = read_version(cfg.version_file)
    version = commit if branch == "unknown" else f"{commit} ({branch})"
    section = Section("AGENT", f"up {fmt_duration(uptime)} · {cfg.chat_backend} · v {version}")

    first = read_cgroup(cfg.cgroup_dir)
    started = time.monotonic()
    lag_ms, _ = await asyncio.gather(
        measure_loop_lag(), asyncio.sleep(cfg.cpu_sample_seconds if first else 0)
    )
    second = read_cgroup(cfg.cgroup_dir)
    window = time.monotonic() - started

    bits: list[str] = []
    level = INFO
    issue = None
    if second is None:
        bits.append("container metrics n/a (cgroup v2 files not visible)")
    else:
        if second.mem_current is not None:
            limit = "no limit" if second.mem_max is None else fmt_bytes(second.mem_max)
            bits.append(f"mem {fmt_bytes(second.mem_current)} / {limit}")
            if second.mem_max:
                used_pct = 100.0 * second.mem_current / second.mem_max
                if used_pct >= cfg.container_mem_warn_percent:
                    level, issue = WARN, f"bot memory {used_pct:.0f}% of limit"
        if second.cpu_usage_usec is not None:
            avg = 100.0 * second.cpu_usage_usec / 1e6 / max(uptime, 1.0)
            now_txt = ""
            if first and first.cpu_usage_usec is not None and window > 0:
                now_pct = 100.0 * (second.cpu_usage_usec - first.cpu_usage_usec) / 1e6 / window
                now_txt = f"{max(now_pct, 0):.0f}% now / "
            bits.append(f"cpu {now_txt}{avg:.0f}% avg (of 1 core)")
    bits.append(f"loop lag {lag_ms:.0f} ms")
    if lag_ms >= cfg.loop_lag_warn_ms:
        level, issue = WARN, f"event loop lag {lag_ms:.0f} ms"
    tasks = len(asyncio.all_tasks())
    bg = getattr(ctx.memory_service, "pending_tasks", 0) if ctx.memory_service else 0
    bits.append(f"{tasks} tasks ({bg} memory bg)")
    section.items.append(Item("container: " + " · ".join(bits), level, issue=issue))

    if second is not None and second.oom_kills:
        section.items.append(
            Item(f"container OOM kills: {second.oom_kills}", WARN, issue="bot container OOM-killed")
        )
    proc = _proc_self()
    section.items.append(
        Item(
            f"process: rss {fmt_bytes(proc['rss'])} · threads {proc['threads']} · "
            f"fds {proc['fds']} · container pids {second.pids if second else 'n/a'} · "
            f"started {datetime.fromtimestamp(PROCESS_STARTED_EPOCH, timezone.utc):%Y-%m-%d %H:%MZ}",
            full=True,
        )
    )
    if second is not None and second.cpu_quota_cores:
        section.items.append(Item(f"cpu quota: {second.cpu_quota_cores:g} cores", full=True))
    return section


def _model_desc(model: LoadedModel, now: datetime) -> str:
    """Placement/size/context/keep-alive of one loaded model."""
    bits = [model.placement, fmt_bytes(model.size)]
    if model.context_length:
        bits.append(f"ctx {model.context_length}")
    bits.append(f"keep-alive {format_keep_alive(model.expires_at, now=now)}")
    return ", ".join(bits)


def _ollama_items(ctx: StatusContext, probe: OllamaProbe) -> list[Item]:
    """Build the Ollama section lines from a probe result (pure, testable)."""
    cfg = ctx.cfg
    now_dt = datetime.fromtimestamp(ctx.now, timezone.utc)
    items: list[Item] = []
    if not probe.reachable:
        items.append(
            Item(
                f"UNREACHABLE: {probe.error or 'no answer'} - the agent cannot answer",
                FAIL,
                issue="Ollama unreachable",
                critical=True,
            )
        )
    else:
        if probe.ps_error:
            items.append(
                Item(f"/api/ps failed ({probe.ps_error}); model state unknown", WARN,
                     issue="Ollama /api/ps failed")
            )
        else:
            chat = next((m for m in probe.models if model_matches(m.name, cfg.ollama_model)), None)
            if chat is None:
                items.append(
                    Item(f"chat {cfg.ollama_model}: not loaded (loads on next message; first reply slower)")
                )
            else:
                level, issue = OK, None
                text = f"chat {chat.name}: {_model_desc(chat, now_dt)}"
                if chat.size_vram < chat.size:
                    level, issue = WARN, f"chat model not fully on GPU ({chat.placement})"
                if cfg.num_ctx and chat.context_length and chat.context_length != cfg.num_ctx:
                    level = WARN
                    issue = f"loaded ctx {chat.context_length} != OLLAMA_NUM_CTX {cfg.num_ctx}"
                    text += f" (configured {cfg.num_ctx})"
                elif not cfg.num_ctx:
                    text += " (OLLAMA_NUM_CTX unset: model default)"
                items.append(Item(text, level, issue=issue))
            if cfg.rag_enabled and cfg.memory_backend != "service":
                embed = next(
                    (m for m in probe.models if model_matches(m.name, cfg.rag_embed_model)), None
                )
                want = "CPU" if cfg.rag_embed_num_gpu == 0 else "Ollama's choice"
                if embed is None:
                    items.append(Item(f"embed {cfg.rag_embed_model}: not loaded (num_gpu "
                                      f"{cfg.rag_embed_num_gpu}: {want})"))
                else:
                    level, issue = OK, None
                    if cfg.rag_embed_num_gpu == 0 and embed.size_vram > 0:
                        level, issue = WARN, "embed model on GPU although RAG_EMBED_NUM_GPU=0"
                    items.append(
                        Item(f"embed {embed.name}: {_model_desc(embed, now_dt)} (num_gpu "
                             f"{cfg.rag_embed_num_gpu})", level, issue=issue)
                    )
            extras = [
                m for m in probe.models
                if not model_matches(m.name, cfg.ollama_model)
                and not (cfg.rag_enabled and model_matches(m.name, cfg.rag_embed_model))
            ]
            if extras:
                items.append(
                    Item("other loaded: " + ", ".join(m.name for m in extras[:4]), INFO)
                )
            for model in probe.models:
                items.append(
                    Item(
                        f"loaded {model.name}: {model.parameter_size or '?'} "
                        f"{model.quantization or '?'}, vram {fmt_bytes(model.size_vram)} of "
                        f"{fmt_bytes(model.size)}",
                        full=True,
                    )
                )
        items.append(Item(f"probe: GET /api/version + /api/ps in {probe.ms:.0f} ms", full=True))

    # Bot-side call tracking (metadata only).
    stats = MODEL_STATS
    durations = stats.recent_durations()
    if stats.total_calls:
        last = stats.last_success
        avg = sum(durations) / len(durations) if durations else None
        text = "chat calls: "
        text += f"last ok {last.seconds:.1f}s ({fmt_age(last.finished_epoch, ctx.now)})" if last else "no success yet"
        if avg is not None:
            text += f" · avg {avg:.1f}s (n={len(durations)})"
        text += f" · errors {stats.total_errors}/{stats.total_calls} · retries {stats.total_retries}"
        items.append(Item(text))
    else:
        items.append(Item("chat calls: none since bot start"))
    error = stats.last_error
    if error is not None:
        recent = ctx.now - error.finished_epoch <= cfg.model_error_window_seconds
        items.append(
            Item(
                f"last model error: {error.outcome} after {error.seconds:.0f}s "
                f"({fmt_age(error.finished_epoch, ctx.now)})",
                WARN if recent else INFO,
                issue="recent chat-model error" if recent else None,
            )
        )
    retry = stats.last_retry_epoch
    if retry is not None:
        items.append(Item(f"last connection retry: {fmt_age(retry, ctx.now)}", full=True))
    age = stats.oldest_in_flight_seconds()
    if age is not None:
        stuck = age >= cfg.stuck_warn_seconds
        items.append(
            Item(
                f"chat call in flight for {age:.0f}s" + (" - model may be stuck" if stuck else ""),
                WARN if stuck else INFO,
                issue="chat call possibly stuck" if stuck else None,
            )
        )
    return items


async def collect_ollama(ctx: StatusContext) -> Section:
    """OLLAMA / MODEL: reachability, loaded models, placement, context, call stats."""
    probe = await probe_ollama(ctx.cfg.ollama_host, ctx.cfg.http_timeout)
    head = f"{probe.version or 'unreachable'} · {safe_host(ctx.cfg.ollama_host)}"
    return Section("OLLAMA", head, _ollama_items(ctx, probe))


async def collect_tools(ctx: StatusContext) -> Section:
    """TOOLS / MCP: per-server state, exposed vs hidden tools, allowlist and routing."""
    cfg = ctx.cfg
    hub = ctx.hub
    if hub is None:
        return Section("TOOLS", "n/a", [Item("MCP hub not available", NA)])
    snap = hub.status_snapshot()
    section = Section("TOOLS", f"{snap['tools_total']} exposed · routing {cfg.tool_routing}")
    if not snap["servers"]:
        text = "no MCP servers configured (chat-only mode)"
        if not snap["config_found"]:
            text = "no MCP config file found (chat-only mode)"
        section.items.append(Item(text))
    for server in snap["servers"]:
        name, state = server["name"], server["state"]
        counts = f"{len(server['exposed'])} exposed, {len(server['hidden'])} hidden"
        base = f"{name} ({server['transport']}): {state}"
        if state == "connected":
            section.items.append(Item(f"{base}, {counts}", OK))
        elif state == "disabled":
            section.items.append(Item(f"{base} ({server['group']} tools off: MEMORY_TOOLS=off)"))
        else:
            error = f" ({server['error']})" if server.get("error") else ""
            section.items.append(
                Item(f"{base}{error}, {counts}", WARN, issue=f"MCP server '{name}' {state}")
            )
        if server["exposed"]:
            section.items.append(Item(f"{name} tools: " + ", ".join(server["exposed"]), full=True))
        if server["hidden"]:
            section.items.append(Item(f"{name} hidden: " + ", ".join(server["hidden"]), full=True))
    allow = snap["allowlist"] + (
        f" ({snap['allowlist_size']})" if snap["allowlist_size"] is not None else ""
    )
    trim = f"{snap['desc_max_chars'] or 'off'}/{snap['param_desc_max_chars'] or 'off'}"
    section.items.append(
        Item(
            f"allowlist: {allow} · readonly {'on' if snap['readonly'] else 'off'} · "
            f"desc trim {trim} chars · MEMORY_TOOLS {cfg.memory_tools}"
        )
    )
    section.items.append(
        Item(
            f"routing: default {cfg.tool_routing} (per chat: /tools) · follow-up window "
            f"{fmt_duration(cfg.tool_followup_seconds)} · {snap['hidden_total']} hidden total",
            full=True,
        )
    )
    return section


def _recall_avg_ms(service: Any) -> float | None:
    """Average recall latency (ms) over the client's ring buffer."""
    times = list(getattr(service, "_recall_ms", []) or [])
    return sum(times) / len(times) if times else None


async def collect_memory(ctx: StatusContext) -> Section:
    """MEMORY: service health + counters, dual-write currency, legacy summary/RAG state."""
    cfg = ctx.cfg
    service = ctx.memory_service
    service_backend = cfg.memory_backend == "service" and service is not None
    dual = "on" if cfg.memory_dual_write or service_backend else "off"
    legacy_task = ctx.legacy_task
    probe_task = asyncio.ensure_future(service.probe(cfg.http_timeout)) if service else None
    rag_task = (
        asyncio.ensure_future(asyncio.to_thread(scan_rag, cfg.rag_dir))
        if ctx.rag is not None
        else None
    )
    try:
        legacy: LegacyScan | None = await asyncio.shield(legacy_task) if legacy_task else None
        probe = await probe_task if probe_task else None
        rag_stats = await rag_task if rag_task else None
    finally:
        for task in (probe_task, rag_task):
            if task is not None and not task.done():
                task.cancel()

    section = Section("MEMORY", f"backend {cfg.memory_backend} · recall {cfg.memory_recall} · dual-write {dual}")
    now = ctx.now

    # --- memory service
    if service is None:
        section.items.append(Item("service: not configured (MEMORY_SERVICE_URL empty); legacy memory only"))
    else:
        host = safe_host(cfg.memory_service_url)
        breaker = service.breaker_state
        if probe and probe["healthy"]:
            stats = probe["stats"] or {}
            db = f" · DB {fmt_bytes(stats['db_bytes'])}" if stats.get("db_bytes") else ""
            last_ts = epoch_from_iso(stats.get("last_ts"))
            section.items.append(
                Item(
                    f"service {host}: reachable ({probe['ms']:.0f} ms) · {stats.get('turns', '?')} turns · "
                    f"{stats.get('chunks', '?')} chunks · {stats.get('sessions', '?')} sessions · "
                    f"{stats.get('facts_active', '?')} facts{db} · last turn {fmt_age(last_ts, now)}",
                    OK,
                )
            )
            if stats:
                section.items.append(
                    Item(
                        f"service detail: days {stats.get('days')} · summaries {stats.get('summaries')} · "
                        f"embed {stats.get('embed_model')} dim {stats.get('embed_dim')} · "
                        f"index {stats.get('index_mb')} MB · redactions {stats.get('redactions')}",
                        full=True,
                    )
                )
        else:
            error = probe["error"] if probe else "no probe"
            section.items.append(
                Item(
                    f"service {host}: UNREACHABLE ({error})",
                    FAIL,
                    issue="memory service unreachable",
                    critical=service_backend,
                )
            )
        failures = getattr(service, "_failures", 0)
        if breaker == "closed":
            section.items.append(Item(f"breaker: closed ({failures} consecutive failures)"))
        else:
            section.items.append(
                Item(
                    f"breaker: {breaker} (retry in {service.breaker_retry_in():.0f}s)",
                    WARN,
                    issue=f"memory breaker {breaker}",
                )
            )
        c = service.counters
        avg = _recall_avg_ms(service)
        avg_txt = f"avg {avg:.0f} ms" if avg is not None else "avg n/a"
        section.items.append(
            Item(
                f"since start: appends ok {c['append_ok']}/failed {c['append_fail']} · recall calls "
                f"{c['recall_calls']} (gate fired {c['recall_fired']}, skipped {c['recall_skipped']}, "
                f"errors {c['recall_errors']}, injected {c['recall_injected']}) · {avg_txt} · "
                f"last append {fmt_age(service.last_append_epoch, now)}"
            )
        )
        section.items.append(Item(f"recall stats: {service.shadow_summary()}", full=True))

    # --- dual-write currency against the legacy transcript
    if service is not None and dual == "on" and legacy is not None and not legacy.error:
        if probe and probe["healthy"] and probe["stats"]:
            stats = probe["stats"]
            turns = int(stats.get("turns") or 0)
            missing = legacy.records - turns
            last_ts = epoch_from_iso(stats.get("last_ts"))
            lag = None
            if legacy.newest_record_epoch is not None and last_ts is not None:
                lag = legacy.newest_record_epoch - last_ts
            drift = missing > cfg.dual_write_missing_warn or (
                lag is not None and lag > cfg.dual_write_lag_seconds
            )
            detail = f"legacy {legacy.records} records vs service {turns} turns (diff {missing:+d}"
            if lag is not None:
                detail += f", newest legacy record {max(lag, 0):.0f}s ahead of service"
            detail += ")"
            if drift:
                section.items.append(
                    Item(f"dual-write DRIFT: {detail}", WARN, issue="memory dual-write drift")
                )
            else:
                section.items.append(Item(f"dual-write current: {detail}", OK))
    elif service is not None and dual == "off":
        section.items.append(Item("dual-write: off (service does not receive new turns)"))

    # --- legacy memory / summary
    if legacy is None or legacy.error:
        section.items.append(
            Item(f"legacy memory: n/a ({legacy.error if legacy else 'not scanned'})", WARN,
                 issue="legacy memory state unavailable")
        )
    elif service_backend:
        section.items.append(
            Item(f"legacy transcript: {legacy.records} records in {len(legacy.sessions)} sessions "
                 "(system of record; no rolling summary on the service backend)")
        )
    else:
        stale_days = cfg.summary_stale_days
        backlog = legacy.backlog_max
        level, issue = OK, None
        if backlog > cfg.memory_backlog_warn:
            level, issue = WARN, f"summary backlog {backlog} > {cfg.memory_backlog_warn}"
        newest_summary = max(
            (s.summary_epoch for s in legacy.sessions.values() if s.summary_epoch), default=None
        )
        for scan in legacy.sessions.values():
            old = scan.summary_epoch is not None and now - scan.summary_epoch > stale_days * 86400
            if old and scan.records - scan.summarized > 0:
                level, issue = WARN, f"summary older than {stale_days:g}d with backlog"
        chars = sum(s.summary_chars for s in legacy.sessions.values())
        section.items.append(
            Item(
                f"legacy: {legacy.records} records/{len(legacy.sessions)} sessions · summarized "
                f"{legacy.summarized} · backlog {backlog} (warn >{cfg.memory_backlog_warn}) · "
                f"summary {chars} chars, updated {fmt_age(newest_summary, now)}",
                level,
                issue=issue,
            )
        )

    # --- RAG
    if ctx.rag is None:
        why = "service backend" if service_backend else "RAG_ENABLED=0"
        section.items.append(Item(f"RAG: off ({why})"))
    else:
        where = "CPU" if cfg.rag_embed_num_gpu == 0 else "Ollama decides"
        if rag_stats:
            dbs, chunks, size = rag_stats
            section.items.append(
                Item(f"RAG: on · {cfg.rag_embed_model} (num_gpu {cfg.rag_embed_num_gpu} = {where}) · "
                     f"{chunks} chunks in {dbs} index(es), {fmt_bytes(size)}")
            )
        else:
            section.items.append(Item(f"RAG: on · {cfg.rag_embed_model} (num_gpu {cfg.rag_embed_num_gpu})"))
    return section


def _disk_items(host: dict[str, Any], cfg: StatusConfig) -> list[Item]:
    """Disk lines: one per distinct filesystem, with usage thresholds."""
    items: list[Item] = []
    seen: dict[str, str] = {}
    parts: list[str] = []
    level, issues = OK, []
    for disk in host.get("disks") or []:
        if not isinstance(disk, dict):
            continue
        label, mount, pct = disk.get("label"), disk.get("mount"), _num(disk.get("pct"))
        if mount in seen:
            parts.append(f"{label}: same volume as {seen[mount]}")
            continue
        seen[mount] = str(label)
        used, total = _num(disk.get("used")), _num(disk.get("total"))
        parts.append(f"{label} {fmt_bytes(used)}/{fmt_bytes(total)} ({fmt_pct(pct)})")
        if pct is not None and pct >= cfg.disk_fail_percent:
            level = _raise(level, issues, FAIL, f"disk {label} {pct:.0f}% full")
        elif pct is not None and pct >= cfg.disk_warn_percent:
            level = _raise(level, issues, WARN, f"disk {label} {pct:.0f}% full")
    if parts:
        items.append(Item("disk: " + " · ".join(parts), level, issue=issues or None))
    return items


def _host_items(host: dict[str, Any], cfg: StatusConfig, now: float) -> tuple[str, list[Item]]:
    """Render the host.json content (pure; tolerant of missing fields)."""
    items: list[Item] = []
    cpu = host.get("cpu") if isinstance(host.get("cpu"), dict) else {}
    load = cpu.get("load") if isinstance(cpu.get("load"), list) and len(cpu["load"]) == 3 else None
    count = _num(cpu.get("count"))
    usage = _num(cpu.get("usage_pct"))
    head_bits = [f"up {fmt_duration(_num(host.get('uptime_s')))}"]
    if load:
        head_bits.append(f"load {load[0]}/{load[1]}/{load[2]}" + (f" ({int(count)} cpu)" if count else ""))
    level, issues = OK, []
    if load and count:
        ratio = float(load[1]) / count
        if ratio >= cfg.load_warn_ratio:
            level = _raise(level, issues, WARN, f"load5 {load[1]} on {int(count)} cpus")
    if usage is not None and usage >= cfg.cpu_warn_percent:
        level = _raise(level, issues, WARN, f"host cpu {usage:.0f}%")
    items.append(Item(f"cpu: {fmt_pct(usage)} busy · load5/cpu {float(load[1]) / count:.2f}"
                      if load and count else f"cpu: {fmt_pct(usage)} busy", level, issue=issues or None))

    mem = host.get("mem") if isinstance(host.get("mem"), dict) else {}
    total, avail = _num(mem.get("total")), _num(mem.get("available"))
    if total and avail is not None:
        avail_pct = 100.0 * avail / total
        swap_total, swap_free = _num(mem.get("swap_total")) or 0, _num(mem.get("swap_free")) or 0
        swap_used = swap_total - swap_free
        level, issues = OK, []
        if avail_pct < cfg.mem_available_warn_percent:
            level = _raise(level, issues, WARN, f"host memory {avail_pct:.0f}% available")
        swap_pct = 100.0 * swap_used / swap_total if swap_total else 0.0
        if swap_total and swap_pct >= cfg.swap_warn_percent:
            level = _raise(level, issues, WARN, f"swap {swap_pct:.0f}% used")
        items.append(
            Item(
                f"mem: {fmt_bytes(total - avail)}/{fmt_bytes(total)} used · avail {fmt_bytes(avail)} "
                f"({avail_pct:.0f}%) · swap {fmt_bytes(swap_used)}/{fmt_bytes(swap_total)}",
                level,
                issue=issues or None,
            )
        )
    items.extend(_disk_items(host, cfg))

    gpu = host.get("gpu") if isinstance(host.get("gpu"), dict) else None
    temps = host.get("temps") if isinstance(host.get("temps"), dict) else {}
    level, issues = OK, []
    bits: list[str] = []
    if gpu:
        bits.append(f"gpu {gpu.get('busy_pct', 'n/a')}% busy")
        used, vt = _num(gpu.get("vram_used")), _num(gpu.get("vram_total"))
        if used is not None and vt:
            bits.append(f"VRAM {fmt_bytes(used)}/{fmt_bytes(vt)}")
            if 100.0 * used / vt >= cfg.vram_warn_percent:
                level = _raise(level, issues, WARN, f"VRAM {100.0 * used / vt:.0f}% used")
    else:
        bits.append("gpu n/a")
    tbits = []
    for key, label, limit in (("cpu_c", "cpu", cfg.cpu_temp_warn_c), ("gpu_c", "gpu", cfg.gpu_temp_warn_c),
                              ("nvme_c", "nvme", None)):
        value = _num(temps.get(key))
        if value is not None:
            tbits.append(f"{label} {value:.0f}C")
            if limit is not None and value >= limit:
                level = _raise(level, issues, WARN, f"{label} temperature {value:.0f}C")
    if tbits:
        bits.append("temp " + " ".join(tbits))
    items.append(Item(" · ".join(bits), level, issue=issues or None))

    units = host.get("units") if isinstance(host.get("units"), dict) else {}
    states = units.get("states") if isinstance(units.get("states"), dict) else {}
    if states:
        level, issues = OK, []
        bits = []
        for unit, state in states.items():
            short = str(unit).replace(".service", "").replace("ollama-watchdog.timer", "watchdog-timer")
            bits.append(f"{short} {state}")
            if state != "active":
                critical = unit in {"ollama.service", "docker.service"}
                level = _raise(level, issues, FAIL if critical else WARN, f"{short} {state}")
        items.append(Item("units: " + " · ".join(bits), level, issue=issues or None))
    failed = units.get("failed") if isinstance(units.get("failed"), list) else []
    if failed:
        # Informational: unrelated failed units must not make the agent look degraded.
        items.append(Item("failed systemd units: " + ", ".join(str(n) for n in failed[:5])))

    containers = host.get("containers")
    if isinstance(containers, list) and containers:
        level, issues = OK, []
        bits = []
        for entry in containers:
            if not isinstance(entry, dict):
                continue
            state, health = str(entry.get("state")), str(entry.get("health"))
            text = f"{entry.get('name')} {state}" + (f" ({health})" if health not in ("none", "None") else "")
            bits.append(text)
            if state != "running" or health == "unhealthy":
                level = _raise(level, issues, FAIL, f"container {entry.get('name')} {state}/{health}")
            elif entry.get("oom_killed"):
                level = _raise(level, issues, WARN, f"container {entry.get('name')} was OOM-killed")
        items.append(Item("containers: " + " · ".join(bits), level, issue=issues or None))

    wd = host.get("watchdog") if isinstance(host.get("watchdog"), dict) else None
    if wd:
        restarts = int(_num(wd.get("restarts_24h")) or 0)
        level = WARN if restarts >= cfg.watchdog_restarts_warn else OK
        issue = f"Ollama restarted {restarts}x in 24h (watchdog)" if level == WARN else None
        last_restart = epoch_from_iso(wd.get("last_restart_at"))
        text = f"ollama watchdog: {restarts} restart(s)/24h"
        if restarts:
            text += f", last {fmt_age(last_restart, now)}"
        items.append(Item(text, level, issue=issue))
        last = wd.get("last_line") if isinstance(wd.get("last_line"), dict) else None
        if last:
            items.append(Item(f"watchdog last run: {str(last.get('text'))[:110]} "
                              f"({fmt_age(epoch_from_iso(last.get('ts')), now)})", full=True))
        for event in (wd.get("events") or [])[-5:]:
            if isinstance(event, dict):
                items.append(Item(f"watchdog event {event.get('ts')}: {str(event.get('text'))[:100]}", full=True))
        items.append(Item(f"watchdog counters 24h: hung {wd.get('hung_24h')} · cooldown skips "
                          f"{wd.get('cooldown_24h')}", full=True))
    if host.get("reboot_required"):
        items.append(Item("reboot pending (/var/run/reboot-required)"))
    exporter = host.get("exporter") if isinstance(host.get("exporter"), dict) else {}
    generated = _num(host.get("generated_epoch"))
    items.append(
        Item(f"exporter: data {fmt_duration(now - generated) if generated else 'n/a'} old · run took "
             f"{exporter.get('duration_ms', 'n/a')} ms · host {host.get('hostname', 'n/a')}", full=True)
    )
    errors = host.get("errors") if isinstance(host.get("errors"), dict) else {}
    for name in sorted(errors):
        items.append(Item(f"host probe '{name}': n/a ({str(errors[name])[:60]})", WARN,
                          issue=f"host probe {name} failed"))
    return " · ".join(head_bits), items


def load_host_file(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Read host.json: (data, None) or (None, reason) for missing/unreadable/corrupt."""
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None, "file missing"
    except OSError as exc:
        return None, type(exc).__name__
    try:
        data = json.loads(raw)
    except ValueError:
        return None, "corrupt JSON"
    if not isinstance(data, dict):
        return None, "unexpected format"
    return data, None


async def collect_host(ctx: StatusContext) -> Section:
    """HOST SYSTEM: render the exporter's JSON, or degrade to 'n/a (exporter not running)'."""
    cfg = ctx.cfg
    host, reason = await asyncio.to_thread(load_host_file, cfg.host_file)
    if host is not None:
        generated = _num(host.get("generated_epoch")) or epoch_from_iso(host.get("generated_at"))
        if generated is None:
            host, reason = None, "no timestamp"
        elif ctx.now - generated > cfg.host_stale_seconds:
            host, reason = None, f"data {fmt_duration(ctx.now - generated)} old"
    if host is None:
        return Section(
            "HOST",
            "n/a",
            [
                Item(
                    f"n/a (exporter not running: {reason}); install ops/pai-status or see README",
                    WARN,
                    issue="host metrics unavailable (exporter not running)",
                )
            ],
        )
    head, items = _host_items(host, cfg, ctx.now)
    return Section("HOST", head, items)


async def collect_session(ctx: StatusContext) -> Section:
    """THIS CHAT: per-chat settings and how much history feeds the prompt (no ids)."""
    cfg = ctx.cfg
    section = Section("THIS CHAT")
    if ctx.settings is not None:
        values = dict(ctx.settings.status_items(ctx.session_id))
        section.items.append(
            Item(" · ".join(f"{name} {value}" for name, value in values.items()))
        )
    else:
        section.items.append(Item("settings n/a", NA))
    legacy = await asyncio.shield(ctx.legacy_task) if ctx.legacy_task else None
    if legacy is not None and ctx.memory is not None:
        try:
            from .ids import safe_session_id

            entry = legacy.sessions.get(safe_session_id(ctx.session_id))
        except ValueError:
            entry = None
        records = entry.records if entry else 0
        recent = getattr(ctx.memory, "recent_turns", 4) * 2
        if cfg.memory_backend == "service":
            feed = f"prompt gets the last {min(records, recent)} records + service fact profile"
        else:
            summary = f" + summary {entry.summary_chars} chars" if entry and entry.summary_chars else ""
            feed = f"prompt gets the last {min(records, recent)} records{summary}"
        section.items.append(Item(f"history: {records} records · {feed}"))
    return section


COLLECTORS: tuple[tuple[str, Callable[[StatusContext], Awaitable[Section]]], ...] = (
    ("AGENT", collect_agent),
    ("OLLAMA", collect_ollama),
    ("TOOLS", collect_tools),
    ("MEMORY", collect_memory),
    ("HOST", collect_host),
    ("THIS CHAT", collect_session),
)


async def _run_collector(
    name: str, fn: Callable[[StatusContext], Awaitable[Section]], ctx: StatusContext
) -> Section:
    """Run one collector under its timeout; failures become an 'n/a (reason)' WARN section."""
    try:
        return await asyncio.wait_for(fn(ctx), timeout=ctx.cfg.collector_timeout)
    except asyncio.TimeoutError:
        reason = f"timed out after {ctx.cfg.collector_timeout:g}s"
    except Exception as exc:  # noqa: BLE001 - a probe must never break /status
        reason = type(exc).__name__
        logger.warning("/status collector %s failed: %s", name, reason, exc_info=True)
    return Section(
        name, "n/a", [Item(f"n/a ({reason})", WARN, issue=f"{name.lower()} check failed ({reason})")]
    )


# ---------------------------------------------------------------------------- verdict
def compute_verdict(sections: list[Section]) -> tuple[str, list[str]]:
    """('OK'|'DEGRADED'|'DOWN', issue texts) from every WARN/FAIL item."""
    issues: list[tuple[int, str]] = []
    down = False
    for section in sections:
        for item in section.items:
            if _RANK[item.level] == 0:
                continue
            down = down or item.critical
            raw = item.issue or item.text
            for text in [raw] if isinstance(raw, str) else raw:
                issues.append((-_RANK[item.level], text))
    issues.sort(key=lambda pair: pair[0])
    texts = [text for _, text in issues]
    if down:
        return "DOWN", texts
    return ("DEGRADED" if texts else "OK"), texts


def _marker(level: str, style: str) -> str:
    """Status marker for a level in the configured style."""
    emoji = {OK: "✅", INFO: "•", NA: "⚪", WARN: "⚠️", FAIL: "❌"}
    text = {OK: "OK", INFO: "-", NA: "n/a", WARN: "WARN", FAIL: "FAIL"}
    return (emoji if style == "emoji" else text)[level]


def render_report(
    sections: list[Section], cfg: StatusConfig, *, full: bool, now: float, took: float | None = None
) -> str:
    """Pure formatter: verdict line + one block per section (+ thresholds in full mode)."""
    verdict, issues = compute_verdict(sections)
    line = f"Overall: {verdict}"
    if issues:
        shown = "; ".join(issues[:4])
        if len(issues) > 4:
            shown += f"; +{len(issues) - 4} more"
        line += f" - {shown}"
    stamp = datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    meta = f"PrivateAI status · {stamp}" + (f" · {took:.1f}s" if took is not None else "")
    blocks = [f"{line}\n{meta}"]
    for section in sections:
        header = f"{_marker(section.level, cfg.markers)} {section.name}"
        if section.head:
            header += f" · {section.head}"
        lines = [header]
        for item in section.items:
            if item.full and not full:
                continue
            if item.level in (WARN, FAIL):
                lines.append(f"{_marker(item.level, cfg.markers)} {item.text}")
            else:
                lines.append(f"• {item.text}")
        blocks.append("\n".join(lines))
    if full:
        blocks.append(thresholds_block(cfg))
    else:
        blocks.append("More: /status full · /memory · /tools · /think · /verbose · /timeout")
    return "\n\n".join(blocks)


def thresholds_block(cfg: StatusConfig) -> str:
    """Documented verdict thresholds (full view)."""
    return (
        "THRESHOLDS (env-overridable STATUS_*)\n"
        f"• DOWN: Ollama unreachable; memory service unreachable on the service backend\n"
        f"• WARN: host data older than {cfg.host_stale_seconds:g}s · load5/cpu >= {cfg.load_warn_ratio:g} · "
        f"cpu >= {cfg.cpu_warn_percent:g}% · mem avail < {cfg.mem_available_warn_percent:g}% · "
        f"swap >= {cfg.swap_warn_percent:g}% · disk >= {cfg.disk_warn_percent:g}% (FAIL "
        f"{cfg.disk_fail_percent:g}%) · VRAM >= {cfg.vram_warn_percent:g}% · temps cpu >= "
        f"{cfg.cpu_temp_warn_c:g}C / gpu >= {cfg.gpu_temp_warn_c:g}C · watchdog restarts/24h >= "
        f"{cfg.watchdog_restarts_warn}\n"
        f"• WARN: chat model not fully on GPU · loaded ctx != OLLAMA_NUM_CTX · chat call in flight > "
        f"{cfg.stuck_warn_seconds:g}s · model error in last {fmt_duration(cfg.model_error_window_seconds)} · "
        f"loop lag >= {cfg.loop_lag_warn_ms:g} ms · container mem >= {cfg.container_mem_warn_percent:g}%\n"
        f"• WARN: MCP server not connected · memory breaker not closed · memory service unreachable "
        f"(FAIL marker) · dual-write drift (> {cfg.dual_write_missing_warn} missing records or "
        f"> {cfg.dual_write_lag_seconds:g}s behind) · summary backlog > {cfg.memory_backlog_warn} or "
        f"summary older than {cfg.summary_stale_days:g}d with backlog\n"
        "• FAIL: systemd unit ollama/docker not active · container not running/unhealthy"
    )


async def build_status_report(
    *,
    settings: Any = None,
    session_id: str | int = "",
    memory: Any = None,
    rag: Any = None,
    hub: Any = None,
    memory_service: Any = None,
    full: bool = False,
    cfg: StatusConfig | None = None,
    collectors: tuple[tuple[str, Callable[[StatusContext], Awaitable[Section]]], ...] | None = None,
) -> str:
    """Run all collectors concurrently and return the rendered, secret-scrubbed text."""
    started = time.monotonic()
    cfg = cfg or StatusConfig()
    ctx = StatusContext(
        cfg=cfg,
        settings=settings,
        session_id=session_id,
        memory=memory,
        rag=rag,
        hub=hub,
        memory_service=memory_service,
        full=full,
    )
    data_dir = getattr(memory, "data_dir", None)
    ctx.legacy_task = asyncio.ensure_future(asyncio.to_thread(scan_legacy_memory, data_dir))
    try:
        results = await asyncio.gather(
            *(_run_collector(name, fn, ctx) for name, fn in (collectors or COLLECTORS)),
            return_exceptions=True,
        )
    finally:
        if not ctx.legacy_task.done():
            ctx.legacy_task.cancel()
    sections: list[Section] = []
    for (name, _), result in zip(collectors or COLLECTORS, results):
        if isinstance(result, BaseException):  # defence in depth: _run_collector catches Exception
            result = Section(name, "n/a", [Item(f"n/a ({type(result).__name__})", WARN,
                                                issue=f"{name.lower()} check failed")])
        sections.append(result)
    text = render_report(sections, cfg, full=full, now=ctx.now, took=time.monotonic() - started)
    return redact_secrets(text)


def split_message(text: str, limit: int) -> list[str]:
    """Split text into chunks <= limit, preferring blank-line then line boundaries."""
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    current = ""
    for block in text.split("\n\n"):
        candidate = f"{current}\n\n{block}" if current else block
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            parts.append(current)
            current = ""
        if len(block) <= limit:
            current = block
            continue
        for line in block.split("\n"):  # block too big: fall back to line boundaries
            while len(line) > limit:  # a single huge line: hard split
                if current:
                    parts.append(current)
                    current = ""
                parts.append(line[:limit])
                line = line[limit:]
            candidate = f"{current}\n{line}" if current else line
            if len(candidate) <= limit:
                current = candidate
            else:
                parts.append(current)
                current = line
    if current:
        parts.append(current)
    return parts

