"""HTTP client for the external memory service (dual-write, recall, explicit search).

Design rules (see docs/memory-service-design.md):

* The service is an *add-on*: every call has a short timeout, every failure is
  swallowed (metadata-only warning) and a circuit breaker stops calling a dead
  service altogether, so a chat turn never waits on it or breaks because of it.
* Nothing in this module logs message text: only counts, scores, decisions,
  char counts and latencies.
* The bot's JSONL transcript stays the system of record; the service only gets
  a copy of every completed turn (idempotent: same timestamp/role/content).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from time import monotonic
from typing import Any

import httpx

from .config import (
    MEMORY_APPEND_TIMEOUT,
    MEMORY_BREAKER_COOLDOWN,
    MEMORY_BREAKER_FAILURES,
    MEMORY_BREAKER_MAX_COOLDOWN,
    MEMORY_RECALL_MAX_CHARS,
    MEMORY_RECALL_TIMEOUT,
)
from .memory import MemoryStore

logger = logging.getLogger(__name__)

MAX_QUERY_CHARS = 2000
MAX_TURN_CHARS = 20_000
RECALL_HEADER = "Possibly relevant earlier conversation (ignore it if it is unrelated):"
PROFILE_HEADER = "Known facts about the user:"


@dataclass(frozen=True)
class RecallResult:
    """Metadata (plus the block text, never logged) of one /v1/recall answer."""

    fired: bool
    hits: int
    top: float | None
    text: str
    ms: float
    why: str

    @property
    def chars(self) -> int:
        """Characters the block would inject (0 when the gate did not fire)."""
        return len(self.text)


def _why(reason: str) -> str:
    """Reduce the service's gate reason to a short, text-free category."""
    lowered = reason.lower()
    if lowered.startswith("strong"):
        return "strong"
    if "cue" in lowered and lowered.startswith("past"):
        return "cue"
    if lowered.startswith("empty"):
        return "empty"
    if lowered.startswith("no gate"):
        return "none"
    return "other"


def format_recall_block(text: str, max_chars: int = MEMORY_RECALL_MAX_CHARS) -> str:
    """Wrap a recall result as one compact, size-capped system-message body."""
    body = text.strip()
    if not body:
        return ""
    room = max(max_chars - len(RECALL_HEADER) - 1, 0)
    if room <= 0:
        return ""
    return f"{RECALL_HEADER}\n{body[:room]}"


class ServiceMemory:
    """Async client for the memory service's append / recall / search endpoints."""

    def __init__(
        self,
        base_url: str,
        token: str = "",
        *,
        recall_mode: str = "shadow",
        dual_write: bool = True,
        backend: str = "legacy",
        recall_timeout: float = MEMORY_RECALL_TIMEOUT,
        append_timeout: float = MEMORY_APPEND_TIMEOUT,
        breaker_failures: int = MEMORY_BREAKER_FAILURES,
        breaker_cooldown: float = MEMORY_BREAKER_COOLDOWN,
        breaker_max_cooldown: float = MEMORY_BREAKER_MAX_COOLDOWN,
        clock: Callable[[], float] = monotonic,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Configure the client; no network I/O happens until the first call."""
        self.base_url = base_url.rstrip("/")
        self._token = token
        self.recall_mode = recall_mode
        self.dual_write = dual_write
        self.backend = backend
        self.recall_timeout = recall_timeout
        self.append_timeout = append_timeout
        self.breaker_failures = max(1, breaker_failures)
        self.breaker_cooldown = breaker_cooldown
        self.breaker_max_cooldown = max(breaker_max_cooldown, breaker_cooldown)
        self._clock = clock
        self._transport = transport
        self._http: httpx.AsyncClient | None = None
        self._tasks: set[asyncio.Task[Any]] = set()

        # Circuit breaker state.
        self._failures = 0
        self._trips = 0
        self._open_until: float | None = None
        self._probe_inflight = False

        # Counters since process start (for /memory and tests).
        self.counters: dict[str, int] = {
            "recall_calls": 0,
            "recall_fired": 0,
            "recall_skipped": 0,
            "recall_errors": 0,
            "recall_injected": 0,
            "append_ok": 0,
            "append_fail": 0,
            "skipped_open": 0,
        }
        # Wall time of the last successful dual-write append (None until one succeeds).
        self.last_append_epoch: float | None = None
        self._recall_ms: deque[float] = deque(maxlen=500)
        self._top_scores: deque[float] = deque(maxlen=500)

    # ------------------------------------------------------------------ plumbing
    def _client(self) -> httpx.AsyncClient:
        """Create the shared AsyncClient lazily (inside the running loop)."""
        if self._http is None:
            headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
            self._http = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                transport=self._transport,
                timeout=httpx.Timeout(10.0, connect=3.0),
            )
        return self._http

    async def close(self) -> None:
        """Cancel in-flight background calls and close the HTTP client."""
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    @property
    def pending_tasks(self) -> int:
        """Background calls (dual-write / shadow recall) still in flight."""
        return len(self._tasks)

    def spawn(self, coro: Awaitable[Any]) -> asyncio.Task[Any]:
        """Run a coroutine in the background; exceptions are swallowed and logged."""
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._task_done)
        return task

    def _task_done(self, task: asyncio.Task[Any]) -> None:
        """Forget a finished background task and never let its error escape."""
        self._tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.warning("memory service background task failed: %s", type(exc).__name__)

    # ------------------------------------------------------------ circuit breaker
    @property
    def breaker_state(self) -> str:
        """closed | open | half-open (one probe allowed)."""
        if self._open_until is None:
            return "closed"
        return "half-open" if self._clock() >= self._open_until else "open"

    def breaker_retry_in(self) -> float:
        """Seconds until the next probe is allowed (0 when closed/half-open)."""
        if self._open_until is None:
            return 0.0
        return max(0.0, self._open_until - self._clock())

    def _allow(self) -> bool:
        """True if a call may go out now (closed, or one half-open probe)."""
        if self._open_until is None:
            return True
        if self._clock() < self._open_until:
            return False
        if self._probe_inflight:
            return False
        self._probe_inflight = True
        return True

    def _on_success(self, was_probe: bool) -> None:
        """Reset the breaker after a good answer."""
        recovered = self._open_until is not None
        self._failures = 0
        self._trips = 0
        self._open_until = None
        self._probe_inflight = False
        if recovered:
            logger.info("memory service recovered; resuming calls")

    def _on_failure(self, op: str, exc: BaseException, was_probe: bool, ms: float) -> None:
        """Count a failure and open the breaker once the threshold is reached."""
        if was_probe:
            self._probe_inflight = False
        self._failures += 1
        error = type(exc).__name__
        if isinstance(exc, httpx.HTTPStatusError):
            error = f"HTTP {exc.response.status_code}"
        if was_probe or self._failures >= self.breaker_failures:
            cooldown = min(self.breaker_cooldown * (2**self._trips), self.breaker_max_cooldown)
            self._trips += 1
            self._open_until = self._clock() + cooldown
            logger.warning(
                "memory service unavailable (op=%s error=%s after %d failure(s)); "
                "pausing calls for %.0fs",
                op,
                error,
                self._failures,
                cooldown,
            )
        else:
            logger.warning(
                "memory service call failed op=%s error=%s ms=%.0f (%d/%d before pausing)",
                op,
                error,
                ms,
                self._failures,
                self.breaker_failures,
            )

    async def _request(
        self,
        op: str,
        method: str,
        path: str,
        *,
        timeout: float,
        json: Any = None,
        params: dict[str, Any] | None = None,
        force: bool = False,
    ) -> dict[str, Any] | None:
        """One bounded HTTP call. Returns parsed JSON, or None on any failure/breaker skip."""
        was_probe = False
        if not force:
            before_open = self._open_until is not None
            if not self._allow():
                self.counters["skipped_open"] += 1
                return None
            was_probe = before_open
        started = monotonic()
        try:
            response = await asyncio.wait_for(
                self._client().request(method, path, json=json, params=params),
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError("unexpected response shape")
        except asyncio.CancelledError:
            if was_probe:
                self._probe_inflight = False
            raise
        except Exception as exc:  # noqa: BLE001 - the service must never break a turn
            self._on_failure(op, exc, was_probe, (monotonic() - started) * 1000)
            return None
        self._on_success(was_probe)
        return data

    # --------------------------------------------------------------------- append
    async def append_turn(
        self,
        session_id: str | int,
        user_message: str,
        assistant_message: str,
        timestamp: str | None = None,
    ) -> bool:
        """Send one completed turn (user + assistant record). Never raises."""
        stamp = {"timestamp": timestamp} if timestamp else {}
        payload = {
            "session_id": str(session_id),
            "turns": [
                {"role": "user", "content": user_message[:MAX_TURN_CHARS], **stamp},
                {"role": "assistant", "content": assistant_message[:MAX_TURN_CHARS], **stamp},
            ],
        }
        started = monotonic()
        data = await self._request(
            "append", "POST", "/v1/turns", timeout=self.append_timeout, json=payload
        )
        ms = (monotonic() - started) * 1000
        if data is None:
            self.counters["append_fail"] += 1
            return False
        self.counters["append_ok"] += 1
        self.last_append_epoch = time.time()
        logger.info(
            "memory dual-write: turns_new=%s chunks_new=%s ms=%.0f",
            data.get("turns_new", "?"),
            data.get("chunks_new", "?"),
            ms,
        )
        return True

    def dual_write_turn(
        self,
        session_id: str | int,
        user_message: str,
        assistant_message: str,
        timestamp: str | None = None,
    ) -> None:
        """Fire-and-forget append: returns immediately, the reply never waits on it."""
        self.spawn(self.append_turn(session_id, user_message, assistant_message, timestamp))

    # --------------------------------------------------------------------- recall
    async def recall(
        self,
        query: str,
        session_id: str | int | None,
        *,
        exclude_recent: int = 8,
        max_hits: int = 2,
        max_chars: int = MEMORY_RECALL_MAX_CHARS,
    ) -> RecallResult | None:
        """Ask the service's gate+search for a recall block. None on failure/skip."""
        payload: dict[str, Any] = {
            "query": query[:MAX_QUERY_CHARS],
            "exclude_recent": exclude_recent,
            "max_hits": max_hits,
            "max_chars": max_chars,
        }
        if session_id is not None:
            payload["session"] = str(session_id)
        started = monotonic()
        data = await self._request(
            "recall", "POST", "/v1/recall", timeout=self.recall_timeout, json=payload
        )
        ms = (monotonic() - started) * 1000
        self.counters["recall_calls"] += 1
        if data is None:
            self.counters["recall_errors"] += 1
            return None
        hits = data.get("hits")
        top = data.get("top_cos")
        text = str(data.get("text") or "")[:max_chars]
        fired = bool(data.get("inject")) and bool(text)
        result = RecallResult(
            fired=fired,
            hits=len(hits) if isinstance(hits, list) else 0,
            top=float(top) if isinstance(top, (int, float)) else None,
            text=text if fired else "",
            ms=ms,
            why=_why(str(data.get("reason") or "")),
        )
        self.counters["recall_fired" if fired else "recall_skipped"] += 1
        self._recall_ms.append(ms)
        if result.top is not None:
            self._top_scores.append(result.top)
        return result

    def log_recall(self, result: RecallResult | None, mode: str, ms: float | None = None) -> None:
        """Log one metadata-only recall line (never any message or hit text)."""
        label = "memory shadow" if mode == "shadow" else "memory recall"
        if result is None:
            logger.info("%s: error=unavailable ms=%.0f", label, ms or 0.0)
            return
        top = f"{result.top:.2f}" if result.top is not None else "n/a"
        logger.info(
            "%s: gate=%s hits=%d top=%s chars=%d ms=%.0f why=%s",
            label,
            "fired" if result.fired else "skipped",
            result.hits,
            top,
            result.chars,
            result.ms,
            result.why,
        )

    async def shadow_recall(self, query: str, session_id: str | int | None) -> None:
        """Background shadow recall: query, log metadata, discard the text."""
        started = monotonic()
        result = await self.recall(query, session_id)
        self.log_recall(result, "shadow", (monotonic() - started) * 1000)

    def start_shadow_recall(self, query: str, session_id: str | int | None) -> None:
        """Schedule a shadow recall without awaiting it (zero added latency)."""
        self.spawn(self.shadow_recall(query, session_id))

    async def recall_block(self, query: str, session_id: str | int | None) -> str:
        """`on` mode: await the (bounded) recall and return the capped system-message body."""
        started = monotonic()
        result = await self.recall(query, session_id)
        self.log_recall(result, "on", (monotonic() - started) * 1000)
        if result is None or not result.fired:
            return ""
        block = format_recall_block(result.text)
        if block:
            self.counters["recall_injected"] += 1
        return block

    # ------------------------------------------------------- other endpoints
    async def search(self, query: str, k: int = 5) -> str | None:
        """Explicit one-shot search across all sessions (for /recall); the service caps the text."""
        payload: dict[str, Any] = {"query": query[:MAX_QUERY_CHARS], "k": k}
        data = await self._request(
            "search", "POST", "/v1/search", timeout=max(self.recall_timeout, 5.0), json=payload,
            force=True,
        )
        if data is None:
            return None
        return str(data.get("text") or "")

    async def stats(self) -> dict[str, Any] | None:
        """GET /v1/stats (bypasses the breaker: an explicit status check may heal it)."""
        return await self._request(
            "stats", "GET", "/v1/stats", timeout=max(self.recall_timeout, 5.0), force=True
        )

    async def probe(self, timeout: float = 2.0) -> dict[str, Any]:
        """Read-only health probe for /status: GET /healthz then GET /v1/stats.

        Unlike the chat-path calls this never touches the circuit breaker or the
        counters (a status check must not change behaviour). Returns
        ``{"healthy": bool, "ms": float, "stats": dict|None, "error": str|None}``;
        it never raises.
        """
        started = monotonic()
        try:
            client = self._client()
            health = await asyncio.wait_for(client.get("/healthz"), timeout=timeout)
            health.raise_for_status()
            remaining = max(0.2, timeout - (monotonic() - started))
            stats_response = await asyncio.wait_for(client.get("/v1/stats"), timeout=remaining)
            stats_response.raise_for_status()
            stats = stats_response.json()
            return {
                "healthy": True,
                "ms": (monotonic() - started) * 1000,
                "stats": stats if isinstance(stats, dict) else None,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001 - /status must never raise
            error = type(exc).__name__
            if isinstance(exc, httpx.HTTPStatusError):
                error = f"HTTP {exc.response.status_code}"
            elif isinstance(exc, asyncio.TimeoutError):
                error = "timeout"
            return {
                "healthy": False,
                "ms": (monotonic() - started) * 1000,
                "stats": None,
                "error": error,
            }

    async def core(self, session_id: str | int, turns: int = 1) -> dict[str, Any] | None:
        """GET /v1/core for the fact profile (service backend)."""
        return await self._request(
            "core",
            "GET",
            "/v1/core",
            timeout=self.recall_timeout,
            params={"session": str(session_id), "turns": turns},
        )

    # ------------------------------------------------------------------- status
    def shadow_summary(self) -> str:
        """One-line, text-free summary of recall activity since process start."""
        c = self.counters
        times = sorted(self._recall_ms)
        if times:
            p50 = times[len(times) // 2]
            p95 = times[min(len(times) - 1, int(len(times) * 0.95))]
            latency = f"ms p50={p50:.0f} p95={p95:.0f}"
        else:
            latency = "ms n/a"
        top = (
            f"top avg={sum(self._top_scores) / len(self._top_scores):.2f}"
            if self._top_scores
            else "top n/a"
        )
        return (
            f"recalls={c['recall_calls']} fired={c['recall_fired']} skipped={c['recall_skipped']} "
            f"errors={c['recall_errors']} injected={c['recall_injected']} {top} {latency}"
        )

    def status_text(self) -> str:
        """Config + counters block for the /memory command (no conversation text)."""
        c = self.counters
        return (
            f"settings: backend={self.backend} recall={self.recall_mode} "
            f"dual_write={'on' if self.dual_write else 'off'}\n"
            f"breaker: {self.breaker_state}"
            + (f" (retry in {self.breaker_retry_in():.0f}s)" if self.breaker_state == "open" else "")
            + f"\ndual-write since start: ok={c['append_ok']} failed={c['append_fail']} "
            f"skipped(breaker)={c['skipped_open']}\n"
            f"recall since start: {self.shadow_summary()}"
        )


class CoreMemoryStore(MemoryStore):
    """MEMORY_BACKEND=service: tiny core context instead of summary + RAG.

    The prompt gets the service's fact profile (if any) plus the last N records
    read from the local JSONL (still the system of record, so the very latest
    turn is always present even if a dual-write was lost). No rolling summary
    is read or written and no Ollama call is made for summarisation.
    """

    def __init__(self, *args: Any, service: ServiceMemory | None = None, **kwargs: Any) -> None:
        """Same arguments as MemoryStore plus the service client for the fact profile."""
        super().__init__(*args, **kwargs)
        self.service = service

    async def context_messages(self, session_id: str | int) -> list[dict[str, str]]:
        """Return [fact profile?] + the last N records (compact, size-capped)."""
        transcript_path, _ = self._paths(session_id)
        async with self._lock(session_id):
            records = self._read_records(transcript_path)

        messages: list[dict[str, str]] = []
        remaining = self.max_context_chars
        if self.service is not None:
            core = await self.service.core(session_id, turns=1)
            profile = str((core or {}).get("profile") or "").strip()[:600]
            if profile:
                messages.append({"role": "system", "content": f"{PROFILE_HEADER}\n{profile}"})
                remaining -= len(profile)

        selected: deque[dict[str, str]] = deque()
        for record in reversed(records[-(self.recent_turns * 2) :]):
            content = str(record.get("content", ""))
            if not content:
                continue
            content = content[-remaining:] if remaining > 0 else ""
            if not content:
                break
            selected.appendleft({"role": str(record.get("role", "user")), "content": content})
            remaining -= len(content)
            if remaining <= 0:
                break
        messages.extend(selected)
        return messages

    async def summarize_if_needed(self, session_id: str | int, client: Any, model: str) -> bool:
        """No rolling summary in service mode."""
        return False
