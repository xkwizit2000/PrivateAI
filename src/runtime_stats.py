"""Light in-memory tracking of chat-model calls for /status (metadata only).

Nothing here stores prompts, replies or any chat text: only durations, outcome
categories (exception class names / "timeout") and timestamps.
"""

from __future__ import annotations

import itertools
import time
from collections import deque
from dataclasses import dataclass
from time import monotonic

# Wall/monotonic start of this Python process (for the uptime line).
PROCESS_STARTED_MONOTONIC = monotonic()
PROCESS_STARTED_EPOCH = time.time()


@dataclass(frozen=True)
class CallRecord:
    """One finished chat-model call."""

    finished_epoch: float
    seconds: float
    outcome: str | None  # None = success, else "timeout" / exception class name


class ModelCallStats:
    """Ring buffer of the last N model calls plus retry/in-flight bookkeeping."""

    def __init__(self, maxlen: int = 20) -> None:
        """Create empty stats keeping the last ``maxlen`` finished calls."""
        self._calls: deque[CallRecord] = deque(maxlen=maxlen)
        self._retries: deque[float] = deque(maxlen=maxlen)
        self._in_flight: dict[int, float] = {}
        self._ids = itertools.count(1)
        self.total_calls = 0
        self.total_errors = 0
        self.total_retries = 0

    def begin(self) -> int:
        """Register a call that just started; returns its id for :meth:`end`."""
        call_id = next(self._ids)
        self._in_flight[call_id] = monotonic()
        return call_id

    def end(self, call_id: int, outcome: str | None) -> None:
        """Record the finish of a call (``outcome`` None = success)."""
        started = self._in_flight.pop(call_id, None)
        if started is None:
            return
        self.total_calls += 1
        if outcome is not None:
            self.total_errors += 1
        self._calls.append(CallRecord(time.time(), monotonic() - started, outcome))

    def record_retry(self) -> None:
        """Note that a connection-type failure triggered a wait-and-retry."""
        self.total_retries += 1
        self._retries.append(time.time())

    @property
    def in_flight(self) -> int:
        """Number of model calls currently running."""
        return len(self._in_flight)

    def oldest_in_flight_seconds(self) -> float | None:
        """Age of the longest-running in-flight call, or None when idle."""
        if not self._in_flight:
            return None
        return monotonic() - min(self._in_flight.values())

    @property
    def last_call(self) -> CallRecord | None:
        """Most recent finished call (success or error)."""
        return self._calls[-1] if self._calls else None

    @property
    def last_success(self) -> CallRecord | None:
        """Most recent successful call."""
        for record in reversed(self._calls):
            if record.outcome is None:
                return record
        return None

    @property
    def last_error(self) -> CallRecord | None:
        """Most recent failed call."""
        for record in reversed(self._calls):
            if record.outcome is not None:
                return record
        return None

    @property
    def last_retry_epoch(self) -> float | None:
        """Wall time of the latest retry event."""
        return self._retries[-1] if self._retries else None

    def recent_durations(self) -> list[float]:
        """Durations (s) of the successful calls still in the ring buffer."""
        return [r.seconds for r in self._calls if r.outcome is None]


# Process-wide instance used by agent.py and the /status report.
MODEL_STATS = ModelCallStats()
