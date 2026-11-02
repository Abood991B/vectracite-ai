"""A minimal circuit breaker for outbound dependencies.

Three states: closed (calls allowed), open (calls rejected until the recovery window elapses), and
half-open (one probe allowed; success closes, failure reopens). It is deliberately small: the point is
to stop paying a dependency's timeout on every request while it is known to be down.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Literal

State = Literal["closed", "open", "half_open"]


class CircuitBreaker:
    """Thread-safe breaker with an injectable clock so tests do not sleep."""

    def __init__(
        self,
        *,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._failure_threshold = max(1, failure_threshold)
        self._recovery_timeout = recovery_timeout
        self._clock = clock
        self._lock = threading.Lock()
        self._state: State = "closed"
        self._failures = 0
        self._opened_at = 0.0

    @property
    def state(self) -> State:
        with self._lock:
            return self._state

    def allow(self) -> bool:
        """Return whether a call may be attempted now."""
        with self._lock:
            if self._state == "open" and self._clock() - self._opened_at >= self._recovery_timeout:
                self._state = "half_open"
            return self._state != "open"

    def record_success(self) -> None:
        with self._lock:
            self._state = "closed"
            self._failures = 0

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._state == "half_open" or self._failures >= self._failure_threshold:
                self._state = "open"
                self._opened_at = self._clock()

    def reset(self) -> None:
        """Return the breaker to closed. Intended for tests and manual recovery."""
        with self._lock:
            self._state = "closed"
            self._failures = 0
            self._opened_at = 0.0
