"""Phase 6.7 — Artifact Production Ratio (APR) + circuit breaker.

Operationalises the §2 design principle "Tasks transform state;
conversations are scaffolding."  APR is the per-system metric that
distinguishes a healthy agent platform (artifacts produced per token
spent) from an Auto-GPT / BabyAGI-style collapse into expensive
self-narration.

    APR = artifacts_in_window / (tokens_in_window / 1000)

Counts only state-mutation events: status_change, attach_output,
observe (when persisted), approval response, dream node update,
decision-ledger entry.  Does NOT count: agent-to-agent messages,
intermediate reasoning, internal critique narration.

Healthy band:        0.5 – 2.0 artifacts per 1k tokens
Collapse signal:     APR < 0.1 sustained over rolling 30-minute window

When the collapse signal fires:
  1. inbox_poller stops dispatching new tasks.
  2. In-flight tasks complete their current turn and are paused.
  3. Sidebar surfaces banner: "Veilguard halted: APR=<N>".
  4. Operator unblocks via /apr/resume.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

# Phase 6.7 tunables.  Public so tests can monkey-patch the floor for
# fast-firing fixtures (AC-35).
HEALTHY_LOW: float = 0.5
HEALTHY_HIGH: float = 2.0
CIRCUIT_BREAKER_FLOOR: float = 0.1
CIRCUIT_BREAKER_WINDOW_S: float = 1800.0  # 30 minutes


@dataclass
class _Sample:
    ts: float
    artifacts: int
    tokens: int


@dataclass
class APRWindow:
    """Rolling-window APR calculator.

    Append samples via `record(artifacts, tokens)`; query the current
    APR via `current_apr(now=...)`; circuit-breaker state via
    `breaker_should_fire(now=...)`.

    The window is `CIRCUIT_BREAKER_WINDOW_S` seconds wide; samples
    older than that are dropped on each `current_apr` call.  This keeps
    the window cheap (no background sweep needed).
    """

    window_s: float = CIRCUIT_BREAKER_WINDOW_S
    floor:    float = CIRCUIT_BREAKER_FLOOR
    samples:  deque[_Sample] = field(default_factory=deque)
    # State machine: 'normal' | 'tripped'.  Once tripped, requires
    # explicit `clear()` to resume — even if APR climbs back into the
    # healthy band, we don't auto-resume because the work that caused
    # the collapse may still be inflight.
    state: str = "normal"
    tripped_at_ts: float | None = None

    def record(
        self,
        *,
        artifacts: int,
        tokens: int,
        ts: float | None = None,
    ) -> None:
        """Append a sample.  Idempotent on (artifacts=0, tokens=0)."""
        if artifacts == 0 and tokens == 0:
            return
        if artifacts < 0 or tokens < 0:
            raise ValueError(
                f"artifacts and tokens must be non-negative, "
                f"got artifacts={artifacts}, tokens={tokens}"
            )
        t = time.time() if ts is None else ts
        self.samples.append(_Sample(ts=t, artifacts=artifacts, tokens=tokens))
        self._evict_old(now=t)

    def _evict_old(self, *, now: float) -> None:
        cutoff = now - self.window_s
        while self.samples and self.samples[0].ts < cutoff:
            self.samples.popleft()

    def current_apr(self, *, now: float | None = None) -> float | None:
        """Return APR for the current window, or None if no data.

        APR = sum(artifacts) / (sum(tokens) / 1000.0)
        Returns None when token sum is 0 (cold start; no division).
        """
        t = time.time() if now is None else now
        self._evict_old(now=t)
        if not self.samples:
            return None
        total_artifacts = sum(s.artifacts for s in self.samples)
        total_tokens = sum(s.tokens for s in self.samples)
        if total_tokens == 0:
            return None
        return total_artifacts / (total_tokens / 1000.0)

    def breaker_should_fire(self, *, now: float | None = None) -> bool:
        """True iff APR has been below the floor for the full window.

        Insufficient-data case (cold start, fewer than 5 samples) →
        False; the breaker won't fire on a quiet startup.  Phase 6.7
        AC-34 covers the boundary cases explicitly.
        """
        t = time.time() if now is None else now
        self._evict_old(now=t)
        # Need at least 5 samples spanning at least 1/3 of the window.
        if len(self.samples) < 5:
            return False
        span = self.samples[-1].ts - self.samples[0].ts
        if span < (self.window_s / 3.0):
            return False
        apr = self.current_apr(now=t)
        if apr is None:
            return False
        return apr < self.floor

    def trip(self, *, now: float | None = None) -> None:
        """Mark the breaker as tripped.  Caller is responsible for the
        side effects (pausing dispatch, surfacing sidebar banner)."""
        t = time.time() if now is None else now
        self.state = "tripped"
        self.tripped_at_ts = t

    def clear(self) -> None:
        """Operator-acked resume.  Wipes the sample window so the breaker
        doesn't re-fire on the same data immediately."""
        self.state = "normal"
        self.tripped_at_ts = None
        self.samples.clear()

    def is_tripped(self) -> bool:
        return self.state == "tripped"


# ── Singleton wired by ledger writes + audit module ────────────────────


GLOBAL_APR: APRWindow = APRWindow()


def apr_record_artifact(n: int = 1) -> None:
    """Hook called from ledger writers + memory writers + approval gate.

    Every state-mutation event (status_change, attach_output, observe
    persisted, approval response, comment_added, dream node update,
    decision-ledger entry) counts as one artifact.

    Safe to call from sync code — no event loop dependency.
    """
    if n <= 0:
        return
    GLOBAL_APR.record(artifacts=n, tokens=0)


def apr_record_tokens(input_tokens: int = 0, output_tokens: int = 0) -> None:
    """Hook called from the audit / adapter layer after each LLM round-trip.

    Counts total tokens billed (input + output, including cache_read +
    cache_create per memory note `architecture_token_accounting`).
    """
    total = max(0, input_tokens) + max(0, output_tokens)
    if total == 0:
        return
    GLOBAL_APR.record(artifacts=0, tokens=total)


def apr_should_pause_dispatch() -> bool:
    """Inbox-poller calls this before claiming new work.  If True, the
    circuit breaker is firing — pause new dispatches until operator
    unblocks via `apr_resume()`.
    """
    if GLOBAL_APR.is_tripped():
        return True
    if GLOBAL_APR.breaker_should_fire():
        GLOBAL_APR.trip()
        return True
    return False


def apr_resume() -> None:
    """Operator unblock — wipe the window and resume dispatch."""
    GLOBAL_APR.clear()


def apr_snapshot() -> dict[str, float | str | None]:
    """Diagnostic snapshot for the /apr/status endpoint + sidebar banner."""
    return {
        "apr":          GLOBAL_APR.current_apr(),
        "state":        GLOBAL_APR.state,
        "tripped_at":   GLOBAL_APR.tripped_at_ts,
        "n_samples":    len(GLOBAL_APR.samples),
        "window_s":     GLOBAL_APR.window_s,
        "floor":        GLOBAL_APR.floor,
        "healthy_low":  HEALTHY_LOW,
        "healthy_high": HEALTHY_HIGH,
    }
