"""Recall-time hint fan-out across registered connectors.

Veilguard's recall step calls :func:`gather_hints` after TCMM's own
recall returns. Connectors run their `hint()` methods in parallel,
each bounded by a per-connector deadline. Slow or failing connectors
are dropped for the current turn (and circuit-broken for a short
window so subsequent turns don't re-suffer the same failure).

Score normalization happens via each connector's `calibrate()` so the
returned snippets compete fairly across heterogeneous sources
(BM25-like, cosine, vendor-opaque). Output is a flat, score-sorted
list ready to feed into the renderer or hand off to the TCMM fuser.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from .base import Connector
from .types import Snippet, UserContext


logger = logging.getLogger("connectors.recall")


# ─── circuit breaker state ────────────────────────────────────────────


@dataclass
class _BreakerState:
    """Per-connector failure tracking for the circuit breaker.

    Three failures within `_BREAKER_WINDOW_S` seconds opens the breaker
    for `_BREAKER_COOLDOWN_S` seconds — calls during cooldown are
    skipped without invoking the connector. After cooldown expires,
    the next call is a probe; success closes the breaker, failure
    resets the cooldown.
    """

    failure_times: list[float] = field(default_factory=list)
    open_until: float = 0.0


_BREAKER_WINDOW_S = 60.0
_BREAKER_THRESHOLD = 3
_BREAKER_COOLDOWN_S = 60.0


_breaker_state: dict[str, _BreakerState] = {}
_breaker_lock = asyncio.Lock()


async def _record_failure(name: str) -> None:
    async with _breaker_lock:
        state = _breaker_state.setdefault(name, _BreakerState())
        now = time.monotonic()
        # Drop failures outside the rolling window
        state.failure_times = [
            t for t in state.failure_times if now - t <= _BREAKER_WINDOW_S
        ]
        state.failure_times.append(now)
        if len(state.failure_times) >= _BREAKER_THRESHOLD:
            state.open_until = now + _BREAKER_COOLDOWN_S
            logger.warning(
                f"connector {name!r} circuit-breaker OPEN for "
                f"{_BREAKER_COOLDOWN_S:.0f}s after "
                f"{len(state.failure_times)} failures within "
                f"{_BREAKER_WINDOW_S:.0f}s"
            )


async def _record_success(name: str) -> None:
    async with _breaker_lock:
        state = _breaker_state.get(name)
        if state is None:
            return
        # A successful call clears the breaker entirely.
        state.failure_times.clear()
        state.open_until = 0.0


async def _is_breaker_open(name: str) -> bool:
    async with _breaker_lock:
        state = _breaker_state.get(name)
        if state is None:
            return False
        return time.monotonic() < state.open_until


# ─── public fan-out ───────────────────────────────────────────────────


async def gather_hints(
    connectors: list[Connector],
    prompt: str,
    user_ctx: UserContext,
    *,
    deadline_ms: int = 300,
    top_k_per_connector: int = 10,
) -> list[Snippet]:
    """Fan out `hint()` across the given connectors in parallel.

    Each connector gets at most ``deadline_ms`` milliseconds. A
    connector exceeding the deadline contributes zero snippets for
    this turn but is NOT recorded as a failure (timeouts are expected
    under load and shouldn't trip the breaker — only thrown exceptions
    do). Connectors with the breaker open are skipped immediately.

    Returns a flat list of `Snippet`s sorted by descending score.
    Scores are normalized per-connector via `connector.calibrate()`
    before sorting, so connectors with different native scoring scales
    compete fairly.
    """
    if not connectors or not prompt:
        return []

    deadline_s = max(0.05, deadline_ms / 1000.0)

    async def _one(c: Connector) -> list[Snippet]:
        if await _is_breaker_open(c.name):
            return []
        try:
            result = await asyncio.wait_for(
                c.hint(
                    prompt=prompt,
                    user_ctx=user_ctx,
                    deadline_ms=deadline_ms,
                    top_k=top_k_per_connector,
                ),
                timeout=deadline_s,
            )
        except asyncio.TimeoutError:
            logger.info(
                f"connector {c.name!r} hint() exceeded "
                f"{deadline_ms}ms — dropped this turn"
            )
            return []
        except NotImplementedError:
            # Connector lists HINT in capabilities but didn't implement —
            # configuration bug, log once and move on.
            logger.warning(
                f"connector {c.name!r} advertises HINT but raised "
                "NotImplementedError"
            )
            return []
        except Exception as e:
            logger.warning(
                f"connector {c.name!r} hint() failed with "
                f"{type(e).__name__}: {e}"
            )
            await _record_failure(c.name)
            return []

        await _record_success(c.name)

        if not isinstance(result, list):
            return []

        # Apply per-connector score calibration. We mutate in place
        # because Snippet is a regular dataclass — no benefit to
        # producing a copy here, and the snippet is owned solely by
        # the recall pipeline at this point.
        normalized: list[Snippet] = []
        for snippet in result:
            if not isinstance(snippet, Snippet):
                continue
            try:
                snippet.score = float(c.calibrate(snippet.score))
            except (TypeError, ValueError):
                snippet.score = 0.0
            normalized.append(snippet)
        return normalized

    results = await asyncio.gather(
        *[_one(c) for c in connectors],
        return_exceptions=False,
    )

    flat: list[Snippet] = []
    for snippets in results:
        flat.extend(snippets)

    flat.sort(key=lambda s: s.score, reverse=True)
    return flat
