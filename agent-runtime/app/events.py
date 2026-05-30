"""In-memory pub/sub for sidebar live updates.

Replaces the 30s polling loop in the LibreChat Work Queue sidebar with a
Server-Sent Events stream.  When any ledger mutation happens (proposal
created, proposal status changed, task created, task status changed,
comment added, lesson promoted), the writer publishes a small JSON event
on the broadcaster; HTTP subscribers receive each event as an SSE
message and re-fetch the affected entity.

Design choices:
- In-memory asyncio.Queue per subscriber.  No external dependency
  (Redis/NATS) because the multi-agent platform is single-process today.
  When we shard agent-runtime we move to a real bus.
- Per-(tenant_id, user_id) subscriber filter.  An event is only fanned
  out to subscribers whose namespace matches.
- Lossy under back-pressure.  Slow subscribers get their queue capped at
  256 events; older events are dropped silently.  Sidebar UX tolerates
  this — the next refresh re-syncs from /work_items.
- No replay buffer.  Subscribers see events from the moment they
  connect.  Initial state comes from a one-shot /work_items GET.

Wire-up pattern in writer functions:
    from app.events import broadcast
    broadcast({"type": "proposal_created", "tenant_id": t, "user_id": u, "id": pid})

Wire-up pattern in HTTP endpoint:
    return StreamingResponse(
        event_stream(tenant_id, user_id),
        media_type="text/event-stream",
    )
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict
from typing import Any, AsyncIterator

logger = logging.getLogger(__name__)

# Bounded queue per subscriber so a stuck client cannot OOM the service.
_QUEUE_MAX = 256

# subscribers keyed by (tenant_id, user_id) for cheap fan-out filter.
# Multiple subscribers per namespace are supported (e.g. user has two
# browser tabs open) — each gets its own queue.
_subscribers: dict[
    tuple[str, str], list[asyncio.Queue[dict[str, Any]]]
] = defaultdict(list)
_lock = asyncio.Lock()


def broadcast(event: dict[str, Any]) -> None:
    """Publish a single event.  Non-blocking, drops on full queues.

    Expected keys: type, tenant_id, user_id, plus payload-specific fields
    (id, status, etc.).  An optional `_ts` field is auto-stamped.

    Safe to call from sync code — schedules into the running event loop
    via call_soon_threadsafe when one exists, otherwise no-ops.  This
    means workers running outside the FastAPI loop (LifecycleWorker,
    DreamScanner) still publish correctly.
    """
    event.setdefault("_ts", time.time())
    key = (event.get("tenant_id", ""), event.get("user_id", ""))
    queues = _subscribers.get(key, [])
    if not queues:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No loop in this thread — schedule onto the main loop if known.
        loop = _main_loop
        if loop is None:
            logger.debug("broadcast: no event loop available, dropping %r", event)
            return
    loop.call_soon_threadsafe(_fanout, key, event)


def _fanout(key: tuple[str, str], event: dict[str, Any]) -> None:
    """Runs in the event loop — put_nowait on each subscriber queue."""
    queues = _subscribers.get(key, [])
    for q in queues:
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning(
                "broadcast: queue full for %r, dropping %s",
                key, event.get("type"),
            )


# Set during FastAPI startup so background workers can publish.
_main_loop: asyncio.AbstractEventLoop | None = None


def attach_main_loop(loop: asyncio.AbstractEventLoop) -> None:
    """FastAPI startup calls this so non-loop threads can broadcast()."""
    global _main_loop
    _main_loop = loop


async def event_stream(
    tenant_id: str, user_id: str
) -> AsyncIterator[str]:
    """Per-connection SSE generator.

    Sends a `ready` event immediately so the client knows the stream is
    open, then yields one SSE `data:` block per published event.

    A 25s heartbeat (comment line) keeps idle proxies from closing the
    connection.  EventSource auto-reconnects, so even if a proxy does
    close, the client will reattach within a few seconds.
    """
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=_QUEUE_MAX)
    key = (tenant_id, user_id)
    async with _lock:
        _subscribers[key].append(queue)
    try:
        # Initial ready event.
        yield _format_sse({"type": "ready", "_ts": time.time()})
        while True:
            try:
                evt = await asyncio.wait_for(queue.get(), timeout=25.0)
                yield _format_sse(evt)
            except asyncio.TimeoutError:
                # Heartbeat — keeps the TCP path warm and tells the proxy
                # the connection is alive.
                yield ": heartbeat\n\n"
    finally:
        async with _lock:
            try:
                _subscribers[key].remove(queue)
            except ValueError:
                pass
            if not _subscribers[key]:
                _subscribers.pop(key, None)


def _format_sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, default=str)}\n\n"


def subscriber_count() -> int:
    """Total open SSE connections — diagnostic only."""
    return sum(len(qs) for qs in _subscribers.values())
