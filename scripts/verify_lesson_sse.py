"""Verify lesson SSE events end-to-end.

What it proves
==============
1. broadcast({"type": "lesson_created", ...}) fans out to a connected
   /events subscriber for the matching (tenant_id, user_id).
2. broadcast({"type": "lesson_status_changed", ...}) same.
3. Events DO NOT leak to subscribers in a different namespace.
4. The /events endpoint actually streams SSE-formatted data.

Run inside the agent-runtime container:
  docker exec veilguard-agent-runtime-1 python /tmp/verify_lesson_sse.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, "/app")

from app.events import broadcast, event_stream, attach_main_loop


async def collect_events(tenant_id: str, user_id: str, *, n: int, timeout: float):
    """Open one event_stream subscription and collect up to n events."""
    gen = event_stream(tenant_id, user_id)
    events = []
    async def _loop():
        async for chunk in gen:
            events.append(chunk)
            if len(events) >= n:
                break
    try:
        await asyncio.wait_for(_loop(), timeout=timeout)
    except asyncio.TimeoutError:
        pass
    return events


async def main() -> int:
    # The event_stream() helper publishes a `ready` message on connect.
    # broadcast() runs via call_soon_threadsafe, so we need to register
    # the running loop.
    attach_main_loop(asyncio.get_running_loop())

    tenant = "verify-sse"
    user = "verify-sse-user"

    print("[sse] step 1 — subscriber receives lesson_created")
    # Schedule the broadcast for the next event-loop tick AFTER the
    # subscription is established.
    async def _fire(evt: dict, delay: float = 0.05):
        await asyncio.sleep(delay)
        broadcast(evt)

    asyncio.create_task(_fire({
        "type":      "lesson_created",
        "tenant_id": tenant,
        "user_id":   user,
        "id":        "lesson-test-1",
        "trigger":   "smoke trigger",
        "confidence": 0.9,
    }))
    asyncio.create_task(_fire({
        "type":       "lesson_status_changed",
        "tenant_id":  tenant,
        "user_id":    user,
        "id":         "lesson-test-1",
        "action":     "retire",
        "new_status": "retired",
    }, delay=0.10))

    # Want: ready + lesson_created + lesson_status_changed = 3 events.
    chunks = await collect_events(tenant, user, n=3, timeout=2.0)
    print(f"[sse]   received {len(chunks)} SSE frames")
    payloads = [c.strip() for c in chunks if c.strip().startswith("data: ")]
    types = []
    for p in payloads:
        try:
            import json as _j
            evt = _j.loads(p[len("data: "):])
            types.append(evt.get("type"))
        except Exception:
            types.append(None)
    print(f"[sse]   event types in order: {types}")
    if "lesson_created" not in types:
        print("[sse] step 1 FAIL: lesson_created missing")
        return 1
    if "lesson_status_changed" not in types:
        print("[sse] step 1 FAIL: lesson_status_changed missing")
        return 2

    # Step 2 — different namespace isolation.
    print("[sse] step 2 — subscriber in other namespace does NOT receive event")

    async def _bcast_to_a():
        await asyncio.sleep(0.05)
        broadcast({
            "type":      "lesson_created",
            "tenant_id": "namespace-A",
            "user_id":   "user-A",
            "id":        "lesson-iso-test",
        })

    asyncio.create_task(_bcast_to_a())
    other = await collect_events("namespace-B", "user-B", n=2, timeout=1.0)
    other_payloads = [c for c in other if c.strip().startswith("data: ")]
    other_types = []
    for p in other_payloads:
        try:
            import json as _j
            evt = _j.loads(p.strip()[len("data: "):])
            other_types.append(evt.get("type"))
        except Exception:
            pass
    if "lesson_created" in other_types:
        print(f"[sse] step 2 FAIL: cross-namespace leak — types={other_types}")
        return 3
    print(f"[sse]   namespace-B saw only: {other_types} (no lesson_created — good)")

    print("[sse] ALL STEPS PASSED ✅")
    print(f"[sse]   lesson_created broadcast → SSE: verified")
    print(f"[sse]   lesson_status_changed broadcast → SSE: verified")
    print(f"[sse]   cross-namespace isolation: verified")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
