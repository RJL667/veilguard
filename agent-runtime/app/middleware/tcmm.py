"""TCMM render middleware.

Pre-call: fetch the TCMM blob for this conversation via HTTP and inject
as the system prefix.  Byte-stable across sibling sub-calls so the
Anthropic prompt cache hits (per spec §3.7.0 + Phase 0.1).

Post-call: a thin `observe()` wrapper so agent outputs land back in
TCMM with the right `extracted_by=agent:<aid>` provenance (Phase 0.2).

Memoization:
  We cache the rendered TCMM blob per (parent_cid, tcmm_version) so
  parallel sibling sub-calls (Pattern C fanout) share one render.  TTL
  is 5 minutes; longer would risk staleness if the user observes new
  content mid-conversation.  This memoization is in-process; Wave 2
  may move it to Redis if we run multiple agent-runtime replicas.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from ..config import (
    TCMM_ENABLED,
    TCMM_URL,
    TCMM_RENDER_TIMEOUT_S,
    VEILGUARD_INTERNAL_SECRET,
)
from . import tenant

logger = logging.getLogger("agent-runtime.middleware.tcmm")

# IMPORTANT (2026-05-22): we do NOT normalize cache_control on TCMM's
# output.  TCMM has per-provider renderers (`project_tcmm_renderer_architecture`
# memory) that already place cache_control markers correctly for Anthropic.
# The pii-proxy slots TCMM bytes in directly; we mirror that here so the
# Anthropic prompt cache key remains byte-stable across the (proxy → TCMM
# → direct) and (proxy → TCMM → agent-runtime → SDK) paths.  If the SDK
# rewrites markers, the cache validation spike (scripts/spike_cache_validation.py)
# will surface it; defensive utilities still live in middleware/cache_control.py.


# ── Memoization cache ────────────────────────────────────────────────────
# Keyed on (parent_cid, tcmm_version, agent_id).  Agent_id is in the key
# because Director's recall scope (conv + team + blackboard) differs from
# Researcher's (private + team + blackboard) — different blob.
#
# TTL: 5 minutes.  Trade-off: longer = more sibling cache hits; shorter
# = less stale on rapid user observe.

_CACHE_TTL_S = 300.0


@dataclass
class _CachedRender:
    blob: list[dict[str, Any]]
    cached_at: float
    tcmm_version: str


_render_cache: dict[str, _CachedRender] = {}
_cache_lock = asyncio.Lock()


def _cache_key(parent_cid: str, agent_id: str) -> str:
    return f"{parent_cid}:{agent_id}"


async def _purge_expired() -> None:
    """Best-effort cache eviction; called opportunistically on miss."""
    now = time.time()
    expired = [
        k for k, v in _render_cache.items()
        if now - v.cached_at > _CACHE_TTL_S
    ]
    for k in expired:
        _render_cache.pop(k, None)


# ── TCMM service client ──────────────────────────────────────────────────


async def _fetch_tcmm_render(
    *,
    conversation_id: str,
    user_id: str,
    agent_id: str,
    model: str,
) -> tuple[list[dict[str, Any]], str]:
    """Call tcmm-service `/render` and return (blocks, tcmm_version).

    The TCMM render endpoint returns a list of content blocks formatted
    for the requested model's provider conventions.  We accept Anthropic
    shape (a list of `{type: "text", text: "..."}` blocks).

    tcmm_version is an opaque string the service returns; if it changes
    we know the on-disk archive has been updated and we should invalidate.

    Errors:
      - Network timeout → log + return ([], "") so the agent runs without
        TCMM context (degraded mode is better than 503-cascade).
      - 4xx → propagate (probably a config issue worth surfacing).
      - 5xx → log + return ([], "").
    """
    if not TCMM_ENABLED:
        return [], "disabled"

    headers = {}
    if VEILGUARD_INTERNAL_SECRET:
        headers["x-veilguard-internal-secret"] = VEILGUARD_INTERNAL_SECRET

    url = f"{TCMM_URL.rstrip('/')}/render"
    payload = {
        "conversation_id": conversation_id,
        "user_id": user_id,
        "agent_id": agent_id,
        "model": model,
    }

    try:
        async with httpx.AsyncClient(timeout=TCMM_RENDER_TIMEOUT_S) as client:
            r = await client.post(url, json=payload, headers=headers)
        if r.status_code >= 500:
            logger.error(
                f"[tcmm] render returned {r.status_code}; "
                f"body={r.text[:200]!r}; running in degraded mode"
            )
            return [], ""
        if r.status_code >= 400:
            logger.error(
                f"[tcmm] render returned {r.status_code} (config?): "
                f"body={r.text[:200]!r}"
            )
            return [], ""

        data = r.json()
        blocks = data.get("blocks", [])
        version = data.get("version", "")
        return blocks, version

    except httpx.TimeoutException:
        logger.error(
            f"[tcmm] render timed out after {TCMM_RENDER_TIMEOUT_S}s; "
            "running in degraded mode"
        )
        return [], ""
    except Exception as e:
        logger.exception(f"[tcmm] render failed: {e}; degraded mode")
        return [], ""


# ── Public API ───────────────────────────────────────────────────────────


async def get_system_prefix(
    *,
    conversation_id: str,
    user_id: str,
    agent_id: str,
    model: str,
    ttl: str = "1h",  # kept for API compatibility; TCMM owns the actual TTL
) -> list[dict[str, Any]]:
    """Return TCMM's system prefix blocks UNMODIFIED.

    TCMM's per-provider renderer already places cache_control markers
    correctly for the requested model.  We do NOT touch them — TCMM
    is the source of truth for cache placement (per memory:
    `project_tcmm_renderer_architecture`).  The pii-proxy slots TCMM
    output directly today; we mirror that behaviour here so the
    Anthropic prompt cache key remains identical across both paths.

    Memoization: per-process in-memory cache keyed on
    (parent_cid, agent_id) so sibling sub-calls (Pattern C fanout)
    share one HTTP round-trip to TCMM.  TTL 5 min; sub-cids resolve
    to their parent_cid so siblings hit the same cache slot.

    Always returns a list (possibly empty if TCMM is unavailable).
    """
    parent_cid = (
        tenant.parent_cid_from_sub(conversation_id)
        or conversation_id
    )

    cache_k = _cache_key(parent_cid, agent_id)

    async with _cache_lock:
        cached = _render_cache.get(cache_k)
        if cached and (time.time() - cached.cached_at) < _CACHE_TTL_S:
            logger.debug(
                f"[tcmm] render-cache HIT parent_cid={parent_cid[:12]} agent={agent_id}"
            )
            return cached.blob

    # Cache miss: fetch from TCMM and trust the bytes.
    blocks, version = await _fetch_tcmm_render(
        conversation_id=parent_cid,
        user_id=user_id,
        agent_id=agent_id,
        model=model,
    )

    async with _cache_lock:
        await _purge_expired()
        _render_cache[cache_k] = _CachedRender(
            blob=blocks,
            cached_at=time.time(),
            tcmm_version=version,
        )

    n_markers = sum(1 for b in blocks if "cache_control" in b)
    logger.debug(
        f"[tcmm] render-cache MISS parent_cid={parent_cid[:12]} "
        f"agent={agent_id} blocks={len(blocks)} markers={n_markers} version={version}"
    )
    return blocks


def system_prefix_to_string(blocks: list[dict[str, Any]]) -> str:
    """Flatten content blocks back into a single string.

    The Anthropic Python SDK / claude-agent-sdk accepts system_prompt
    as either a string or a list of blocks (with cache_control on
    blocks).  When the SDK wants a string and we have blocks, this
    helper concatenates `text` fields preserving order.

    Loses the cache_control markers — only use this if the SDK call
    site explicitly wants a string.
    """
    return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")


async def observe_agent_output(
    *,
    conversation_id: str,
    user_id: str,
    agent_id: str,
    text: str,
    source: str = "agent",
    channel: str = "agent_private",
) -> bool:
    """Best-effort `observe()` to TCMM with agent_id provenance.

    [F4_AGENT_SCOPED_OBSERVE_2026_05_26]  Was calling a non-existent
    /observe endpoint on tcmm-service — silently 404'd for weeks, so
    NO agent observations ever persisted.  v3/v4 confirmed via the
    cross-conv recall probe: model said "I don't have any memory blocks
    from a startup investigation earlier today."

    Fix: route through /ingest_turn with an agent-scoped namespace
    (`agent/<agent_id>/observations/<user_id>`) instead of the
    per-dispatch sub-cid.  The persona prompts already promise the
    `agent/<id>/observations/` namespace; this makes them truthful.

    Recall integration is a follow-up: a /pre_request that searches
    `<user_id>:<conv_id>` AND `<user_id>:agent/<id>/observations/<user_id>`
    is the next step.  For now persistence is restored; recall via
    explicit /search with namespace=`agent/<id>/observations/<user_id>`
    works today.

    `conversation_id` is kept in the per-item metadata so we can
    cross-link an observation back to the sub-conversation that produced
    it, but the namespace it LIVES in is agent-scoped, not conv-scoped.

    Returns:
        True  — observation persisted in TCMM
        False — observation was NOT persisted (TCMM unreachable, 4xx/5xx,
                empty agent_id, or TCMM disabled).  Callers MUST surface
                this to the LLM so it doesn't spin in a "I observed
                successfully but nothing's there" loop — that's how the
                infinite-replan failure mode happens.  Fixed 2026-05-27
                after a researcher burned 100+ stream events believing
                its observations were persisted when TCMM was down.
    """
    if not TCMM_ENABLED:
        return False
    if not agent_id:
        # Without an agent_id we can't build the scoped namespace —
        # drop rather than fall back to a conv-scoped write (that's
        # exactly the historical bug).
        logger.warning("[tcmm] observe dropped — empty agent_id")
        return False

    headers = {}
    if VEILGUARD_INTERNAL_SECRET:
        headers["x-veilguard-internal-secret"] = VEILGUARD_INTERNAL_SECRET

    # Agent-scoped TCMM conversation_id.  Per the persona contracts
    # (agents/researcher.md etc.), observations land in
    # `agent/<agent_id>/observations/`.  We tack on user_id so the
    # tenant_id filter at the data-access layer still applies.
    agent_conv = f"agent/{agent_id}/observations/{user_id}"

    url = f"{TCMM_URL.rstrip('/')}/ingest_turn"
    # [F4b_INGEST_FIELD_NAMES_2026_05_28] Two silent-drop bugs fixed:
    # (1) TCMM reads `item.get("text")` but we previously sent `content`,
    #     so EVERY observation was skipped at line 1381 in
    #     adapters/veilguard_adapter.py.  Verified live: ingest_turn
    #     replied {"added": 0, "requested": 1} but the middleware
    #     returned True because the HTTP status was 200.
    # (2) TCMM's _ORIGIN_MAP supports "assistant_text"/"tool_use"/
    #     "tool_result"/"system"/"user" — NOT "observation".  Unknown
    #     origins silently fall through to "user" semantics, which is
    #     wrong for agent-authored output.  Use "assistant_text".
    payload = {
        "conversation_id": agent_conv,
        "user_id": user_id,
        "items": [
            {
                "text": text,                       # TCMM contract: `text`
                "origin": "assistant_text",         # in _ORIGIN_MAP
                "extracted_by": f"agent:{agent_id}",
                # [CHANNEL_2026_05_29] spec §6.1 — top-level item field
                # (mirrors extracted_by; NOT metadata). Routes the write
                # to the right `channel` column at ingest; the namespace
                # above stays the cache-stable agent-private path.
                "channel": channel,
                "source_kind": "TOOL_RESULT",
                "metadata": {
                    "source": source,
                    "source_conversation_id": conversation_id,
                },
            }
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=TCMM_RENDER_TIMEOUT_S) as client:
            r = await client.post(url, json=payload, headers=headers)
        if r.status_code >= 400:
            logger.warning(
                f"[tcmm] ingest_turn(observe) returned {r.status_code}; "
                f"agent output not persisted: {r.text[:200]}"
            )
            return False
        ct = r.headers.get("content-type", "")
        body = r.json() if ct.startswith("application/json") else {}
        # [F4b] gate truthy return on `added > 0` so silent ingest-side
        # drops surface to callers and the LLM-facing tool can warn the
        # agent (instead of pretending the observation persisted).
        added = int(body.get("added", 0) or 0)
        requested = int(body.get("requested", 1) or 1)
        if added <= 0:
            logger.warning(
                f"[tcmm] ingest_turn(observe) returned 200 but added={added}/"
                f"{requested}; agent output NOT persisted (likely TCMM "
                f"rejected the shape).  agent={agent_id} ns={agent_conv!r}"
            )
            return False
        logger.debug(
            f"[tcmm] observe persisted: agent={agent_id} "
            f"ns={agent_conv!r} added={added}/{requested}"
        )
        return True
    except Exception as e:
        logger.warning(f"[tcmm] observe failed: {e}; agent output not persisted")
        return False


def invalidate_cache(parent_cid: str, agent_id: Optional[str] = None) -> None:
    """Drop the memoized render for a parent_cid.

    Called by callers that know TCMM state has changed (e.g., after a
    user observes new content mid-conversation).  Pass agent_id to
    invalidate just one agent's view; omit to invalidate all agents
    for that parent_cid.
    """
    if agent_id:
        _render_cache.pop(_cache_key(parent_cid, agent_id), None)
    else:
        keys = [k for k in _render_cache if k.startswith(f"{parent_cid}:")]
        for k in keys:
            _render_cache.pop(k, None)


__all__ = [
    "get_system_prefix",
    "system_prefix_to_string",
    "observe_agent_output",
    "invalidate_cache",
]
