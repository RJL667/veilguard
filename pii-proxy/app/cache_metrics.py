"""Per-tenant cache telemetry rollup (Phase 7, step 2).

Tracks Anthropic prompt-cache usage per tenant in a rolling in-process
deque so we can answer:

  * What's this tenant's cache hit rate?
  * What's their write amplification (writes per total input)?
  * Per-tier breakdown when the two-tier layout is in effect.

Feeds the circuit breaker (step 6) and serves a /cache/stats endpoint
for ops visibility.

The state lives in-process (per worker). For a multi-worker deployment,
sum across workers at the dashboard layer or migrate to a shared store
(Redis / SQLite WAL) — that's a deliberate "later" decision; right now
single-worker rollup is sufficient for the design we're validating.

Schema of each recorded sample:

    {
      "ts": 1714862400.0,             # epoch seconds (time.time())
      "tenant_id": "rudolf@…",        # or "" for unattributed
      "conv_id": "conv_abc",          # short id
      "model": "claude-opus-4-7",     # response.model
      "cache_create": 5234,           # response.usage.cache_creation_input_tokens
      "cache_read":   18920,          # response.usage.cache_read_input_tokens
      "input_tokens": 412,            # response.usage.input_tokens (uncached)
      "output_tokens": 880,           # response.usage.output_tokens
      "tier_hints": {                 # optional per-tier diagnostics
        "preamble_bytes": 9216,
        "stable_bytes":   4108,
        "working_bytes":  3200,
      },
    }

Derived metrics (computed on demand in `get_tenant_stats`):

    total_in   = sum(cache_create + cache_read + input_tokens)
    hit_rate   = sum(cache_read)   / total_in
    write_rate = sum(cache_create) / total_in
    write_amplification = sum(cache_create) / max(1, sum(cache_read))
        — interpretive: ≥1.0 means caching is hurting you for this
        tenant (more writes than reads); below 0.1 means caching is
        paying off well.
    effective_token_multiplier (per the architectural conversation):
        ( 1.0 * input + 1.25 * cache_create_5m
                      + 2.0 * cache_create_1h
                      + 0.1 * cache_read ) / total_in
        — currently we don't split create by TTL in the audit signal,
        so we approximate using a configurable WRITE_COST_WEIGHT
        (defaults to 1.6, midpoint of 1.25/2.0 since our default split
        is preamble+stable at 1h and working at 5m).
"""
from __future__ import annotations

import time
import threading
from collections import deque
from typing import Any


# ── Configuration knobs ─────────────────────────────────────────────────
# Per-tenant rolling window size. Each sample is ~200 bytes, so 1000
# samples * 1000 tenants = ~200 MB max in the worst case. Adjust if
# memory pressure becomes a concern.
_MAX_SAMPLES_PER_TENANT = 1000

# Hard cap on number of distinct tenants tracked. When exceeded, the
# least-recently-seen tenant is dropped. Prevents unbounded growth in
# multi-tenant scenarios with one-shot users.
_MAX_TENANTS = 5000

# Blended write cost used by effective_token_multiplier. Set between
# 1.25 (all writes go to the 5m tier) and 2.0 (all writes go to the 1h
# tier). The Phase 7 default of stable=1h / working=5m on roughly-equal
# byte counts puts the blended cost near 1.6 — adjust if your traffic
# skews. Override via env var CACHE_WRITE_COST_WEIGHT.
import os as _os
try:
    _WRITE_COST_WEIGHT = float(_os.environ.get("CACHE_WRITE_COST_WEIGHT", "1.6"))
except (TypeError, ValueError):
    _WRITE_COST_WEIGHT = 1.6

# ── Cache-thrash circuit breaker thresholds (step 6) ───────────────────
# A tenant is "thrashing" when caching is strictly costing them money:
# their hit rate is too low to amortize the 1.25× / 2× write premium.
# When the breaker trips, the request path strips cache_control entirely
# for that tenant — they pay 1.0× input on every request instead of the
# perpetual ~1.6× write rate. Re-evaluated on every request; if the
# tenant's pattern recovers, caching turns back on.
#
# CIRCUIT_MIN_SAMPLES — don't trip on small samples. A fresh tenant with
#                      2 requests and 0 hits is normal cold-start state,
#                      not thrash. Default 20.
# CIRCUIT_HIT_RATE_FLOOR — below this hit_rate over the window, suspect.
#                         Default 0.10 (10%).
# CIRCUIT_WRITE_AMP_CEILING — above this write_amplification, suspect.
#                            Default 1.0 (more writes than reads).
# CIRCUIT_WINDOW_SECONDS — rolling window for the decision. Default 1 hour.
# Both signals must agree before the breaker trips (AND, not OR).
try:
    _CIRCUIT_MIN_SAMPLES = int(_os.environ.get("CACHE_CIRCUIT_MIN_SAMPLES", "20"))
except (TypeError, ValueError):
    _CIRCUIT_MIN_SAMPLES = 20
try:
    _CIRCUIT_HIT_RATE_FLOOR = float(_os.environ.get("CACHE_CIRCUIT_HIT_RATE_FLOOR", "0.10"))
except (TypeError, ValueError):
    _CIRCUIT_HIT_RATE_FLOOR = 0.10
try:
    _CIRCUIT_WRITE_AMP_CEILING = float(_os.environ.get("CACHE_CIRCUIT_WRITE_AMP_CEILING", "1.0"))
except (TypeError, ValueError):
    _CIRCUIT_WRITE_AMP_CEILING = 1.0
try:
    _CIRCUIT_WINDOW_SECONDS = float(_os.environ.get("CACHE_CIRCUIT_WINDOW_SECONDS", "3600"))
except (TypeError, ValueError):
    _CIRCUIT_WINDOW_SECONDS = 3600.0


# ── State ────────────────────────────────────────────────────────────────
# Map tenant_id -> deque of samples. OrderedDict-like behaviour via
# move_to_end on access (Python 3.7+ dicts preserve insertion order).
_state: dict[str, deque] = {}
_state_lock = threading.RLock()


# ── Public API ───────────────────────────────────────────────────────────
def record_request(
    tenant_id: str,
    conv_id: str,
    *,
    model: str | None = None,
    cache_create: int | None = None,
    cache_read: int | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    tier_hints: dict[str, Any] | None = None,
) -> None:
    """Record one Anthropic-response usage sample for a tenant.

    All counters None-tolerant — caller doesn't have to filter; missing
    fields are treated as 0 in aggregates. Safe to call from request
    hot path: ~5 µs per call, no I/O.
    """
    if not tenant_id:
        tenant_id = ""  # unattributed bucket
    sample = {
        "ts": time.time(),
        "tenant_id": tenant_id,
        "conv_id": (conv_id or "")[:24],
        "model": model or "",
        "cache_create": int(cache_create or 0),
        "cache_read": int(cache_read or 0),
        "input_tokens": int(input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
        "tier_hints": tier_hints or {},
    }
    with _state_lock:
        dq = _state.get(tenant_id)
        if dq is None:
            # Evict LRU tenant if at cap.
            if len(_state) >= _MAX_TENANTS:
                try:
                    _state.pop(next(iter(_state)))
                except StopIteration:
                    pass
            dq = deque(maxlen=_MAX_SAMPLES_PER_TENANT)
            _state[tenant_id] = dq
        else:
            # Move-to-end via re-insert (preserves LRU eviction order).
            try:
                _state.pop(tenant_id)
                _state[tenant_id] = dq
            except KeyError:
                pass
        dq.append(sample)


def get_tenant_stats(
    tenant_id: str,
    window_seconds: float | None = None,
) -> dict[str, Any]:
    """Aggregate per-tenant cache metrics over an optional time window.

    ``window_seconds=None`` aggregates the entire rolling window.
    Otherwise only samples newer than ``time.time() - window_seconds``
    are included.

    Returns a dict with sample counts, raw token sums, and derived
    rates. Safe to call from the circuit breaker (step 6) — O(N) over
    the tenant's window.
    """
    cutoff = (time.time() - window_seconds) if window_seconds else None
    with _state_lock:
        dq = _state.get(tenant_id)
        samples = list(dq) if dq else []

    if cutoff is not None:
        samples = [s for s in samples if s["ts"] >= cutoff]

    n = len(samples)
    if n == 0:
        return {
            "tenant_id": tenant_id,
            "n_samples": 0,
            "window_seconds": window_seconds,
            "cache_create_total": 0,
            "cache_read_total": 0,
            "input_tokens_total": 0,
            "output_tokens_total": 0,
            "hit_rate": None,
            "write_rate": None,
            "write_amplification": None,
            "effective_token_multiplier": None,
        }

    cc = sum(s["cache_create"] for s in samples)
    cr = sum(s["cache_read"] for s in samples)
    inp = sum(s["input_tokens"] for s in samples)
    out = sum(s["output_tokens"] for s in samples)
    total_in = cc + cr + inp

    if total_in <= 0:
        hit_rate = None
        write_rate = None
        eff = None
    else:
        hit_rate = cr / total_in
        write_rate = cc / total_in
        # Effective amortized token cost vs uncached baseline.
        # uncached baseline = total_in (each token billed at 1.0×).
        # actual cost = 1.0×inp + W×cc + 0.1×cr.
        actual = inp + (_WRITE_COST_WEIGHT * cc) + (0.1 * cr)
        eff = actual / total_in

    write_amp = (cc / cr) if cr > 0 else None

    return {
        "tenant_id": tenant_id,
        "n_samples": n,
        "window_seconds": window_seconds,
        "cache_create_total": cc,
        "cache_read_total": cr,
        "input_tokens_total": inp,
        "output_tokens_total": out,
        "hit_rate": hit_rate,
        "write_rate": write_rate,
        "write_amplification": write_amp,
        "effective_token_multiplier": eff,
    }


def list_tenants() -> list[str]:
    """Return tenant IDs currently held in the rolling state (LRU order).
    """
    with _state_lock:
        return list(_state.keys())


def get_all_stats(window_seconds: float | None = None) -> list[dict[str, Any]]:
    """Stats for every tenant in the rolling state."""
    return [get_tenant_stats(t, window_seconds) for t in list_tenants()]


def is_cache_thrashing(tenant_id: str) -> tuple[bool, dict]:
    """Decide whether to disable cache_control for this tenant.

    Returns (should_strip, reason_dict). Cheap (O(N) over tenant window).
    Caller pattern in the proxy:

        from app import cache_metrics
        strip, why = cache_metrics.is_cache_thrashing(tenant_id)
        if strip:
            logger.info(f"[CACHE-CIRCUIT] tripped tenant={tenant_id[:8]} {why}")
            _strip_cache_control(data)
        else:
            _apply_anthropic_cache(data)

    Decision policy (both signals required, AND):
      1. At least CIRCUIT_MIN_SAMPLES samples in the window — small
         samples are noisy. New tenants get the benefit of caching by
         default; the breaker only kicks in once we have evidence.
      2. hit_rate <= CIRCUIT_HIT_RATE_FLOOR.
      3. write_amplification >= CIRCUIT_WRITE_AMP_CEILING (or cache_read
         == 0, which makes amplification mathematically infinite).
    """
    stats = get_tenant_stats(tenant_id, window_seconds=_CIRCUIT_WINDOW_SECONDS)
    n = stats["n_samples"]
    if n < _CIRCUIT_MIN_SAMPLES:
        return (False, {"reason": "insufficient_samples", "n_samples": n,
                        "required": _CIRCUIT_MIN_SAMPLES})

    hit_rate = stats["hit_rate"]
    write_amp = stats["write_amplification"]

    # write_amp is None when cache_read == 0; treat that as "infinite"
    # amplification — strictly worse than not caching at all if any
    # writes happened.
    inf_amp = (write_amp is None and stats["cache_create_total"] > 0)
    amp_bad = inf_amp or (write_amp is not None and write_amp >= _CIRCUIT_WRITE_AMP_CEILING)

    hit_bad = (hit_rate is not None and hit_rate <= _CIRCUIT_HIT_RATE_FLOOR)

    if hit_bad and amp_bad:
        return (True, {
            "reason": "cache_thrash",
            "hit_rate": hit_rate,
            "write_amplification": write_amp if write_amp is not None else "inf",
            "n_samples": n,
            "thresholds": {
                "hit_rate_floor": _CIRCUIT_HIT_RATE_FLOOR,
                "write_amp_ceiling": _CIRCUIT_WRITE_AMP_CEILING,
            },
        })
    return (False, {
        "reason": "ok",
        "hit_rate": hit_rate,
        "write_amplification": write_amp,
        "n_samples": n,
    })


def reset(tenant_id: str | None = None) -> None:
    """Clear stored samples. Pass tenant_id=None to clear ALL tenants
    (for tests). Otherwise clears just that tenant's window."""
    with _state_lock:
        if tenant_id is None:
            _state.clear()
        else:
            _state.pop(tenant_id, None)


# ── Convenience: extract from response usage ────────────────────────────
def record_from_usage(
    usage: dict | None,
    *,
    tenant_id: str,
    conv_id: str,
    model: str | None = None,
    tier_hints: dict[str, Any] | None = None,
) -> None:
    """Extract usage fields from an Anthropic response.usage dict and
    record. Safe to call with usage=None (no-op).
    """
    if not isinstance(usage, dict):
        return
    record_request(
        tenant_id=tenant_id,
        conv_id=conv_id,
        model=model,
        cache_create=usage.get("cache_creation_input_tokens"),
        cache_read=usage.get("cache_read_input_tokens"),
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        tier_hints=tier_hints,
    )
