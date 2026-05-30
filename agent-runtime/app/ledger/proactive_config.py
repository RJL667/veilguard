"""Per-tenant proactive-stream config CRUD (§3.7.3, Phase 3).

One row per (tenant_id, user_id).  Missing row → returns defaults.
The DreamScanner + LifecycleWorker + OutcomesWorker should check
``get_or_default(tenant_id, user_id)`` before doing per-tenant work,
and skip if ``proactive_stream_enabled=False`` or ``paused=True``.

Defaults are conservative — generous enough to be useful, capped
enough to prevent the $24/day-at-100-tenants surprise the spec
calls out.

Read-side has zero Lance writes; admin/upsert mutations are
explicit + audit-logged via `updated_ts`.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, asdict
from typing import Any, Optional

from .store import LedgerStore


# Default values — spec §3.7.3
DEFAULTS = {
    "proactive_stream_enabled":          True,
    "proactive_cycles_per_day":          12,    # every 2h
    "proactive_approval_cap_per_day":    20,
    "cost_ceiling_per_tenant_per_day_usd": 5.0,
    "paused":                            False,
    "paused_reason":                     None,
    "paused_at_ts":                      None,
}


@dataclass
class ProactiveConfig:
    """In-memory view of a tenant_proactive_config row.

    Use ``get_or_default(tenant_id, user_id)`` for the read path —
    it gracefully synthesises a default config if no row exists.
    """
    tenant_id:                            str
    user_id:                              str
    proactive_stream_enabled:             bool
    proactive_cycles_per_day:             int
    proactive_approval_cap_per_day:       int
    cost_ceiling_per_tenant_per_day_usd:  float
    paused:                               bool
    paused_reason:                        Optional[str]
    paused_at_ts:                         Optional[float]
    is_default:                           bool = False  # True when synthesised

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def stream_active(self) -> bool:
        """One-property check workers should use: enabled AND not paused."""
        return bool(self.proactive_stream_enabled) and not bool(self.paused)


def _new_id() -> str:
    return f"pcfg-{uuid.uuid4().hex[:12]}"


def _now() -> float:
    return time.time()


def get_or_default(tenant_id: str, user_id: str) -> ProactiveConfig:
    """Return the proactive config for (tenant_id, user_id).

    If no row exists, returns a ProactiveConfig populated with
    ``DEFAULTS`` and ``is_default=True``.  Workers can then treat
    "no config row" identically to "default config row" — no
    None-handling at call sites.
    """
    try:
        tbl = LedgerStore.get().table("tenant_proactive_config")
        arr = (
            tbl.search()
            .where(
                f"tenant_id = '{tenant_id}' AND user_id = '{user_id}'"
            )
            .limit(1)
            .to_arrow()
        )
        if arr.num_rows > 0:
            return ProactiveConfig(
                tenant_id=arr.column("tenant_id")[0].as_py(),
                user_id=arr.column("user_id")[0].as_py(),
                proactive_stream_enabled=bool(
                    arr.column("proactive_stream_enabled")[0].as_py()
                ),
                proactive_cycles_per_day=int(
                    arr.column("proactive_cycles_per_day")[0].as_py()
                ),
                proactive_approval_cap_per_day=int(
                    arr.column("proactive_approval_cap_per_day")[0].as_py()
                ),
                cost_ceiling_per_tenant_per_day_usd=float(
                    arr.column("cost_ceiling_per_tenant_per_day_usd")[0].as_py()
                ),
                paused=bool(arr.column("paused")[0].as_py()),
                paused_reason=arr.column("paused_reason")[0].as_py(),
                paused_at_ts=arr.column("paused_at_ts")[0].as_py(),
                is_default=False,
            )
    except Exception:
        # Table might not exist yet on first call — fall through to defaults
        pass
    return ProactiveConfig(
        tenant_id=tenant_id,
        user_id=user_id,
        proactive_stream_enabled=DEFAULTS["proactive_stream_enabled"],
        proactive_cycles_per_day=DEFAULTS["proactive_cycles_per_day"],
        proactive_approval_cap_per_day=DEFAULTS["proactive_approval_cap_per_day"],
        cost_ceiling_per_tenant_per_day_usd=DEFAULTS["cost_ceiling_per_tenant_per_day_usd"],
        paused=DEFAULTS["paused"],
        paused_reason=DEFAULTS["paused_reason"],
        paused_at_ts=DEFAULTS["paused_at_ts"],
        is_default=True,
    )


# Sentinel for "leave this field at its current value".  Lets
# ``None`` mean "explicitly clear" for nullable string fields like
# ``paused_reason`` — important for the resume() workflow where we
# want to NULL out the reason on stream-resume.
_UNSET = object()


def upsert(
    *,
    tenant_id: str,
    user_id: str,
    proactive_stream_enabled=_UNSET,
    proactive_cycles_per_day=_UNSET,
    proactive_approval_cap_per_day=_UNSET,
    cost_ceiling_per_tenant_per_day_usd=_UNSET,
    paused=_UNSET,
    paused_reason=_UNSET,
) -> ProactiveConfig:
    """Create or update the per-tenant config.

    Sentinel ``_UNSET`` means "leave the field unchanged".  ``None``
    for nullable fields (``paused_reason``) means "explicitly clear it"
    — this is what ``resume()`` needs to wipe the pause reason.
    Returns the row as it exists after the write.
    """
    tbl = LedgerStore.get().table("tenant_proactive_config")
    current = get_or_default(tenant_id, user_id)
    now = _now()
    fields = {
        "proactive_stream_enabled":
            current.proactive_stream_enabled if proactive_stream_enabled is _UNSET else bool(proactive_stream_enabled),
        "proactive_cycles_per_day":
            current.proactive_cycles_per_day if proactive_cycles_per_day is _UNSET else int(proactive_cycles_per_day),
        "proactive_approval_cap_per_day":
            current.proactive_approval_cap_per_day if proactive_approval_cap_per_day is _UNSET else int(proactive_approval_cap_per_day),
        "cost_ceiling_per_tenant_per_day_usd":
            current.cost_ceiling_per_tenant_per_day_usd if cost_ceiling_per_tenant_per_day_usd is _UNSET else float(cost_ceiling_per_tenant_per_day_usd),
        "paused":
            current.paused if paused is _UNSET else bool(paused),
        "paused_reason":
            current.paused_reason if paused_reason is _UNSET else paused_reason,
    }
    # Stamp paused_at_ts when transitioning to paused
    new_paused_at = current.paused_at_ts
    if paused is not _UNSET:
        if paused and not current.paused:
            new_paused_at = now
        elif not paused:
            new_paused_at = None

    if current.is_default:
        # Insert
        row = {
            "id":         _new_id(),
            "tenant_id":  tenant_id,
            "user_id":    user_id,
            **fields,
            "paused_at_ts": new_paused_at,
            "created_ts": now,
            "updated_ts": now,
            "extras_json": None,
        }
        tbl.add([row])
    else:
        # Update — Lance lacks a unique-key concept here, so we update
        # by (tenant_id, user_id).  Only one row per pair by convention.
        tbl.update(
            where=f"tenant_id = '{tenant_id}' AND user_id = '{user_id}'",
            values={
                **fields,
                "paused_at_ts": new_paused_at,
                "updated_ts": now,
            },
        )
    return get_or_default(tenant_id, user_id)


def pause(*, tenant_id: str, user_id: str, reason: str) -> ProactiveConfig:
    """Convenience: pause the proactive stream with a reason."""
    return upsert(
        tenant_id=tenant_id, user_id=user_id,
        paused=True, paused_reason=reason,
    )


def resume(*, tenant_id: str, user_id: str) -> ProactiveConfig:
    """Convenience: clear pause + reason."""
    return upsert(
        tenant_id=tenant_id, user_id=user_id,
        paused=False, paused_reason=None,
    )


__all__ = [
    "DEFAULTS",
    "ProactiveConfig",
    "get_or_default",
    "upsert",
    "pause",
    "resume",
]
