"""Phase 4 — Magentic-One-style task_ledger + progress_ledger (§Phase 4).

Spec: "Magentic-One-style ledger on Tasks: structured `task_ledger`
(facts/guesses/plan) + `progress_ledger` (per-step self-reflection).
Persisted in the Task row's extras."

This module stores the structured data inside `agent_tasks.extras_json`
under two top-level keys:
  extras_json:
    task_ledger:    {facts, guesses, plan, last_updated_ts}
    progress_ledger: [{step, observation, plan_change, ts, by_agent_id}, ...]

API:
  * `read_ledgers(task_id, tenant, user)` → {task_ledger, progress_ledger}
  * `write_task_ledger(task_id, facts, guesses, plan, by_agent_id)`
  * `append_progress(task_id, step, observation, plan_change?, by_agent_id)`

Keeps reads cheap (parse JSON once per call) and writes idempotent
(merge with whatever's already there).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

from .store import LedgerStore, ns_filter

logger = logging.getLogger(__name__)


def _now() -> float:
    return time.time()


def _load_extras(extras_json: Optional[str]) -> dict[str, Any]:
    """Parse extras_json into a dict; tolerate empty/malformed values."""
    if not extras_json:
        return {}
    try:
        d = json.loads(extras_json)
        return d if isinstance(d, dict) else {}
    except (TypeError, ValueError):
        return {}


def _dump_extras(d: dict[str, Any]) -> str:
    return json.dumps(d, default=str)


def read_ledgers(
    *,
    task_id: str,
    tenant_id: str,
    user_id: str,
) -> dict[str, Any]:
    """Return ``{task_ledger, progress_ledger}`` for ``task_id``.

    Empty dicts/lists when the task has no ledger entries yet.
    Returns ``None`` if the task itself doesn't exist.
    """
    tbl = LedgerStore.get().table("agent_tasks")
    arr = (
        tbl.search()
        .where(f"{ns_filter(tenant_id, user_id)} AND id = '{task_id}'")
        .select(["id", "extras_json"])
        .limit(1)
        .to_arrow()
    )
    if arr.num_rows == 0:
        return {"task_ledger": {}, "progress_ledger": []}
    extras = _load_extras(arr.column("extras_json")[0].as_py())
    return {
        "task_ledger":     extras.get("task_ledger", {}),
        "progress_ledger": extras.get("progress_ledger", []),
    }


def write_task_ledger(
    *,
    task_id: str,
    tenant_id: str,
    user_id: str,
    facts: Optional[list[str]] = None,
    guesses: Optional[list[str]] = None,
    plan: Optional[list[str]] = None,
    by_agent_id: str = "",
) -> dict[str, Any]:
    """Overwrite the task_ledger.  Returns the new merged extras_json.

    Magentic-One pattern: the task_ledger is the agent's CURRENT
    understanding — facts (known), guesses (uncertain), plan (next
    steps).  Each call replaces it in full so the agent doesn't have
    to track prior state.  `progress_ledger` is the append-only
    history (use `append_progress`).
    """
    tbl = LedgerStore.get().table("agent_tasks")
    arr = (
        tbl.search()
        .where(f"{ns_filter(tenant_id, user_id)} AND id = '{task_id}'")
        .select(["id", "extras_json"])
        .limit(1)
        .to_arrow()
    )
    if arr.num_rows == 0:
        raise ValueError(f"task {task_id!r} not found")
    extras = _load_extras(arr.column("extras_json")[0].as_py())
    extras["task_ledger"] = {
        "facts":           list(facts or []),
        "guesses":         list(guesses or []),
        "plan":            list(plan or []),
        "last_updated_ts": _now(),
        "last_updated_by": by_agent_id,
    }
    now = _now()
    tbl.update(
        where=f"id = '{task_id}'",
        values={"extras_json": _dump_extras(extras), "updated_ts": now},
    )
    return extras["task_ledger"]


def append_progress(
    *,
    task_id: str,
    tenant_id: str,
    user_id: str,
    step: str,
    observation: str,
    plan_change: Optional[str] = None,
    by_agent_id: str = "",
) -> list[dict[str, Any]]:
    """Append one step to the progress_ledger.  Returns the full
    progress_ledger after append (so callers can render the timeline).

    `step` is a short tag — "researched POPIA s14", "wrote draft",
    "submitted to critic".  `observation` is what the agent learned
    or noticed at that step.  `plan_change` is optional — set when
    the step caused the agent to revise its plan (then the caller
    typically follows up with `write_task_ledger` to update the
    plan).
    """
    tbl = LedgerStore.get().table("agent_tasks")
    arr = (
        tbl.search()
        .where(f"{ns_filter(tenant_id, user_id)} AND id = '{task_id}'")
        .select(["id", "extras_json"])
        .limit(1)
        .to_arrow()
    )
    if arr.num_rows == 0:
        raise ValueError(f"task {task_id!r} not found")
    extras = _load_extras(arr.column("extras_json")[0].as_py())
    progress = list(extras.get("progress_ledger") or [])
    progress.append({
        "step":         step,
        "observation":  observation,
        "plan_change":  plan_change,
        "ts":           _now(),
        "by_agent_id":  by_agent_id,
    })
    extras["progress_ledger"] = progress
    tbl.update(
        where=f"id = '{task_id}'",
        values={"extras_json": _dump_extras(extras), "updated_ts": _now()},
    )
    return progress


__all__ = [
    "read_ledgers",
    "write_task_ledger",
    "append_progress",
]
