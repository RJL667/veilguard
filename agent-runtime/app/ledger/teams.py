"""Phase 7.5 — agent_teams CRUD + budget enforcement.

A Team is a named bundle of agents with a shared lead, a budget, and
a cost ceiling.  Two roles:

  1. **Cost envelope.**  Director's `create_task` validator rejects
     new work assigned to a team whose attributed cost has crossed
     `budget_usd × budget_cap`.  The cost source of truth is
     `sum(agent_tasks.cost_attributed_usd WHERE team_id = <id>)`;
     `agent_teams.cost_attributed_cached_usd` is a denormalised
     snapshot for the sidebar / Director view.

  2. **Organisational scaling.**  Once `agent_teams` exists, the
     `team-lead` persona can act as a mini-Director scoped to one
     team — routing within the team, owning the team's review queue,
     reporting up.  Spec §7.5.

API surface kept narrow to match the existing ledger style:

  create_team(...)          — insert one row; returns team_id
  get_team(team_id, ...)    — fetch one row
  list_teams(...)           — enumerate teams for (tenant, user)
  add_member(team_id, ...)  — append agent_id; idempotent
  team_cost_attributed(...) — sum live + return cached recomputed value
  budget_exceeded(...)      — bool guard the Director can pre-check
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Optional

from .store import LedgerStore, ns_filter

logger = logging.getLogger("agent-runtime.ledger.teams")


# Default soft cap multiplier on budget_usd.  budget_cap=1.0 means
# "block at exactly budget"; 1.2 = 20% slack.  Operator-tunable per team.
DEFAULT_BUDGET_CAP = 1.0


def _now() -> float:
    return time.time()


def _new_id() -> str:
    return f"team-{uuid.uuid4().hex[:12]}"


# ── CRUD ────────────────────────────────────────────────────────────────


def create_team(
    *,
    tenant_id: str,
    user_id: str,
    name: str,
    lead_agent_id: str,
    member_agent_ids: Optional[list[str]] = None,
    budget_usd: float = 0.0,
    budget_cap: Optional[float] = None,
    created_by_agent_id: Optional[str] = None,
    extras_json: Optional[str] = None,
) -> str:
    """Insert a new team.  Returns the new team_id.

    Validation:
      - `name` non-empty
      - `lead_agent_id` non-empty (Director may set later via update,
        but at creation we want a clear ownership root)
      - `budget_usd >= 0` (0 = "no spend allowed"; useful for stub teams
        whose work is gated until the user funds them)
      - `budget_cap >= 1.0` (must allow at least 100% of budget)
    """
    if not name.strip():
        raise ValueError("create_team requires non-empty name")
    if not lead_agent_id.strip():
        raise ValueError("create_team requires lead_agent_id")
    if budget_usd < 0:
        raise ValueError(f"budget_usd must be >= 0, got {budget_usd}")
    cap = float(budget_cap or DEFAULT_BUDGET_CAP)
    if cap < 1.0:
        raise ValueError(f"budget_cap must be >= 1.0, got {cap}")

    team_id = _new_id()
    now = _now()
    tbl = LedgerStore.get().table("agent_teams")
    tbl.add([{
        "id":                            team_id,
        "kind":                          "team",
        "status":                        "active",
        "parent_id":                     None,
        "lineage_chain":                 [],
        "tenant_id":                     tenant_id,
        "user_id":                       user_id,
        "created_by_agent_id":           created_by_agent_id,
        "created_ts":                    now,
        "updated_ts":                    now,
        "cost_attributed_usd":           0.0,
        "name":                          name.strip(),
        "lead_agent_id":                 lead_agent_id.strip(),
        "member_agent_ids":              list(member_agent_ids or []),
        "budget_usd":                    float(budget_usd),
        "budget_cap":                    cap,
        "cost_attributed_cached_usd":    0.0,
        "cost_recomputed_ts":            now,
        "extras_json":                   extras_json,
    }])
    try:
        from ..events import broadcast
        broadcast({
            "type":        "team_created",
            "tenant_id":   tenant_id, "user_id": user_id,
            "id":          team_id, "name": name,
            "lead_agent_id": lead_agent_id,
        })
    except Exception:
        pass
    return team_id


def get_team(
    team_id: str, tenant_id: str, user_id: str,
) -> dict[str, Any] | None:
    """Fetch one team row, or None."""
    try:
        tbl = LedgerStore.get().table("agent_teams")
        arr = (
            tbl.search()
            .where(
                f"{ns_filter(tenant_id, user_id)} AND id = '{team_id}'"
            )
            .limit(1)
            .to_arrow()
        )
        if arr.num_rows == 0:
            return None
        return {col: arr.column(col)[0].as_py() for col in arr.column_names}
    except Exception as e:
        logger.warning(f"[teams] get_team({team_id}) failed: {e}")
        return None


def list_teams(
    *, tenant_id: str, user_id: str, limit: int = 200,
) -> list[dict[str, Any]]:
    """All teams for (tenant, user)."""
    try:
        tbl = LedgerStore.get().table("agent_teams")
        arr = (
            tbl.search()
            .where(ns_filter(tenant_id, user_id))
            .limit(limit)
            .to_arrow()
        )
    except Exception as e:
        logger.warning(f"[teams] list_teams failed: {e}")
        return []
    return [
        {col: arr.column(col)[i].as_py() for col in arr.column_names}
        for i in range(arr.num_rows)
    ]


def add_member(
    *, team_id: str, tenant_id: str, user_id: str, agent_id: str,
) -> bool:
    """Append `agent_id` to a team's `member_agent_ids`.  Idempotent —
    re-adding an existing member returns True without modification.
    Returns False if the team doesn't exist or the write fails."""
    if not agent_id.strip():
        raise ValueError("add_member requires non-empty agent_id")
    team = get_team(team_id, tenant_id, user_id)
    if team is None:
        return False
    members = list(team.get("member_agent_ids") or [])
    if agent_id in members:
        return True
    members.append(agent_id)
    try:
        tbl = LedgerStore.get().table("agent_teams")
        tbl.update(
            where=(
                f"{ns_filter(tenant_id, user_id)} AND id = '{team_id}'"
            ),
            values={
                "member_agent_ids": members,
                "updated_ts":       _now(),
            },
        )
        return True
    except Exception as e:
        logger.warning(f"[teams] add_member failed for {team_id}: {e}")
        return False


# ── Budget enforcement ──────────────────────────────────────────────────


def team_cost_attributed(
    *, team_id: str, tenant_id: str, user_id: str,
    refresh_cache: bool = True,
) -> float:
    """Sum `cost_attributed_usd` across every `agent_tasks` row whose
    `team_id` matches.  This is the live source of truth.

    `refresh_cache=True` (default) writes back the rollup to
    `agent_teams.cost_attributed_cached_usd` + `cost_recomputed_ts` so
    sidebar reads can be cheap.
    """
    try:
        tbl = LedgerStore.get().table("agent_tasks")
        arr = (
            tbl.search()
            .where(
                f"{ns_filter(tenant_id, user_id)} AND team_id = '{team_id}'"
            )
            .select(["cost_attributed_usd"])
            .limit(50_000)
            .to_arrow()
        )
    except Exception as e:
        logger.warning(f"[teams] cost scan failed for {team_id}: {e}")
        return 0.0
    total = 0.0
    for i in range(arr.num_rows):
        try:
            total += float(arr.column("cost_attributed_usd")[i].as_py() or 0.0)
        except Exception:
            continue

    if refresh_cache:
        try:
            t_tbl = LedgerStore.get().table("agent_teams")
            t_tbl.update(
                where=(
                    f"{ns_filter(tenant_id, user_id)} AND id = '{team_id}'"
                ),
                values={
                    "cost_attributed_cached_usd": float(total),
                    "cost_recomputed_ts":         _now(),
                },
            )
        except Exception as e:
            logger.debug(f"[teams] cost cache write failed: {e}")

    return total


def budget_exceeded(
    *, team_id: str, tenant_id: str, user_id: str,
    additional_usd: float = 0.0,
) -> tuple[bool, float, float]:
    """Returns (exceeded, attributed_usd, ceiling_usd).

    `additional_usd` lets the caller test "would adding ~$X to this
    team's queue cross the line?" without writing first.  Director
    uses this in `create_task` to reject upfront.

    Ceiling = `budget_usd × budget_cap`.  Default cap = 1.0 → block
    at exactly the budget.
    """
    team = get_team(team_id, tenant_id, user_id)
    if team is None:
        return False, 0.0, 0.0
    attributed = team_cost_attributed(
        team_id=team_id, tenant_id=tenant_id, user_id=user_id,
        refresh_cache=False,
    )
    budget = float(team.get("budget_usd") or 0.0)
    cap    = float(team.get("budget_cap") or DEFAULT_BUDGET_CAP)
    ceiling = budget * cap
    return (attributed + additional_usd) >= ceiling, attributed, ceiling


__all__ = [
    "DEFAULT_BUDGET_CAP",
    "create_team",
    "get_team",
    "list_teams",
    "add_member",
    "team_cost_attributed",
    "budget_exceeded",
]
