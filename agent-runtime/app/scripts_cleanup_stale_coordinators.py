"""One-shot cleanup of stale pre-guard task junk + coordinator backfill.

Run inside the agent-runtime container (MODULE form — the script does
`from app.ledger...`, so it must run with /app on sys.path; the old
`python /app/app/scripts_…py` form fails with ModuleNotFoundError: app.ledger):
    docker exec -w /app veilguard-agent-runtime-1 python -m app.scripts_cleanup_stale_coordinators <tenant_id>

What it does for the given (tenant_id == user_id) namespace:
  1. Cancels stale director-OWNED subtasks (parent_id set, owner_id=director,
     non-terminal).  These predate the SUBTASK_OWNER_GUARD and can never be
     dispatched (the inbox poller skips director-owned rows), so they hang OPEN.
  2. Cancels director/team-lead OWNED top-level coordinators that have NO
     children and are non-terminal (empty duplicate fan-out roots that never
     got decomposed).
  3. Backfill: for every remaining non-terminal coordinator (has children, all
     children terminal, >=1 done), invokes the autoclose path so it flips done.

Cancellation goes through update_status (open->cancelled), which fires the
[PARENT_AUTOCLOSE_2026_05_29] hook — so cancelling the last open child of a
coordinator will itself close the coordinator.
"""
import sys
from app.ledger.store import LedgerStore, ns_filter
from app.ledger import tasks as T

tenant = sys.argv[1] if len(sys.argv) > 1 else "69c4468a1fde1abc19c7835c"
user = tenant

tbl = LedgerStore.get().table("agent_tasks")
rows = [r for r in tbl.to_arrow().to_pylist()
        if r.get("tenant_id") == tenant and r.get("user_id") == user]
TERMINAL = {"done", "cancelled"}
by_parent = {}
for r in rows:
    by_parent.setdefault(r.get("parent_id"), []).append(r)

cancelled, closed = [], []


def _cancel(tid, why):
    try:
        T.update_status(task_id=tid, tenant_id=tenant, user_id=user,
                        new_status="cancelled", actor_agent_id="system:cleanup")
        cancelled.append((tid, why))
    except Exception as e:
        print(f"  ! cancel {tid} failed: {e}")


# (1) + (2) cancel stale junk
for r in rows:
    if r.get("status") in TERMINAL:
        continue
    tid, owner, pid = r["id"], (r.get("owner_id") or ""), r.get("parent_id")
    kids = by_parent.get(tid, [])
    if pid and owner == "director":
        _cancel(tid, "pre-guard director-owned subtask")
    elif (not pid) and owner in ("director", "team-lead") and not kids:
        _cancel(tid, "empty stale coordinator (no children)")

# (3) backfill autoclose for coordinators whose children are now all terminal
# Re-read after cancellations so freshly-cancelled children are reflected.
rows2 = [r for r in tbl.to_arrow().to_pylist()
         if r.get("tenant_id") == tenant and r.get("user_id") == user]
by_parent2 = {}
for r in rows2:
    by_parent2.setdefault(r.get("parent_id"), []).append(r)
for r in rows2:
    if r.get("status") in TERMINAL:
        continue
    kids = by_parent2.get(r["id"], [])
    if not kids:
        continue
    kstat = [k.get("status") for k in kids]
    if all(s in TERMINAL for s in kstat) and "done" in kstat:
        # nudge via a child to reuse the exact hook logic
        any_child = kids[0]["id"]
        got = T._maybe_autoclose_parents(
            child_task_id=any_child, tenant_id=tenant, user_id=user,
            actor_agent_id="system:cleanup")
        closed += got

print(f"CANCELLED {len(cancelled)}:")
for tid, why in cancelled:
    print(f"  {tid[:8]}  {why}")
print(f"AUTO-CLOSED {len(closed)}: {[c[:8] for c in closed]}")
