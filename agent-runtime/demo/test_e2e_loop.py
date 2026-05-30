"""End-to-end loop test against the LIVE agent-runtime container.

Injects a task directly into the running container's Lance ledger
(no Director LLM call), then watches the inbox poller pick it up and
drive it through the lifecycle:

  open(builder) → in_progress → review(critic-prose) → done

Run inside the container:
    docker exec veilguard-agent-runtime-1 python -m demo.test_e2e_loop

The script polls the ledger every ~3 sec for up to N minutes and
prints status transitions, comments, and file-on-disk verification.
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path

# Resolve `app.*` imports.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.ledger import tasks as tasks_mod
from app.ledger import comments as comments_mod
from app.ledger.store import LedgerStore, ns_filter


# Match the same user_id the daemon is paired as.  When the Builder
# calls write_file, tool_dispatcher.Path 2 forwards x-user-id from the
# tenant context — the daemon must be the one connected with this id.
DAEMON_USER_ID = os.environ.get("VEILGUARD_DEMO_USER_ID", "69c4468a1fde1abc19c7835c")
TENANT_ID      = os.environ.get("VEILGUARD_DEMO_TENANT_ID", DAEMON_USER_ID)

# Target path the Builder will create.  Daemon translates this to the
# Windows host filesystem.
TARGET_PATH = os.environ.get(
    "VEILGUARD_DEMO_TARGET_PATH",
    "C:/Users/rudol/Documents/veilguard/tmp/e2e-loop.txt",
)
TARGET_CONTENT = "hi from end-to-end multi-agent loop\n"


def main() -> None:
    print("=" * 72)
    print("  E2E loop test — Director skipped, task injected directly")
    print("=" * 72)
    print(f"  tenant_id: {TENANT_ID}")
    print(f"  user_id:   {DAEMON_USER_ID}")
    print(f"  target:    {TARGET_PATH}")
    print()

    # ── Inject the task assigned to Builder ─────────────────────────────
    task_id = tasks_mod.create_task(
        tenant_id=TENANT_ID,
        user_id=DAEMON_USER_ID,
        owner_id="builder",
        brief=(
            f"Create a file at {TARGET_PATH} containing exactly the "
            f"text {TARGET_CONTENT!r}.  Use write_file (which routes "
            f"through the user's daemon).  Then attach_output the file "
            f"path and submit_for_review with target=user_deliverable."
        ),
        deliverable_spec=(
            f"One file at {TARGET_PATH} with literal content "
            f"{TARGET_CONTENT!r}.  No additional text.  "
            f"target=user_deliverable on submit_for_review."
        ),
        assigner_id="director",
        origin="foreground",
    )
    print(f"  [INJECT] created task {task_id}")
    print()

    # ── Watch ledger transitions ────────────────────────────────────────
    tbl = LedgerStore.get().table("agent_tasks")

    last_status = None
    last_owner  = None
    last_comments_n = 0
    start_ts = time.time()
    TIMEOUT_S = 240.0

    while time.time() - start_ts < TIMEOUT_S:
        arr = (
            tbl.search()
            .where(f"id = '{task_id}'")
            .limit(1)
            .to_arrow()
        )
        if arr.num_rows == 0:
            print("  [POLL] task not found (deleted?)")
            return
        row = {col: arr.column(col)[0].as_py() for col in arr.column_names}
        status = row["status"]
        owner  = row["owner_id"]
        outputs = row.get("outputs") or []

        if status != last_status or owner != last_owner:
            print(
                f"  [{time.time() - start_ts:6.1f}s] status={status:12} "
                f"owner={owner:14} outputs={outputs}"
            )
            last_status = status
            last_owner  = owner

        # Print any new comments.
        cmts = comments_mod.list_comments(
            task_id=task_id, tenant_id=TENANT_ID, user_id=DAEMON_USER_ID,
        )
        for c in cmts[last_comments_n:]:
            ts_off = c.get("ts", 0) - start_ts
            print(
                f"  [{ts_off:6.1f}s]   comment author={c['author_id']:14} "
                f"kind={c['kind']:18} body={c['body'][:80]!r}"
            )
        last_comments_n = len(cmts)

        if status in ("done", "cancelled"):
            print()
            print(f"  TERMINAL: status={status}")
            print()
            # Verify file existed at some point (best-effort — daemon
            # may have written to a Windows path the container can't see).
            print(f"  Target file (from container view): {TARGET_PATH}")
            try:
                container_view = TARGET_PATH.replace("C:/", "/mnt/c/")
                if os.path.exists(container_view):
                    print(f"  Container can see file at {container_view}")
                else:
                    print(
                        f"  Container cannot reach {container_view} — "
                        "verify on host directly."
                    )
            except Exception as e:
                print(f"  (file-check error: {e})")
            return

        time.sleep(3.0)

    print()
    print(f"  TIMEOUT after {TIMEOUT_S:.0f}s — task still status={last_status}")


if __name__ == "__main__":
    main()
