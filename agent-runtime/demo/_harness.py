"""Shared demo scaffolding.

Sets up a temp Lance dir, loads personas from the real agents/ markdown,
configures the scripted backend, runs a scenario, prints the ledger
state and event trace afterwards.

Usage in a scenario file:

    from demo._harness import (
        setup_demo_env,
        run_scenario,
        print_ledger_state,
        print_event_trace,
    )

    setup_demo_env()  # MUST come before any app.* import

    from app.backends.scripted_backend import set_script, Turn, ToolCall
    from app.personas.loader import load_personas
    from pathlib import Path

    def main():
        # ... define script ...
        set_script({...})
        events = asyncio.run(run_scenario(
            user_message="...",
            foreground_agent="director",
        ))
        print_event_trace(events)
        print_ledger_state(tenant_id="t1", user_id="u1")
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any


# ── Env setup (call BEFORE importing anything from app.*) ───────────────


def setup_demo_env(*, tmp_root: str | None = None) -> str:
    """Set env vars for demo mode.  Returns the temp Lance dir path.

    MUST be called before any `from app.* import ...` because:
      - LEDGER_DB_PATH is read at module load by LedgerStore
      - TCMM_ENABLED gates whether tcmm middleware tries to HTTP
      - BACKEND selects which backend runtime.py uses
    """
    # Enable info logging so demo output shows what middleware + poller
    # are doing.  Without this, the runtime is silent at run time.
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="  [log] %(name)s %(message)s",
    )

    # Windows consoles default to cp1252; demo banners/prints use unicode
    # (→, etc.). Without this, a decorative print can crash the run with
    # UnicodeEncodeError AFTER the test already passed (UAT finding F9,
    # 2026-06-02). Containers are utf-8 so this is a Windows-local guard.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass

    if tmp_root is None:
        tmp_root = tempfile.mkdtemp(prefix="veilguard-demo-")

    lance_dir = os.path.join(tmp_root, "tcmm.db")
    os.makedirs(lance_dir, exist_ok=True)

    os.environ["LEDGER_DB_PATH"] = lance_dir
    os.environ["VEILGUARD_AUDIT_DB_PATH"] = lance_dir
    # Server-side workspace root for the workspace_fs tools AND the
    # mechanical AC executors (output_path_exists resolves a path UNDER
    # this root). Point it at a writable temp dir so demos can produce real
    # artifacts the server-side AC gate checks on review_decision(approved).
    # Default is "/workspace" (absent on dev hosts) → every output_path_exists
    # AC fails → approves auto-downgrade → tasks can never reach done.
    workspace_dir = os.path.join(tmp_root, "workspace")
    os.makedirs(workspace_dir, exist_ok=True)
    os.environ["VEILGUARD_WORKSPACE_ROOT"] = workspace_dir
    os.environ["TCMM_ENABLED"] = "false"
    os.environ["BACKEND"] = "scripted"
    os.environ["ANTHROPIC_API_KEY"] = "demo-key-not-real"
    os.environ["AGENTS_DIR"] = str(
        Path(__file__).resolve().parent.parent.parent / "agents"
    )
    # Disable approval gate to fail-open in demos (no daemon WS).
    os.environ["APPROVAL_FAIL_CLOSED"] = "false"

    # Make `from app.* import ...` resolve when run as `python -m demo.X`
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    return lance_dir


# ── Scenario runner ──────────────────────────────────────────────────────


async def run_scenario(
    *,
    user_message: str,
    foreground_agent: str = "director",
    conversation_id: str = "demo-conv-1",
    user_id: str = "demo-user",
    tenant_id: str = "demo-tenant",
    backend: str | None = None,
) -> list[dict[str, Any]]:
    """Invoke run_agent_query with the named backend; return all events.

    backend defaults to the BACKEND env var (set by setup_demo_env to
    "scripted").  Live demos override via env or pass explicitly.

    Caller must have already installed a script via
    `backends.scripted_backend.set_script(...)` if using scripted backend.
    """
    # Late import so env vars take effect.
    from app.runtime import run_agent_query
    from app.personas.loader import load_personas
    from pathlib import Path as _Path

    registry = load_personas(_Path(os.environ["AGENTS_DIR"]))
    persona = registry.get(foreground_agent)
    if persona is None:
        raise RuntimeError(
            f"persona {foreground_agent!r} not found; loaded: "
            f"{[p.agent_id for p in registry.all()]}"
        )

    backend_name = backend or os.environ.get("BACKEND", "scripted")

    events: list[dict[str, Any]] = []
    async for ev in run_agent_query(
        persona=persona,
        conversation_id=conversation_id,
        user_id=user_id,
        tenant_id=tenant_id,
        messages=[{"role": "user", "content": user_message}],
        registry=registry,
        constitution=None,
        backend_name=backend_name,
    ):
        events.append(ev)
    return events


# ── Inbox poller (run-once helper for demos) ─────────────────────────────


async def drain_inbox(
    *,
    user_id: str,
    tenant_id: str,
    max_passes: int = 10,
    verbose: bool = True,
) -> int:
    """Run the inbox poller manually for a finite number of passes.

    Demos use this to dispatch tasks the Director just created without
    standing up the background asyncio loop.  Returns total task
    dispatches initiated across all passes.
    """
    import asyncio
    from app.personas.loader import load_personas
    from app.workers.inbox_poller import InboxPoller, ELIGIBLE_OWNERS
    from app.ledger.store import LedgerStore
    from pathlib import Path as _Path

    registry = load_personas(_Path(os.environ["AGENTS_DIR"]))
    poller = InboxPoller(registry)

    # Diagnostic: show what the poller WOULD find before claiming.
    if verbose:
        try:
            tbl = LedgerStore.get().table("agent_tasks")
            owners = ", ".join(f"'{o}'" for o in ELIGIBLE_OWNERS)
            scan = tbl.search().where(
                f"owner_id IN ({owners}) AND status = 'open'"
            ).to_arrow()
            print(f"  [poller diag] open tasks for IC owners: {scan.num_rows}")
            for i in range(scan.num_rows):
                print(
                    f"    - {scan.column('id')[i].as_py()} "
                    f"owner={scan.column('owner_id')[i].as_py()} "
                    f"status={scan.column('status')[i].as_py()}"
                )
        except Exception as e:
            print(f"  [poller diag] scan failed: {e}")

    for pass_n in range(max_passes):
        try:
            await poller._poll_once()
        except Exception as e:
            print(f"  [poller diag] _poll_once pass {pass_n} error: {e}")
        await asyncio.sleep(0.1)

    # Wait for all in-flight to finish.
    deadline = asyncio.get_event_loop().time() + 30.0
    while poller._in_flight and asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.1)

    # _claimed_history is the authoritative count (in_flight transitions
    # back to empty after each dispatch completes; we want the cumulative
    # claim count).
    return len(poller._claimed_history)


# ── Output helpers ───────────────────────────────────────────────────────


def print_event_trace(events: list[dict[str, Any]]) -> None:
    """Print the event stream as a human-readable trace."""
    print("\n---- EVENT TRACE ----")
    for i, ev in enumerate(events):
        t = ev.get("type", "?")
        if t == "run_start":
            print(f"  [{i:>3}] {t:20} agent={ev.get('agent_id')} "
                  f"backend={ev.get('backend')} model={ev.get('model')}")
        elif t == "assistant_text":
            text = ev.get("text", "")[:80]
            print(f"  [{i:>3}] {t:20} {text!r}")
        elif t == "tool_call":
            print(f"  [{i:>3}] {t:20} {ev.get('name')} (id={(ev.get('id') or '')[:8]})")
        elif t == "tool_dispatch":
            print(f"  [{i:>3}] {t:20} -> {ev.get('name')}")
        elif t == "tool_result":
            err = " [ERROR]" if ev.get("is_error") else ""
            print(f"  [{i:>3}] {t:20} id={(ev.get('id') or '')[:8]}{err}")
        elif t == "final_result":
            r = ev.get("result", "")
            if isinstance(r, str):
                r = r[:80]
            print(f"  [{i:>3}] {t:20} {r!r}")
        elif t == "usage":
            print(f"  [{i:>3}] {t:20} "
                  f"in={ev.get('tokens_input_total')} "
                  f"out={ev.get('tokens_output')} "
                  f"iter={ev.get('iterations')}")
        elif t == "run_end":
            print(f"  [{i:>3}] {t:20} stop_reason={ev.get('stop_reason')!r}")
        elif t == "error":
            print(f"  [{i:>3}] {t:20} code={ev.get('code')} {ev.get('message')}")
        else:
            print(f"  [{i:>3}] {t:20} {ev}")


def print_ledger_state(*, tenant_id: str, user_id: str) -> None:
    """Dump the agent_tasks + task_comments tables for the demo tenant."""
    from app.ledger.store import LedgerStore, ns_filter
    from app.ledger import comments as comments_mod

    try:
        store = LedgerStore.get()
        tasks_tbl = store.table("agent_tasks")
    except Exception as e:
        print(f"\n[demo] ledger state unavailable: {e}")
        return

    print("\n---- LEDGER: agent_tasks ----")
    try:
        arr = (
            tasks_tbl.search()
            .where(ns_filter(tenant_id, user_id))
            .to_arrow()
        )
        if arr.num_rows == 0:
            print("  (no tasks)")
        else:
            for i in range(arr.num_rows):
                row = {col: arr.column(col)[i].as_py() for col in arr.column_names}
                tid = row["id"]
                print(
                    f"  {tid} status={row['status']:12} owner={row['owner_id']:14} "
                    f"brief={row['brief'][:60]!r}"
                )
                if row.get("outputs"):
                    for o in row["outputs"]:
                        print(f"    output: {o}")
                # Comment chain
                try:
                    cmts = comments_mod.list_comments(
                        task_id=tid, tenant_id=tenant_id, user_id=user_id
                    )
                    for c in cmts:
                        print(
                            f"    [{c.get('ts', 0):.1f}] "
                            f"{c['author_id']:14} {c['kind']:18} "
                            f"{c['body'][:60]!r}"
                        )
                except Exception as e:
                    print(f"    (comments unavailable: {e})")
    except Exception as e:
        print(f"  (scan failed: {e})")


def print_summary_banner(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def patch_latest_task_placeholders(
    *,
    tenant_id: str,
    user_id: str,
) -> int:
    """Walk the current scripted script and replace placeholder strings
    `__LATEST_TASK_FOR_<owner>__` with the latest open task assigned to
    that owner.

    Demos use this between Director's turn (which creates the task) and
    the inbox drain (which dispatches Researcher).  Without the patch
    Researcher's tool args would carry the literal placeholder string.

    Returns the number of substitutions made.
    """
    from app.backends.scripted_backend import _CURRENT_SCRIPT
    from app.ledger.store import LedgerStore, ns_filter

    if _CURRENT_SCRIPT is None:
        return 0

    # Build owner → task_id map by scanning open tasks.
    try:
        tbl = LedgerStore.get().table("agent_tasks")
        arr = tbl.search().where(
            f"{ns_filter(tenant_id, user_id)} AND status = 'open'"
        ).to_arrow()
    except Exception:
        return 0

    owner_to_task: dict[str, str] = {}
    for i in range(arr.num_rows):
        owner = arr.column("owner_id")[i].as_py()
        tid = arr.column("id")[i].as_py()
        owner_to_task.setdefault(owner, tid)  # first wins (most recent at top)

    n_subst = 0
    for persona_id, turns in _CURRENT_SCRIPT.items():
        for turn in turns:
            for tc in turn.tool_calls:
                for key, val in list(tc.input.items()):
                    if not isinstance(val, str):
                        continue
                    if not val.startswith("__LATEST_TASK_FOR_"):
                        continue
                    target_owner = val[len("__LATEST_TASK_FOR_"):].rstrip("_").lower()
                    if target_owner in owner_to_task:
                        tc.input[key] = owner_to_task[target_owner]
                        n_subst += 1
    return n_subst
