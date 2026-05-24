"""Demo — Pattern A: Director answers solo (no delegation, no tools).

User asks a simple question; Director's persona handles it from recall
alone.  No subtasks created.  Single audit row.

Run:
    python -m demo.scenario_pattern_a_solo
"""

import asyncio
from demo._harness import (
    setup_demo_env,
    run_scenario,
    print_event_trace,
    print_ledger_state,
    print_summary_banner,
)

# MUST come before any app.* import.
TMP_DIR = setup_demo_env()

from app.backends.scripted_backend import set_script, Turn, ToolCall  # noqa: E402


def main() -> None:
    print_summary_banner("DEMO - Pattern A: Solo Director")
    print(f"  Lance dir: {TMP_DIR}")

    # Script: Director answers in a single turn, no tools.
    set_script({
        "director": [
            Turn(
                text=(
                    "Based on recall, the user is in South Africa Standard "
                    "Time (UTC+2). No delegation needed."
                ),
                stop_reason="end_turn",
            ),
        ],
    })

    user_msg = "What time zone is the user in?"
    print(f"\n  USER: {user_msg}")

    events = asyncio.run(run_scenario(
        user_message=user_msg,
        foreground_agent="director",
    ))

    print_event_trace(events)
    print_ledger_state(tenant_id="demo-tenant", user_id="demo-user")

    # Assertions baked into the demo: this scenario should NOT have
    # created any tasks (pure Pattern A).
    from app.ledger.store import LedgerStore, ns_filter
    try:
        tasks_tbl = LedgerStore.get().table("agent_tasks")
        arr = tasks_tbl.search().where(ns_filter("demo-tenant", "demo-user")).to_arrow()
        n = arr.num_rows
    except Exception:
        n = -1

    print_summary_banner("RESULT")
    print(f"  Tasks created: {n} (expected: 0 — Pattern A is solo, no delegation)")
    print(f"  Director text: {events[-3].get('result', '?') if len(events) >= 3 else '?'}"
          [:100])

    if n == 0:
        print("\n  [PASS] solo Director path works; no spurious tasks created.")
    else:
        print(f"\n  [FAIL] expected 0 tasks, got {n}")


if __name__ == "__main__":
    main()
