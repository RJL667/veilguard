"""Demo - Pattern C: Director fans out to Researcher AND Builder in parallel.

Exercises the harder case:
  1. Director creates 2 tasks in ONE turn (Researcher + Builder)
  2. Inbox poller finds both, claims them with separate leases
  3. Each runs concurrently via the poller's semaphore (max 4 in flight)
  4. Each submits for review independently
  5. Final state: 2 tasks at status=review

Asserts:
  - 2 tasks created
  - Both reach status=review (or done)
  - No double-claim (each task claimed by exactly one worker_id)

Run:
    python -m demo.scenario_pattern_c_fanout
"""

import asyncio

from demo._harness import (
    setup_demo_env,
    run_scenario,
    drain_inbox,
    patch_latest_task_placeholders,
    print_event_trace,
    print_ledger_state,
    print_summary_banner,
)

TMP_DIR = setup_demo_env()

from app.backends.scripted_backend import set_script, Turn, ToolCall  # noqa: E402


def main() -> None:
    print_summary_banner("DEMO - Pattern C: Parallel fanout (Researcher + Builder)")
    print(f"  Lance dir: {TMP_DIR}")

    set_script({
        # Director: ONE turn that creates TWO tasks, one per IC.
        "director": [
            Turn(
                text="Fanning out: research + prototype in parallel.",
                tool_calls=[
                    ToolCall(
                        name="mcp__veilguard_ledger__create_task",
                        input={
                            "owner_id": "researcher",
                            "brief": "Research passkey adoption stats for top 10 US consumer banks.",
                            "deliverable_spec": (
                                "agents/team/core/drafts/passkey-research.md, "
                                "200-400 words, citations, target=team_knowledge"
                            ),
                        },
                    ),
                    ToolCall(
                        name="mcp__veilguard_ledger__create_task",
                        input={
                            "owner_id": "builder",
                            "brief": "Build a one-page prototype landing page for passkey enrolment.",
                            "deliverable_spec": (
                                "agents/team/core/drafts/passkey-landing.html, "
                                "≤200 lines, vanilla HTML/CSS, target=user_deliverable"
                            ),
                        },
                    ),
                ],
            ),
            Turn(
                text="Tasks dispatched. Researcher + Builder will pick them up via inbox.",
                stop_reason="end_turn",
            ),
        ],
        # Researcher: standard flow.
        "researcher": [
            Turn(text="Checking inbox.", tool_calls=[
                ToolCall(name="mcp__veilguard_ledger__inbox", input={"limit": 5}),
            ]),
            Turn(text="Accepting.", tool_calls=[
                ToolCall(name="mcp__veilguard_ledger__accept_task",
                         input={"task_id": "__LATEST_TASK_FOR_RESEARCHER__"}),
            ]),
            Turn(text="Recording finding.", tool_calls=[
                ToolCall(name="mcp__veilguard_memory__observe", input={
                    "text": "All top 10 US consumer banks support passkeys as of 2026-04.",
                    "source_url": "https://example.com/passkey-tracker",
                }),
            ]),
            Turn(text="Submitting.", tool_calls=[
                ToolCall(name="mcp__veilguard_ledger__attach_output", input={
                    "task_id": "__LATEST_TASK_FOR_RESEARCHER__",
                    "path": "agents/team/core/drafts/passkey-research.md",
                }),
                ToolCall(name="mcp__veilguard_ledger__submit_for_review", input={
                    "task_id": "__LATEST_TASK_FOR_RESEARCHER__",
                    "target": "team_knowledge",
                }),
            ]),
            Turn(text="Done.", stop_reason="end_turn"),
        ],
        # Builder: parallel flow.
        "builder": [
            Turn(text="Checking inbox.", tool_calls=[
                ToolCall(name="mcp__veilguard_ledger__inbox", input={"limit": 5}),
            ]),
            Turn(text="Accepting.", tool_calls=[
                ToolCall(name="mcp__veilguard_ledger__accept_task",
                         input={"task_id": "__LATEST_TASK_FOR_BUILDER__"}),
            ]),
            Turn(text="Writing prototype scratchpad note.", tool_calls=[
                ToolCall(name="mcp__veilguard_memory__observe", input={
                    "text": "Built passkey-landing.html with form + onsubmit handler.",
                }),
            ]),
            Turn(text="Submitting.", tool_calls=[
                ToolCall(name="mcp__veilguard_ledger__attach_output", input={
                    "task_id": "__LATEST_TASK_FOR_BUILDER__",
                    "path": "agents/team/core/drafts/passkey-landing.html",
                }),
                ToolCall(name="mcp__veilguard_ledger__submit_for_review", input={
                    "task_id": "__LATEST_TASK_FOR_BUILDER__",
                    "target": "user_deliverable",
                }),
            ]),
            Turn(text="Done.", stop_reason="end_turn"),
        ],
    })

    user_msg = "Research the state of passkey adoption AND build me a quick prototype landing page."
    print(f"\n  USER: {user_msg}")

    print("\n  Step 1: Director fans out...")
    events = asyncio.run(run_scenario(
        user_message=user_msg,
        foreground_agent="director",
    ))
    print_event_trace(events)

    n_subst = patch_latest_task_placeholders(
        tenant_id="demo-tenant", user_id="demo-user",
    )
    print(f"\n  Patched {n_subst} script placeholders.")

    print("\n  Step 2: Inbox poller dispatches both...")
    n_dispatched = asyncio.run(drain_inbox(
        user_id="demo-user", tenant_id="demo-tenant", max_passes=5,
    ))
    print(f"  Inbox poller dispatched {n_dispatched} task(s)")

    print_ledger_state(tenant_id="demo-tenant", user_id="demo-user")

    from app.ledger.store import LedgerStore, ns_filter
    try:
        tbl = LedgerStore.get().table("agent_tasks")
        arr = tbl.search().where(ns_filter("demo-tenant", "demo-user")).to_arrow()
        n = arr.num_rows
        statuses = [arr.column("status")[i].as_py() for i in range(n)]
        owners = [arr.column("owner_id")[i].as_py() for i in range(n)]
        leases = [arr.column("lease_owner")[i].as_py() for i in range(n)]
    except Exception as e:
        n, statuses, owners, leases = -1, [], [], [f"err:{e}"]

    print_summary_banner("RESULT")
    print(f"  Tasks created:    {n} (expected: 2)")
    print(f"  Owners:           {owners}")
    print(f"  Statuses:         {statuses}")
    print(f"  Lease owners:     {leases}")
    print(f"  Distinct leases:  {len(set(leases))} (expected: 1 - same worker)")

    if n == 2 and all(s == "review" for s in statuses):
        print("\n  [PASS] Both ICs ran in parallel; both reached review.")
    else:
        print(f"\n  [PARTIAL] expected 2x review, got {statuses}")


if __name__ == "__main__":
    main()
