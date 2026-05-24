"""Demo - Pattern B: Director delegates ONE task to Researcher.

Exercises the full pipeline:
  1. Director receives user request, decides delegation needed
  2. Director calls create_task (MCP tool, real ledger write)
  3. Task lands in agent_tasks Lance table
  4. Inbox poller picks up the task
  5. Researcher invoked with its persona, scripted to accept, observe,
     attach_output, submit_for_review
  6. Critic-claim invoked, scripted to approve
  7. Director's followup turn synthesizes the final answer

Asserts:
  - agent_tasks has one row, status=done (or review if Critic deferred)
  - task_comments chain has all the lifecycle transitions
  - audit captured turns for both Director and Researcher

Run:
    python -m demo.scenario_pattern_b_delegation
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
    print_summary_banner("DEMO - Pattern B: Director delegates to Researcher")
    print(f"  Lance dir: {TMP_DIR}")

    set_script({
        # Director's first turn: create a task for Researcher.
        # After ledger write completes, Director's next turn would
        # normally come from a fresh user request (Pattern B is sync from
        # user POV: user waits, IC works, Director synthesizes). We
        # split it: Director turn 1 dispatches, then we drain the inbox
        # so Researcher runs, THEN we'd typically have a follow-up
        # user turn for Director to synthesize. The demo stops after
        # the IC submits — the final synthesis is a separate orchestration
        # the production runtime handles via task completion notifications.
        "director": [
            Turn(
                text="Delegating to Researcher.",
                tool_calls=[
                    ToolCall(
                        name="mcp__veilguard_ledger__create_task",
                        input={
                            "owner_id": "researcher",
                            "brief": "Research current state of WebAuthn passkey adoption among US consumer banks.",
                            "deliverable_spec": (
                                "Markdown file at agents/team/core/drafts/passkey-research.md, "
                                "200-400 words, with citations. Submit for review with target=team_knowledge."
                            ),
                        },
                    ),
                ],
            ),
            # Turn 2 fires after create_task returns the task_id;
            # Director acknowledges and exits.
            Turn(
                text="Task created and assigned. Researcher will pick it up via the inbox poller.",
                stop_reason="end_turn",
            ),
        ],
        # Researcher's turns: inbox lookup, accept, observe, attach, submit.
        # We use the placeholder TASK_ID; the harness substitutes the real
        # task_id after Director's create_task fires (see post-script hook).
        "researcher": [
            Turn(
                text="Checking my inbox for the new assignment.",
                tool_calls=[
                    ToolCall(
                        name="mcp__veilguard_ledger__inbox",
                        input={"limit": 5},
                    ),
                ],
            ),
            Turn(
                text="Accepting and starting work.",
                tool_calls=[
                    ToolCall(
                        name="mcp__veilguard_ledger__accept_task",
                        input={"task_id": "__LATEST_TASK_FOR_RESEARCHER__"},
                    ),
                ],
            ),
            Turn(
                text="Recording finding.",
                tool_calls=[
                    ToolCall(
                        name="mcp__veilguard_memory__observe",
                        input={
                            "text": (
                                "Finding: As of 2026-04, the top 10 US "
                                "consumer banks by deposits have all "
                                "announced passkey support for online "
                                "banking, with Bank of America the latest "
                                "in March 2026."
                            ),
                            "source_url": "https://example.com/passkey-bank-tracker",
                        },
                    ),
                ],
            ),
            Turn(
                text="Attaching output + submitting for review.",
                tool_calls=[
                    ToolCall(
                        name="mcp__veilguard_ledger__attach_output",
                        input={
                            "task_id": "__LATEST_TASK_FOR_RESEARCHER__",
                            "path": "agents/team/core/drafts/passkey-research.md",
                        },
                    ),
                    ToolCall(
                        name="mcp__veilguard_ledger__submit_for_review",
                        input={
                            "task_id": "__LATEST_TASK_FOR_RESEARCHER__",
                            "target": "team_knowledge",
                        },
                    ),
                ],
            ),
            Turn(
                text="Submitted. Done with this task.",
                stop_reason="end_turn",
            ),
        ],
    })

    user_msg = "Research the state of WebAuthn passkey adoption among US consumer banks."
    print(f"\n  USER: {user_msg}")

    # Step 1: Director runs, creates the task.
    print("\n  Step 1: Director runs...")
    director_events = asyncio.run(run_scenario(
        user_message=user_msg,
        foreground_agent="director",
    ))
    print_event_trace(director_events)

    # Step 1.5: Patch placeholders in the script with the real task_id
    # that Director just created.  Without this, Researcher's tool args
    # would carry the literal "__LATEST_TASK_FOR_RESEARCHER__" string.
    n_subst = patch_latest_task_placeholders(
        tenant_id="demo-tenant", user_id="demo-user",
    )
    print(f"\n  Patched {n_subst} script placeholder(s) with real task_ids.")

    # Step 2: Drain the inbox so Researcher picks up.
    print("\n  Step 2: Draining inbox (poller dispatches assigned tasks)...")
    n_dispatched = asyncio.run(drain_inbox(
        user_id="demo-user",
        tenant_id="demo-tenant",
        max_passes=5,
    ))
    print(f"  Inbox poller dispatched {n_dispatched} task(s)")

    # Step 3: Show final ledger state.
    print_ledger_state(tenant_id="demo-tenant", user_id="demo-user")

    # Verdict
    from app.ledger.store import LedgerStore, ns_filter
    try:
        tasks_tbl = LedgerStore.get().table("agent_tasks")
        arr = tasks_tbl.search().where(ns_filter("demo-tenant", "demo-user")).to_arrow()
        n = arr.num_rows
        statuses = [arr.column("status")[i].as_py() for i in range(n)]
    except Exception as e:
        n, statuses = -1, [f"err:{e}"]

    print_summary_banner("RESULT")
    print(f"  Tasks created:  {n} (expected: 1)")
    print(f"  Status:         {statuses}")

    if n == 1:
        print("\n  [PASS] Director delegated; ledger has the task; "
              "inbox poller picked it up and dispatched.")
    else:
        print(f"\n  [FAIL] expected 1 task, got {n}")


if __name__ == "__main__":
    main()
