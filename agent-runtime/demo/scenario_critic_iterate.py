"""Demo - Critic rejects, Researcher iterates, Critic approves.

Hardest of the four scenarios because it exercises:
  1. Director delegates to Researcher (Pattern B)
  2. Researcher submits with a deliberately incomplete observation
  3. Critic-claim is manually invoked (this demo simulates the review
     trigger; production would have a scheduled critic-review job or
     event-based dispatch on status=review)
  4. Critic returns changes_requested via review_decision tool → task
     transitions review → in_progress
  5. Researcher's SECOND wake-up: fixes the issue, resubmits
  6. Critic-claim approves → task transitions review → done

Asserts:
  - Two review_decision comments in the chain (one rejection, one approval)
  - Final task status == done
  - Comment chain has at least 7 entries (status changes + reviews)

Run:
    python -m demo.scenario_critic_iterate
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

from app.backends.scripted_backend import (  # noqa: E402
    set_script, Turn, ToolCall, _TURN_COUNTERS,
)


def main() -> None:
    print_summary_banner("DEMO - Critic rejects, Researcher iterates, Critic approves")
    print(f"  Lance dir: {TMP_DIR}")

    set_script({
        "director": [
            Turn(
                text="Delegating research to Researcher.",
                tool_calls=[
                    ToolCall(
                        name="mcp__veilguard_ledger__create_task",
                        input={
                            "owner_id": "researcher",
                            "brief": "Document WebAuthn passkey adoption among US consumer banks.",
                            "deliverable_spec": (
                                "agents/team/core/drafts/passkey-research.md, "
                                "with at least one inline citation per claim"
                            ),
                        },
                    ),
                ],
            ),
            Turn(text="Delegated.", stop_reason="end_turn"),
        ],
        # Researcher's first run: submits without proper citation.
        "researcher": [
            Turn(text="Accepting.", tool_calls=[
                ToolCall(name="mcp__veilguard_ledger__accept_task",
                         input={"task_id": "__LATEST_TASK_FOR_RESEARCHER__"}),
            ]),
            Turn(text="Recording finding (intentionally without source).", tool_calls=[
                ToolCall(name="mcp__veilguard_memory__observe", input={
                    "text": "Finding: All US banks support passkeys.",
                    # no source_url - this is the structural issue
                }),
            ]),
            Turn(
                text="First submit. Pausing for Critic review.",
                tool_calls=[
                    ToolCall(name="mcp__veilguard_ledger__attach_output", input={
                        "task_id": "__LATEST_TASK_FOR_RESEARCHER__",
                        "path": "agents/team/core/drafts/passkey-research.md",
                    }),
                    ToolCall(name="mcp__veilguard_ledger__submit_for_review", input={
                        "task_id": "__LATEST_TASK_FOR_RESEARCHER__",
                        "target": "team_knowledge",
                    }),
                ],
                # Pause here so Step 3 (Critic) can run before
                # Researcher's iterate turns fire in Step 4.
                stop_reason="end_turn",
            ),
            # After Critic rejects, Researcher wakes again via Step 4
            # and iterates.
            Turn(text="Got feedback - adding the citation.", tool_calls=[
                ToolCall(name="mcp__veilguard_memory__observe", input={
                    "text": "Verified at https://example.com/passkey-tracker (2026-04 snapshot).",
                    "source_url": "https://example.com/passkey-tracker",
                }),
            ]),
            Turn(text="Resubmitting with citation.", tool_calls=[
                ToolCall(name="mcp__veilguard_ledger__submit_for_review", input={
                    "task_id": "__LATEST_TASK_FOR_RESEARCHER__",
                    "target": "team_knowledge",
                }),
            ]),
            Turn(text="Done.", stop_reason="end_turn"),
        ],
        # critic-claim: first reject, then approve on second invocation.
        "critic-claim": [
            Turn(text="Reviewing first submission.", tool_calls=[
                ToolCall(name="mcp__veilguard_ledger__get_task",
                         input={"task_id": "__LATEST_TASK_FOR_RESEARCHER__"}),
            ]),
            Turn(text="Missing citation - requesting changes.", tool_calls=[
                ToolCall(name="mcp__veilguard_ledger__review_decision", input={
                    "task_id": "__LATEST_TASK_FOR_RESEARCHER__",
                    "decision": "changes_requested",
                    "notes": "Para 1 claim 'all US banks support passkeys' has no source_url",
                }),
            ]),
            Turn(text="First review complete.", stop_reason="end_turn"),
            # Second invocation (after Researcher resubmits)
            Turn(text="Reviewing second submission.", tool_calls=[
                ToolCall(name="mcp__veilguard_ledger__get_task",
                         input={"task_id": "__LATEST_TASK_FOR_RESEARCHER__"}),
            ]),
            Turn(text="Citation present - approving.", tool_calls=[
                ToolCall(name="mcp__veilguard_ledger__review_decision", input={
                    "task_id": "__LATEST_TASK_FOR_RESEARCHER__",
                    "decision": "approved",
                    "notes": "Claim properly attributed to passkey-tracker source.",
                }),
            ]),
            Turn(text="Approved.", stop_reason="end_turn"),
        ],
    })

    user_msg = "Document passkey adoption among US consumer banks."
    print(f"\n  USER: {user_msg}")

    print("\n  Step 1: Director runs...")
    events = asyncio.run(run_scenario(
        user_message=user_msg, foreground_agent="director",
    ))
    print_event_trace(events)

    patch_latest_task_placeholders(tenant_id="demo-tenant", user_id="demo-user")

    print("\n  Step 2: First Researcher pass (submits without citation)...")
    asyncio.run(drain_inbox(
        user_id="demo-user", tenant_id="demo-tenant", max_passes=5, verbose=False,
    ))

    print("\n  Step 3: Critic-claim reviews (manual invoke since auto-trigger is Phase 3)...")
    critic_events = asyncio.run(run_scenario(
        user_message="Review the task currently in status=review.",
        foreground_agent="critic-claim",
        conversation_id="demo-conv-critic-1",
    ))
    # Print only the tool-related events for brevity
    for ev in critic_events:
        if ev.get("type") in ("tool_dispatch", "tool_result", "final_result"):
            t = ev.get("type")
            if t == "tool_dispatch":
                print(f"    [critic] -> {ev.get('name')}")
            elif t == "tool_result":
                err = " [ERROR]" if ev.get("is_error") else ""
                print(f"    [critic] tool_result{err}")

    print("\n  Step 4: Researcher iterates (status flipped back to in_progress)...")
    # IMPORTANT: the inbox poller filters status=='open', not in_progress.
    # When critic flips review → in_progress, the task is owned by
    # researcher but NOT open.  For Phase 2 we rely on the SAME runtime
    # session to continue.  Researcher just gets re-invoked directly
    # since the task is still owned by them.
    # (Phase 3 inbox poller would also handle in_progress on stale leases.)
    # We invoke Researcher again manually; turn counter continues from
    # where it left off (turn 3+).
    researcher_events = asyncio.run(run_scenario(
        user_message="Continue work on your assigned task — fix the citation issue.",
        foreground_agent="researcher",
        conversation_id="demo-conv-researcher-2",
    ))
    for ev in researcher_events:
        if ev.get("type") in ("tool_dispatch", "tool_result"):
            t = ev.get("type")
            if t == "tool_dispatch":
                print(f"    [researcher] -> {ev.get('name')}")

    print("\n  Step 5: Critic-claim re-reviews + approves...")
    critic_events_2 = asyncio.run(run_scenario(
        user_message="Re-review the task.",
        foreground_agent="critic-claim",
        conversation_id="demo-conv-critic-2",
    ))
    for ev in critic_events_2:
        if ev.get("type") in ("tool_dispatch",):
            print(f"    [critic] -> {ev.get('name')}")

    print_ledger_state(tenant_id="demo-tenant", user_id="demo-user")

    from app.ledger.store import LedgerStore, ns_filter
    from app.ledger import comments
    try:
        tbl = LedgerStore.get().table("agent_tasks")
        arr = tbl.search().where(ns_filter("demo-tenant", "demo-user")).to_arrow()
        n = arr.num_rows
        statuses = [arr.column("status")[i].as_py() for i in range(n)]
        task_id = arr.column("id")[0].as_py() if n > 0 else None
        cmts = comments.list_comments(
            task_id=task_id, tenant_id="demo-tenant", user_id="demo-user",
        ) if task_id else []
    except Exception as e:
        n, statuses, cmts = -1, [], []

    n_review_decisions = sum(1 for c in cmts if c.get("kind") == "review_decision")
    n_status_changes = sum(1 for c in cmts if c.get("kind") == "status_change")

    print_summary_banner("RESULT")
    print(f"  Tasks created:           {n} (expected: 1)")
    print(f"  Final status:            {statuses}")
    print(f"  Total comments in chain: {len(cmts)}")
    print(f"  review_decision count:   {n_review_decisions} (expected: 2 - reject + approve)")
    print(f"  status_change count:     {n_status_changes}")

    if (n == 1 and statuses == ["done"]
            and n_review_decisions == 2):
        print("\n  [PASS] Critic iterate flow worked end-to-end.")
    else:
        print(f"\n  [PARTIAL] expected done + 2 review_decisions; got {statuses} + {n_review_decisions}")


if __name__ == "__main__":
    main()
