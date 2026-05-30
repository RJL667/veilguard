"""Phase 6 wire-up tests — the bits the unit tests in test_phase_6.py
gate the *helpers*; these gate that the helpers are actually plugged
into the production code paths.

  APR        — broadcaster + token recorder + circuit-breaker hook
  Heartbeats — schema present + writer + sweep wired
  Revisions  — review_decision sets extras_json.is_revision=True
"""

from __future__ import annotations

import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# ── APR wire-up ─────────────────────────────────────────────────────────


def test_apr_record_artifact_imported_by_tasks_create_task():
    src = (REPO_ROOT / "app" / "ledger" / "tasks.py").read_text(encoding="utf-8")
    assert "apr_record_artifact" in src, (
        "Phase 6.7 — `apr_record_artifact` must be called from "
        "ledger.tasks.create_task so task creation counts toward APR"
    )


def test_apr_record_artifact_called_by_update_status():
    """update_status should also bump APR (state transition = artifact)."""
    src = (REPO_ROOT / "app" / "ledger" / "tasks.py").read_text(encoding="utf-8")
    # Count occurrences — both create_task and update_status should call it.
    assert src.count("apr_record_artifact") >= 2


def test_apr_record_artifact_called_by_proposals_create():
    src = (REPO_ROOT / "app" / "ledger" / "proposals.py").read_text(encoding="utf-8")
    assert "apr_record_artifact" in src


def test_apr_record_artifact_called_by_state_machine_comments():
    src = (REPO_ROOT / "app" / "ledger" / "comments.py").read_text(encoding="utf-8")
    assert "apr_record_artifact" in src
    # Should fire on state-machine kinds (status_change, review_decision...)
    assert "status_change" in src and "review_decision" in src


def test_apr_record_tokens_called_by_runtime():
    src = (REPO_ROOT / "app" / "runtime.py").read_text(encoding="utf-8")
    assert "apr_record_tokens" in src, (
        "Phase 6.7 — `apr_record_tokens` must be called from runtime.py "
        "after each LLM round-trip so APR has a tokens denominator"
    )


def test_apr_should_pause_dispatch_called_by_inbox_poller():
    src = (REPO_ROOT / "app" / "workers" / "inbox_poller.py").read_text(encoding="utf-8")
    assert "apr_should_pause_dispatch" in src, (
        "Phase 6.7 — inbox-poller must check the APR circuit breaker "
        "before claiming new work, otherwise the breaker is decorative"
    )


# ── Heartbeats wire-up ──────────────────────────────────────────────────


def test_agent_task_heartbeats_in_table_schemas():
    from app.ledger.schemas import TABLE_SCHEMAS
    assert "agent_task_heartbeats" in TABLE_SCHEMAS


def test_agent_task_heartbeats_schema_shape():
    from app.ledger.schemas import TABLE_SCHEMAS
    schema = TABLE_SCHEMAS["agent_task_heartbeats"]()
    field_names = {f.name for f in schema}
    assert {"task_id", "worker_id", "last_beat_at", "lease_ttl_s"}.issubset(field_names)


def test_record_heartbeat_is_exposed_via_memory_writers():
    from app.memory import record_heartbeat
    assert callable(record_heartbeat)


def test_heartbeat_emitted_during_dispatch_loop():
    src = (REPO_ROOT / "app" / "workers" / "inbox_poller.py").read_text(encoding="utf-8")
    assert "record_heartbeat" in src
    # Must be inside the run_agent_query async-for loop (heuristic):
    assert "record_heartbeat" in src.split("async for _ev in run_agent_query")[1]


def test_sweep_stale_heartbeats_wired_into_run_loop():
    src = (REPO_ROOT / "app" / "workers" / "inbox_poller.py").read_text(encoding="utf-8")
    assert "_sweep_stale_heartbeats" in src
    # Called from the main run() loop, not just defined.
    assert src.count("_sweep_stale_heartbeats") >= 2


# ── Revision flag wire-up ───────────────────────────────────────────────


def test_review_decision_sets_is_revision_on_changes_requested():
    src = (REPO_ROOT / "app" / "tools" / "ledger_mcp.py").read_text(encoding="utf-8")
    # Find the changes_requested branch
    assert 'decision == "changes_requested"' in src
    # is_revision flag must be set in that branch — check the same file
    # has both strings co-located (heuristic; structural grep below).
    assert "is_revision" in src, (
        "Phase 6.4 — review_decision(changes_requested) must set "
        "extras_json.is_revision=True so the revision-priority lane fires"
    )
    # Tighter check: the is_revision string appears AFTER the changes_requested branch
    idx_changes = src.index('decision == "changes_requested"')
    idx_isrev = src.index("is_revision")
    assert idx_isrev > idx_changes, (
        "is_revision should be set inside the changes_requested branch"
    )
