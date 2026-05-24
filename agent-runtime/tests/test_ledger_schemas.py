"""Verify decision-ledger Lance schemas are well-formed PyArrow shapes.

This doesn't need actual lancedb — just exercises the schema functions
and asserts the shared skeleton appears on every table.
"""

import pyarrow as pa

from app.ledger.schemas import (
    TABLE_SCHEMAS,
    agent_tasks_schema,
    task_comments_schema,
    task_proposals_schema,
    proposal_outcomes_schema,
    org_memory_schema,
    client_tool_approvals_schema,
    client_tool_bypass_schema,
)


_SHARED_SKELETON_COLUMNS = {
    "id", "kind", "status", "parent_id", "lineage_chain",
    "tenant_id", "user_id", "created_by_agent_id",
    "created_ts", "updated_ts", "cost_attributed_usd",
}


class TestSchemaConstruction:
    def test_all_seven_tables_present(self):
        assert set(TABLE_SCHEMAS.keys()) == {
            "agent_tasks",
            "task_comments",
            "task_proposals",
            "proposal_outcomes",
            "org_memory",
            "client_tool_approvals",
            "client_tool_bypass",
        }

    def test_all_schemas_are_pa_schema_instances(self):
        for name, fn in TABLE_SCHEMAS.items():
            schema = fn()
            assert isinstance(schema, pa.Schema), f"{name} not pa.Schema"


class TestSharedSkeleton:
    def test_agent_tasks_has_skeleton(self):
        cols = set(agent_tasks_schema().names)
        missing = _SHARED_SKELETON_COLUMNS - cols
        assert not missing, f"agent_tasks missing shared skeleton: {missing}"

    def test_task_proposals_has_skeleton(self):
        cols = set(task_proposals_schema().names)
        missing = _SHARED_SKELETON_COLUMNS - cols
        assert not missing, f"task_proposals missing shared skeleton: {missing}"

    def test_proposal_outcomes_has_skeleton(self):
        cols = set(proposal_outcomes_schema().names)
        missing = _SHARED_SKELETON_COLUMNS - cols
        assert not missing, f"proposal_outcomes missing shared skeleton: {missing}"

    def test_org_memory_has_skeleton(self):
        cols = set(org_memory_schema().names)
        missing = _SHARED_SKELETON_COLUMNS - cols
        assert not missing, f"org_memory missing shared skeleton: {missing}"


class TestKindSpecificFields:
    def test_agent_tasks_has_owner_and_brief(self):
        cols = set(agent_tasks_schema().names)
        assert "owner_id" in cols
        assert "brief" in cols
        assert "deliverable_spec" in cols
        assert "outputs" in cols
        assert "comments_head_hash" in cols  # append-only chain pointer

    def test_task_comments_has_chain_fields(self):
        cols = set(task_comments_schema().names)
        assert "prev_hash" in cols
        assert "self_hash" in cols
        assert "kind" in cols
        assert "author_id" in cols

    def test_task_proposals_has_scoring(self):
        cols = set(task_proposals_schema().names)
        assert "signal_type" in cols
        assert "impact_score" in cols
        assert "decay_score" in cols
        assert "recurrence_count" in cols
        assert "resulting_task_id" in cols

    def test_org_memory_has_expiry(self):
        cols = set(org_memory_schema().names)
        assert "expires_at" in cols
        assert "review_after" in cols
        assert "confidence_decay_per_week" in cols
        assert "last_reinforced_ts" in cols
        assert "reinforcement_count" in cols

    def test_approvals_has_audit_fields(self):
        cols = set(client_tool_approvals_schema().names)
        assert "decision" in cols
        assert "args_sha256" in cols
        assert "approval_token" in cols  # TOCTOU defense per §3.8.5
        assert "ts" in cols
        assert "ts_local" in cols  # daemon vs server timestamps

    def test_bypass_has_glob_field(self):
        cols = set(client_tool_bypass_schema().names)
        assert "arg_glob" in cols
        assert "expires_at" in cols
        assert "active" in cols


class TestExtrasJsonForwardCompat:
    """Every table should have extras_json for adding fields without
    schema migration (spec §0.2 + memory architecture_tcmm_api pattern).
    """

    def test_each_table_has_extras_json(self):
        for name, fn in TABLE_SCHEMAS.items():
            cols = set(fn().names)
            assert "extras_json" in cols, f"{name} missing extras_json"


class TestRequiredColumns:
    """Spec §3.8.5: tenant_id + user_id are non-nullable everywhere
    (tenant isolation is structural, not optional).
    """

    def test_tenant_user_non_nullable(self):
        for name, fn in TABLE_SCHEMAS.items():
            schema = fn()
            for col in ("tenant_id", "user_id"):
                if col in schema.names:
                    field = schema.field(col)
                    assert not field.nullable, (
                        f"{name}.{col} should be NOT NULL (tenant isolation)"
                    )
