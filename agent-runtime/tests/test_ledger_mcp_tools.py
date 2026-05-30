"""Unit tests for app.tools.ledger_mcp — the MCP tool wrappers.

The Claude Agent SDK's @tool decorator returns SdkMcpTool objects with
.name / .description / .input_schema / .handler attributes.  The handler
is the actual async callable.

We test:
  - Tool metadata is well-formed (name, description, schema)
  - Tools fail-closed when TenantContext isn't set (via .handler())
  - Required fields are declared in input_schema
"""

from __future__ import annotations

import pytest

from app.middleware import tenant
from app.tools.ledger_mcp import (
    create_task_tool,
    accept_task_tool,
    add_comment_tool,
    submit_for_review_tool,
    convert_proposal_tool,
    shelve_proposal_tool,
    inbox_tool,
    get_task_tool,
    _ALL_TOOLS,
    _SDK_AVAILABLE,
)


# Skip the whole module if the SDK isn't installed — these tests assume
# the real decorator (which produces SdkMcpTool) is present.
pytestmark = pytest.mark.skipif(
    not _SDK_AVAILABLE,
    reason="claude_agent_sdk not installed",
)


def _meta(tool):
    """Return (name, description, input_schema) from a SdkMcpTool or shim."""
    return tool.name, tool.description, tool.input_schema


async def _call(tool, args):
    """Invoke the tool's handler (sync or async)."""
    return await tool.handler(args)


class TestToolMetadata:
    def test_each_tool_has_name_and_schema(self):
        for t in _ALL_TOOLS:
            assert t.name, f"{t} missing name"
            assert t.description, f"{t.name} missing description"
            assert t.input_schema, f"{t.name} missing input_schema"

    def test_input_schemas_are_jsonschema_objects(self):
        for t in _ALL_TOOLS:
            schema = t.input_schema
            assert schema["type"] == "object"
            assert "properties" in schema

    def test_create_task_requires_owner_brief_spec(self):
        schema = create_task_tool.input_schema
        required = set(schema.get("required") or [])
        assert {"owner_id", "brief", "deliverable_spec"} <= required

    def test_tool_count(self):
        # Director:        create_task, assign_task, convert_proposal,
        #                  shelve_proposal, defer_proposal, list_proposals,
        #                  list_lessons_for_review                 = 7
        # IC:              accept_task, add_comment, attach_output,
        #                  submit_for_review, get_task, inbox       = 6
        # Critic:          review_decision                          = 1
        # Phase 7.5 teams: create_team, list_teams, team_cost_report = 3
        # Total                                                      = 17
        assert len(_ALL_TOOLS) == 17


class TestFailClosedWithoutContext:
    """Every tool must fail-closed when TenantContext is unset.

    This prevents accidental cross-tenant writes if the runtime forgets
    to establish context before invoking query().
    """

    @pytest.mark.asyncio
    async def test_create_task_no_context_errors(self):
        result = await _call(create_task_tool, {
            "owner_id": "researcher",
            "brief": "research X",
            "deliverable_spec": "report.md, 500 words",
        })
        assert result.get("isError")
        assert "TenantContext" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_accept_task_no_context_errors(self):
        result = await _call(accept_task_tool, {"task_id": "task-xyz"})
        assert result.get("isError")

    @pytest.mark.asyncio
    async def test_add_comment_no_context_errors(self):
        result = await _call(add_comment_tool, {
            "task_id": "task-xyz",
            "kind": "comment",
            "body": "hello",
        })
        assert result.get("isError")

    @pytest.mark.asyncio
    async def test_submit_for_review_no_context_errors(self):
        result = await _call(submit_for_review_tool, {
            "task_id": "task-xyz",
            "target": "team_knowledge",
        })
        assert result.get("isError")

    @pytest.mark.asyncio
    async def test_convert_proposal_no_context_errors(self):
        result = await _call(convert_proposal_tool, {"proposal_id": "prop-xyz"})
        assert result.get("isError")

    @pytest.mark.asyncio
    async def test_shelve_proposal_no_context_errors(self):
        result = await _call(shelve_proposal_tool, {
            "proposal_id": "prop-xyz",
            "reason": "user defer",
        })
        assert result.get("isError")

    @pytest.mark.asyncio
    async def test_inbox_no_context_errors(self):
        result = await _call(inbox_tool, {})
        assert result.get("isError")

    @pytest.mark.asyncio
    async def test_get_task_no_context_errors(self):
        result = await _call(get_task_tool, {"task_id": "task-xyz"})
        assert result.get("isError")
