"""Unit tests for app.tools.memory_mcp."""

import pytest

from app.middleware import tenant
from app.tools.memory_mcp import (
    recall_tool,
    observe_tool,
    read_constitution_tool,
    _ALL_TOOLS,
    _SDK_AVAILABLE,
)


pytestmark = pytest.mark.skipif(
    not _SDK_AVAILABLE,
    reason="claude_agent_sdk not installed",
)


async def _call(tool, args):
    return await tool.handler(args)


class TestMetadata:
    def test_two_tools_present(self):
        # recall is handled by TCMM directly (not via agent-runtime
        # MCP); memory_mcp exposes only the writers + constitution.
        names = {t.name for t in _ALL_TOOLS}
        assert names == {"observe", "read_constitution"}

    def test_observe_has_text_required(self):
        schema = observe_tool.input_schema
        assert "text" in schema["required"]

    def test_recall_has_query_required(self):
        schema = recall_tool.input_schema
        assert "query" in schema["required"]


class TestFailClosedWithoutContext:
    @pytest.mark.asyncio
    async def test_recall_no_context_errors(self):
        result = await _call(recall_tool, {"query": "test"})
        assert result.get("isError")

    @pytest.mark.asyncio
    async def test_observe_no_context_errors(self):
        result = await _call(observe_tool, {"text": "test"})
        assert result.get("isError")

    @pytest.mark.asyncio
    async def test_read_constitution_no_context_errors(self):
        result = await _call(read_constitution_tool, {})
        assert result.get("isError")


class TestObserveValidation:
    @pytest.mark.asyncio
    async def test_observe_empty_text_errors(self):
        with tenant.set_tenant_context(
            conversation_id="c", user_id="u", tenant_id="t",
            agent_id="researcher",
        ):
            result = await _call(observe_tool, {"text": ""})
            assert result.get("isError")
            assert "empty" in result["content"][0]["text"].lower()

    @pytest.mark.asyncio
    async def test_observe_too_long_errors(self):
        with tenant.set_tenant_context(
            conversation_id="c", user_id="u", tenant_id="t",
            agent_id="researcher",
        ):
            result = await _call(observe_tool, {"text": "x" * 9000})
            assert result.get("isError")
            assert "too long" in result["content"][0]["text"].lower()


class TestReadConstitutionRealFile:
    @pytest.mark.asyncio
    async def test_loads_real_constitution(self, monkeypatch):
        """Verify the tool can load the real CONSTITUTION.md from disk."""
        from pathlib import Path
        from app import config as cfg_mod

        real_path = Path(
            "C:/Users/rudol/Documents/veilguard/agents/CONSTITUTION.md"
        )
        if not real_path.exists():
            pytest.skip(f"CONSTITUTION.md not found at {real_path}")

        monkeypatch.setattr(cfg_mod, "CONSTITUTION_PATH", real_path)

        with tenant.set_tenant_context(
            conversation_id="c", user_id="u", tenant_id="t",
            agent_id="director",
        ):
            result = await _call(read_constitution_tool, {})
            assert not result.get("isError"), result["content"][0]["text"]
            text = result["content"][0]["text"]
            assert "objectives" in text
            assert "constraints" in text
