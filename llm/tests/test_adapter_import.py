"""Tests for veilguard.llm.adapter — lazy TCMM import + clean interface."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ── Adapter wrapper around a FAKE TCMM module ───────────────────────────


@pytest.fixture
def fake_tcmm_adapter(monkeypatch, tmp_path):
    """Stub out the TCMM adapter import so tests don't need TCMM on disk.

    Creates a fake `adapters.anthropic_adapter` module in sys.path under
    a temp dir, and points TCMM_ROOT at it.  Lets us test the wrapper's
    glue without needing a real TCMM install.
    """
    fake_root = tmp_path / "fake_tcmm"
    fake_root.mkdir()
    (fake_root / "adapters").mkdir()
    (fake_root / "adapters" / "__init__.py").write_text("")
    (fake_root / "adapters" / "anthropic_adapter.py").write_text(
        """
class AnthropicGenerationAdapter:
    def __init__(self, *, api_key='', model_name='', system_prompt=None,
                 system_blocks=None, tools=None):
        self.api_key = api_key
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.system_blocks = system_blocks
        self.tools = tools
        self.last_usage = None
        self.last_response_blocks = None
        self.last_stop_reason = None
        self._mode = 'Fake'

    def generate(self, prompt, label=None):
        self.last_usage = {
            'input_tokens': 100,
            'output_tokens': 50,
            'cache_creation_input_tokens': 0,
            'cache_read_input_tokens': 0,
        }
        self.last_response_blocks = [
            {'type': 'text', 'text': f'echo: {prompt}'},
        ]
        self.last_stop_reason = 'end_turn'
        return f'echo: {prompt}'
"""
    )

    monkeypatch.setenv("TCMM_ROOT", str(fake_root))

    # Reset the lazy-import cache so the next import picks up our fake.
    import llm.adapter as adapter_mod
    adapter_mod._ADAPTER_CLS = None

    yield fake_root

    # Cleanup: remove fake_root from sys.path
    fake_root_str = str(fake_root)
    if fake_root_str in sys.path:
        sys.path.remove(fake_root_str)
    adapter_mod._ADAPTER_CLS = None
    # And unimport so other tests don't see the fake
    for k in [k for k in list(sys.modules) if k.startswith("adapters")]:
        del sys.modules[k]


def test_adapter_imports_via_tcmm_root(fake_tcmm_adapter):
    """The wrapper resolves TCMM_ROOT and imports the inner class."""
    from llm.adapter import AnthropicAdapter

    adapter = AnthropicAdapter(model="claude-sonnet-4-5")
    assert adapter._inner.model_name == "claude-sonnet-4-5"


def test_adapter_generate_returns_frozen_result(fake_tcmm_adapter):
    """generate() returns an AdapterResult, not mutated instance attrs."""
    from llm.adapter import AnthropicAdapter, AdapterResult

    async def run():
        adapter = AnthropicAdapter(
            model="claude-sonnet-4-5",
            system_blocks=[{"type": "text", "text": "sys"}],
            tools=[{"name": "t1", "description": "x", "input_schema": {}}],
        )
        return await adapter.generate("hello world")

    result = asyncio.run(run())
    assert isinstance(result, AdapterResult)
    assert result.text == "echo: hello world"
    assert result.content_blocks == [{"type": "text", "text": "echo: hello world"}]
    assert result.usage == {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    assert result.stop_reason == "end_turn"
    assert result.model == "claude-sonnet-4-5"


def test_adapter_passes_system_blocks(fake_tcmm_adapter):
    """system_blocks make it through to the inner adapter."""
    from llm.adapter import AnthropicAdapter

    blocks = [
        {"type": "text", "text": "preamble", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "memory"},
    ]
    adapter = AnthropicAdapter(model="m", system_blocks=blocks)
    assert adapter._inner.system_blocks == blocks


def test_adapter_passes_tools(fake_tcmm_adapter):
    from llm.adapter import AnthropicAdapter

    tools = [
        {"name": "create_task", "description": "...", "input_schema": {"type": "object"}},
    ]
    adapter = AnthropicAdapter(model="m", tools=tools)
    assert adapter._inner.tools == tools


def test_adapter_missing_tcmm_root_raises(monkeypatch):
    """Clean error when TCMM_ROOT points nowhere."""
    monkeypatch.setenv("TCMM_ROOT", "/this/path/definitely/does/not/exist")
    import llm.adapter as adapter_mod
    adapter_mod._ADAPTER_CLS = None
    from llm.adapter import AnthropicAdapter

    with pytest.raises(RuntimeError, match="TCMM_ROOT="):
        AnthropicAdapter(model="m")


def test_result_dataclass_is_frozen(fake_tcmm_adapter):
    """AdapterResult is immutable — caller can't accidentally mutate it."""
    from llm.adapter import AdapterResult

    r = AdapterResult(text="x", content_blocks=[], usage={}, stop_reason="end_turn")
    with pytest.raises((AttributeError, Exception)):
        r.text = "y"  # type: ignore
