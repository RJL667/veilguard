"""Unit tests for middleware.cache_control.

The normalizer is our defense against the documented SDK bug where
cache_control gets placed on every block instead of the last one.
These tests pin the contract: exactly ONE marker, on the LAST block.
"""

import pytest

from app.middleware.cache_control import (
    MAX_CACHE_CONTROL_MARKERS,
    normalize_cache_control,
    count_cache_control_markers,
    validate_total_markers,
)


class TestNormalize:
    def test_empty_passthrough(self):
        assert normalize_cache_control([]) == []

    def test_single_block_gets_marker(self):
        blocks = [{"type": "text", "text": "system content"}]
        out = normalize_cache_control(blocks)
        assert len(out) == 1
        assert out[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

    def test_multiple_blocks_marker_on_last_only(self):
        blocks = [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
            {"type": "text", "text": "third"},
        ]
        out = normalize_cache_control(blocks)
        assert "cache_control" not in out[0]
        assert "cache_control" not in out[1]
        assert "cache_control" in out[2]

    def test_strips_auto_placed_markers(self):
        # Simulate the SDK bug: every block has a marker.
        blocks = [
            {"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "b", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "c", "cache_control": {"type": "ephemeral"}},
        ]
        out = normalize_cache_control(blocks)
        assert "cache_control" not in out[0]
        assert "cache_control" not in out[1]
        assert "cache_control" in out[2]

    def test_custom_ttl_applied(self):
        blocks = [{"type": "text", "text": "x"}]
        out = normalize_cache_control(blocks, ttl="5m")
        assert out[0]["cache_control"]["ttl"] == "5m"

    def test_does_not_mutate_input(self):
        blocks = [{"type": "text", "text": "x", "cache_control": {"type": "old"}}]
        original_marker = blocks[0]["cache_control"]
        out = normalize_cache_control(blocks)
        # Caller's list is unchanged
        assert blocks[0]["cache_control"] is original_marker
        # Output is independent
        assert out[0]["cache_control"]["type"] == "ephemeral"


class TestCountMarkers:
    def test_count_in_system(self):
        sys_blocks = [
            {"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "b"},
            {"type": "text", "text": "c", "cache_control": {"type": "ephemeral"}},
        ]
        assert count_cache_control_markers(sys_blocks) == 2

    def test_count_across_sections(self):
        sys_blocks = [{"type": "text", "text": "s", "cache_control": {}}]
        tools = [{"name": "t", "cache_control": {}}]
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "m", "cache_control": {}},
                ],
            },
        ]
        assert count_cache_control_markers(sys_blocks, tools, msgs) == 3


class TestValidate:
    def test_within_budget(self):
        sys_blocks = [
            {"type": "text", "text": "x", "cache_control": {}},
        ]
        # Should not raise.
        validate_total_markers(sys_blocks)

    def test_exceeds_budget_raises(self):
        sys_blocks = [
            {"type": "text", "text": str(i), "cache_control": {}}
            for i in range(MAX_CACHE_CONTROL_MARKERS + 1)
        ]
        with pytest.raises(ValueError) as exc:
            validate_total_markers(sys_blocks)
        assert "marker budget exceeded" in str(exc.value)
