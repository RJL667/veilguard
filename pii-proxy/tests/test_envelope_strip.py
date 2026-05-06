"""Tests for `_veilguard` envelope stripping in the PII proxy.

The proxy mirrors the connector framework's envelope shape (see
`mcp-tools/connectors/_base/envelope.py`). It strips envelopes from
tool_result content on the LLM-egress leg so the LLM sees only the
inner `content` field — the `_veilguard` metadata is internal
infrastructure consumed by TCMM ingest.

Run with::

    cd pii-proxy && python -m pytest tests/test_envelope_strip.py -v
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

# Make the proxy app importable.
_APP_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_APP_DIR))

from app.main import (  # noqa: E402
    _strip_envelope_from_str,
    _strip_veilguard_envelopes_from_messages,
)


# ─── _strip_envelope_from_str ─────────────────────────────────────────


class TestStripEnvelopeFromStr:
    def test_plain_text_returns_none(self):
        assert _strip_envelope_from_str("just plain text") is None

    def test_empty_string_returns_none(self):
        assert _strip_envelope_from_str("") is None

    def test_whitespace_returns_none(self):
        assert _strip_envelope_from_str("   ") is None

    def test_invalid_json_returns_none(self):
        assert _strip_envelope_from_str("{not valid json") is None

    def test_json_without_veilguard_returns_none(self):
        assert _strip_envelope_from_str('{"foo": "bar"}') is None

    def test_array_json_returns_none(self):
        assert _strip_envelope_from_str("[1, 2, 3]") is None

    def test_envelope_returns_inner_content(self):
        env = json.dumps({
            "content": "doc body here",
            "_veilguard": {"connector": "sharepoint"},
        })
        assert _strip_envelope_from_str(env) == "doc body here"

    def test_envelope_with_full_metadata(self):
        env = json.dumps({
            "content": "incident response runbook",
            "_veilguard": {
                "connector": "sharepoint",
                "tool_ref": {"connector": "sharepoint", "tool": "read",
                             "args": {"item_id": "X"}},
                "acl": ["group:security"],
                "etag": "v7",
                "title": "IR Runbook v3",
            },
        })
        assert _strip_envelope_from_str(env) == "incident response runbook"

    def test_envelope_missing_content_returns_empty_string(self):
        env = json.dumps({"_veilguard": {"connector": "stub"}})
        assert _strip_envelope_from_str(env) == ""

    def test_non_string_input_returns_none(self):
        assert _strip_envelope_from_str(None) is None  # type: ignore[arg-type]
        assert _strip_envelope_from_str(123) is None  # type: ignore[arg-type]


# ─── _strip_veilguard_envelopes_from_messages ─────────────────────────


def _make_envelope(content: str, **meta) -> str:
    return json.dumps({"content": content, "_veilguard": dict(meta)})


class TestStripFromMessages:
    def test_empty_messages_list(self):
        assert _strip_veilguard_envelopes_from_messages([]) == 0

    def test_non_list_input_safe(self):
        assert _strip_veilguard_envelopes_from_messages(None) == 0  # type: ignore[arg-type]
        assert _strip_veilguard_envelopes_from_messages("not a list") == 0  # type: ignore[arg-type]

    def test_user_text_message_unaffected(self):
        msgs = [{"role": "user", "content": "hello"}]
        assert _strip_veilguard_envelopes_from_messages(msgs) == 0
        assert msgs == [{"role": "user", "content": "hello"}]

    def test_assistant_message_with_tool_use_unaffected(self):
        msgs = [{
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "x"}],
        }]
        assert _strip_veilguard_envelopes_from_messages(msgs) == 0

    def test_tool_result_string_content_with_envelope(self):
        env = _make_envelope("doc body", connector="sharepoint", etag="v3")
        msgs = [{
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "t1",
                "content": env,
            }],
        }]
        n = _strip_veilguard_envelopes_from_messages(msgs)
        assert n == 1
        assert msgs[0]["content"][0]["content"] == "doc body"

    def test_tool_result_string_content_plain_unchanged(self):
        msgs = [{
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "t1",
                "content": "plain tool output",
            }],
        }]
        n = _strip_veilguard_envelopes_from_messages(msgs)
        assert n == 0
        assert msgs[0]["content"][0]["content"] == "plain tool output"

    def test_tool_result_list_content_with_text_envelope(self):
        env = _make_envelope("nested doc body", connector="slack")
        msgs = [{
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "t1",
                "content": [
                    {"type": "text", "text": env},
                    {"type": "image", "source": "..."},
                ],
            }],
        }]
        n = _strip_veilguard_envelopes_from_messages(msgs)
        assert n == 1
        assert msgs[0]["content"][0]["content"][0]["text"] == "nested doc body"
        # image sub-block untouched
        assert msgs[0]["content"][0]["content"][1] == {"type": "image", "source": "..."}

    def test_multiple_tool_results_in_one_message(self):
        env1 = _make_envelope("body 1", connector="sharepoint")
        env2 = _make_envelope("body 2", connector="slack")
        msgs = [{
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": env1},
                {"type": "tool_result", "tool_use_id": "t2", "content": env2},
            ],
        }]
        n = _strip_veilguard_envelopes_from_messages(msgs)
        assert n == 2
        assert msgs[0]["content"][0]["content"] == "body 1"
        assert msgs[0]["content"][1]["content"] == "body 2"

    def test_envelopes_across_multi_turn_history(self):
        # Several past turns, each containing tool_result envelopes
        env_a = _make_envelope("past doc", connector="sharepoint")
        env_b = _make_envelope("recent doc", connector="sharepoint")
        msgs = [
            {"role": "user", "content": "first user msg"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t_a", "name": "sp"}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t_a", "content": env_a}]},
            {"role": "assistant", "content": "reply about past doc"},
            {"role": "user", "content": "another question"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t_b", "name": "sp"}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t_b", "content": env_b}]},
        ]
        n = _strip_veilguard_envelopes_from_messages(msgs)
        assert n == 2
        # Both tool_results stripped to inner content
        assert msgs[2]["content"][0]["content"] == "past doc"
        assert msgs[6]["content"][0]["content"] == "recent doc"

    def test_malformed_message_safe(self):
        msgs = [
            "not a dict",
            None,
            {"role": "user"},  # no content
            {"role": "user", "content": None},
            {"role": "user", "content": "string content"},
            {"role": "user", "content": [None, "not a dict"]},
        ]
        # Should not raise
        n = _strip_veilguard_envelopes_from_messages(msgs)
        assert n == 0

    def test_does_not_mutate_when_no_envelope(self):
        msgs = [{
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "t1",
                "content": '{"foo": "bar", "no_veilguard_key": true}',
            }],
        }]
        original = json.dumps(msgs)
        n = _strip_veilguard_envelopes_from_messages(msgs)
        assert n == 0
        assert json.dumps(msgs) == original
