"""Round-trip tests for the `_veilguard` provenance envelope.

The connector side (this code) and the TCMM adapter side
(`adapters/veilguard_adapter.py` in the canonical TCMM tree) maintain
independent parsers of the same wire shape. This suite locks the shape
from the connector side; the adapter has its own smoke test.
"""
from __future__ import annotations

import json
import pathlib
import sys

_CONNECTORS_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CONNECTORS_DIR))

from _base.envelope import parse_envelope, wrap_with_provenance  # noqa: E402


class TestWrap:
    def test_minimal_wrap_round_trip(self):
        s = wrap_with_provenance("hello", connector="stub")
        content, meta = parse_envelope(s)
        assert content == "hello"
        assert meta == {"connector": "stub"}

    def test_full_wrap_round_trip(self):
        s = wrap_with_provenance(
            "doc body",
            connector="sharepoint",
            tool_ref={
                "connector": "sharepoint",
                "tool": "read",
                "args": {"item_id": "01ABC", "site_id": "X"},
            },
            acl=["group:eng", "group:execs"],
            etag="v3",
            title="Q1 Status",
        )
        content, meta = parse_envelope(s)
        assert content == "doc body"
        assert meta["connector"] == "sharepoint"
        assert meta["tool_ref"]["args"]["item_id"] == "01ABC"
        assert meta["acl"] == ["group:eng", "group:execs"]
        assert meta["etag"] == "v3"
        assert meta["title"] == "Q1 Status"

    def test_omits_none_and_empty_fields(self):
        s = wrap_with_provenance(
            "x",
            connector="stub",
            tool_ref=None,
            acl=None,
            etag=None,
            title=None,
        )
        _, meta = parse_envelope(s)
        # Only connector should remain
        assert meta == {"connector": "stub"}

    def test_omits_empty_acl_list(self):
        s = wrap_with_provenance("x", connector="stub", acl=[])
        _, meta = parse_envelope(s)
        assert "acl" not in meta

    def test_unicode_content(self):
        s = wrap_with_provenance("café résumé 北京", connector="stub")
        content, _ = parse_envelope(s)
        assert content == "café résumé 北京"

    def test_wire_format_is_valid_json(self):
        s = wrap_with_provenance("hi", connector="stub")
        # Independent parse — must be valid JSON regardless of our parser
        parsed = json.loads(s)
        assert "content" in parsed
        assert "_veilguard" in parsed


class TestParse:
    def test_plain_text_passes_through(self):
        content, meta = parse_envelope("just plain text from a non-connector tool")
        assert content == "just plain text from a non-connector tool"
        assert meta == {}

    def test_invalid_json_passes_through(self):
        content, meta = parse_envelope("{not valid json")
        assert content == "{not valid json"
        assert meta == {}

    def test_json_without_veilguard_passes_through(self):
        original = '{"foo": "bar", "baz": 1}'
        content, meta = parse_envelope(original)
        assert content == original  # unchanged
        assert meta == {}

    def test_empty_string(self):
        content, meta = parse_envelope("")
        assert content == ""
        assert meta == {}

    def test_whitespace_string(self):
        content, meta = parse_envelope("   ")
        assert content == "   "
        assert meta == {}

    def test_array_json_passes_through(self):
        # Valid JSON but not the envelope shape (top-level array)
        content, meta = parse_envelope('[1, 2, 3]')
        assert content == '[1, 2, 3]'
        assert meta == {}

    def test_envelope_with_non_dict_metadata_safe(self):
        # Defensive: malformed envelope where _veilguard is a string
        s = '{"content": "x", "_veilguard": "not-a-dict"}'
        content, meta = parse_envelope(s)
        assert content == "x"
        assert meta == {}

    def test_envelope_with_missing_content_field(self):
        s = '{"_veilguard": {"connector": "stub"}}'
        content, meta = parse_envelope(s)
        assert content == ""  # default when content key missing
        assert meta == {"connector": "stub"}
