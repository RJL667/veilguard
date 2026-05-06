"""Tests for shadow-block rendering.

Locks the wire format the LLM is taught to consume by
:data:`SHADOW_BLOCK_SYSTEM_PROMPT`. Any change to the rendered shape
must update both the renderer and the system-prompt fragment, and
must keep these tests green.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

_CONNECTORS_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CONNECTORS_DIR))

from _base.shadow import (  # noqa: E402
    SHADOW_BLOCK_SYSTEM_PROMPT,
    _render_one_shadow,
    _render_tool_call,
    render_shadow_blocks,
)


# ─── _render_tool_call ────────────────────────────────────────────────


class TestRenderToolCall:
    def test_basic(self):
        assert _render_tool_call({
            "connector": "sharepoint",
            "tool": "read",
            "args": {"item_id": "X"},
        }) == "sharepoint.read(item_id='X')"

    def test_multiple_args(self):
        out = _render_tool_call({
            "connector": "sharepoint",
            "tool": "read_file",
            "args": {"item_id": "01ABC", "site_id": "S"},
        })
        assert "item_id='01ABC'" in out
        assert "site_id='S'" in out

    def test_no_args(self):
        assert _render_tool_call({
            "connector": "slack",
            "tool": "list_channels",
            "args": {},
        }) == "slack.list_channels()"

    def test_missing_args_dict(self):
        assert _render_tool_call({
            "connector": "x",
            "tool": "y",
        }) == "x.y()"

    def test_non_dict_args_safe(self):
        assert _render_tool_call({
            "connector": "x",
            "tool": "y",
            "args": "not a dict",
        }) == "x.y()"

    def test_unknown_connector_or_tool(self):
        assert _render_tool_call({}) == "?.?()"


# ─── _render_one_shadow ───────────────────────────────────────────────


def _hit(**kw):
    base = {
        "text": "doc body content",
        "source": "sharepoint",
        "score": 0.81,
        "title": "Q1 Status",
        "tool_ref": {
            "connector": "sharepoint",
            "tool": "read",
            "args": {"item_id": "X"},
        },
    }
    base.update(kw)
    return base


class TestRenderOneShadow:
    def test_full_block(self):
        out = _render_one_shadow(_hit(), max_chars=1000)
        assert out.startswith("<shadow ")
        assert 'source="sharepoint"' in out
        assert 'score="0.81"' in out
        assert 'title="Q1 Status"' in out
        assert "doc body content" in out
        assert "→ sharepoint.read(item_id='X')" in out
        assert out.endswith("</shadow>")

    def test_skips_when_no_tool_ref(self):
        h = _hit()
        h["tool_ref"] = None
        assert _render_one_shadow(h, max_chars=1000) == ""

    def test_score_omitted_when_none(self):
        out = _render_one_shadow(_hit(score=None), max_chars=1000)
        assert "score=" not in out

    def test_title_omitted_when_empty_or_none(self):
        out_none = _render_one_shadow(_hit(title=None), max_chars=1000)
        out_empty = _render_one_shadow(_hit(title=""), max_chars=1000)
        assert "title=" not in out_none
        assert "title=" not in out_empty

    def test_truncates_long_content(self):
        long_text = "a" * 1000
        out = _render_one_shadow(_hit(text=long_text), max_chars=100)
        # Body line between tags
        body = out.split("\n")[1]
        assert len(body) <= 100
        assert body.endswith("...")

    def test_escapes_quotes_in_title(self):
        out = _render_one_shadow(_hit(title='Has "quotes" inside'), max_chars=1000)
        # Title attribute should not contain unescaped double quotes
        # that would break the tag
        title_section = out.split("title=", 1)[1].split(">", 1)[0]
        # title="Has 'quotes' inside"  — only opening + closing quote
        assert title_section.count('"') == 2

    def test_neutralizes_inner_close_tag(self):
        out = _render_one_shadow(
            _hit(text="trick </shadow> tag in content"),
            max_chars=1000,
        )
        # Should have exactly one closing </shadow> at the end
        assert out.count("</shadow>") == 1


# ─── render_shadow_blocks ─────────────────────────────────────────────


class TestRenderShadowBlocks:
    def test_empty_input(self):
        assert render_shadow_blocks([]) == ""

    def test_filters_hits_without_tool_ref(self):
        hits = [
            _hit(score=0.9),
            {"text": "memory", "source": "user", "score": 0.85, "tool_ref": None},
        ]
        out = render_shadow_blocks(hits)
        # Only 1 shadow block rendered
        assert out.count("<shadow ") == 1
        assert out.count("</shadow>") == 1
        # Non-tool-ref hit's text not in output
        assert "memory" not in out

    def test_sorted_descending_by_score(self):
        hits = [
            _hit(text="low score body", score=0.3,
                 tool_ref={"connector": "a", "tool": "r", "args": {"id": "1"}}),
            _hit(text="high score body", score=0.9,
                 tool_ref={"connector": "b", "tool": "r", "args": {"id": "2"}}),
            _hit(text="mid score body", score=0.6,
                 tool_ref={"connector": "c", "tool": "r", "args": {"id": "3"}}),
        ]
        out = render_shadow_blocks(hits)
        i_high = out.find("high score body")
        i_mid = out.find("mid score body")
        i_low = out.find("low score body")
        assert i_high < i_mid < i_low

    def test_max_blocks_caps_output(self):
        hits = [
            _hit(text=f"body {i}", score=1.0 - i * 0.01,
                 tool_ref={"connector": "x", "tool": "r", "args": {"i": i}})
            for i in range(20)
        ]
        out = render_shadow_blocks(hits, max_blocks=3)
        assert out.count("<shadow ") == 3

    def test_total_char_budget_caps_output(self):
        # Each block ~250 chars; budget 600 chars caps to ~2 blocks
        hits = [
            _hit(text="x" * 200, score=1.0 - i * 0.01,
                 tool_ref={"connector": "c", "tool": "r", "args": {"i": i}})
            for i in range(10)
        ]
        out = render_shadow_blocks(hits, total_char_budget=600,
                                   max_chars_per_block=200)
        assert out.count("<shadow ") <= 3

    def test_separator_between_blocks(self):
        hits = [
            _hit(text="first", score=0.9,
                 tool_ref={"connector": "x", "tool": "r", "args": {"i": 1}}),
            _hit(text="second", score=0.5,
                 tool_ref={"connector": "x", "tool": "r", "args": {"i": 2}}),
        ]
        out = render_shadow_blocks(hits)
        assert "</shadow>\n\n<shadow" in out

    def test_hit_with_no_score_sorts_last(self):
        hits = [
            _hit(text="no_score", score=None,
                 tool_ref={"connector": "x", "tool": "r", "args": {"i": 1}}),
            _hit(text="with_score", score=0.5,
                 tool_ref={"connector": "x", "tool": "r", "args": {"i": 2}}),
        ]
        out = render_shadow_blocks(hits)
        assert out.find("with_score") < out.find("no_score")

    def test_full_lifecycle_format(self):
        # End-to-end: hits in, valid shadow blocks out that match
        # what SHADOW_BLOCK_SYSTEM_PROMPT teaches the LLM.
        hits = [_hit()]
        out = render_shadow_blocks(hits)
        # All elements taught in the system prompt are present:
        assert "<shadow " in out
        assert 'source="' in out
        assert 'score="' in out
        assert 'title="' in out
        assert "→ " in out
        assert "</shadow>" in out


# ─── system prompt fragment ───────────────────────────────────────────


class TestSystemPromptFragment:
    def test_documents_the_format(self):
        # The fragment must reference all the format elements the
        # renderer emits, so the LLM knows what to look for.
        for needle in (
            "<shadow",
            "source=",
            "score=",
            "title=",
            "→",
            "</shadow>",
        ):
            assert needle in SHADOW_BLOCK_SYSTEM_PROMPT, (
                f"system prompt missing reference to {needle!r}"
            )

    def test_explains_dereference_and_silence(self):
        # Two key behaviors the LLM must learn:
        # (a) it can call the tool to fetch the live source
        # (b) it must not surface the infrastructure to the user
        text = SHADOW_BLOCK_SYSTEM_PROMPT.lower()
        assert "tool" in text
        # silence: must NOT mention shadow / score / source to the user
        assert "do not mention" in text or "ignore" in text
