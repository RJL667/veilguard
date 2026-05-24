"""Tests for the shared parsing + chunking layer.

`_base/parsing.py` is the single point of truth for "bytes → text"
and "text → chunks" for every connector. These tests pin that
contract independently of any specific connector. Tests degrade
gracefully when LlamaIndex isn't installed (fallback path).
"""
from __future__ import annotations

import pathlib
import sys

import pytest

_CONNECTORS_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CONNECTORS_DIR))

from _base.parsing import chunk_text, parse_to_text  # noqa: E402


# ─── parse_to_text ────────────────────────────────────────────────────


class TestParseToTextBasics:
    def test_empty_bytes_returns_empty(self):
        assert parse_to_text(b"", "x.txt") == ""

    def test_plain_text_decoded(self):
        out = parse_to_text(b"hello world", "notes.txt")
        assert out == "hello world"

    def test_text_mime_type_dispatches_to_decode(self):
        out = parse_to_text(b"raw bytes", "noext", mime_type="text/plain")
        assert out == "raw bytes"

    def test_latin1_fallback_for_invalid_utf8(self):
        # \xe9 is invalid UTF-8 standalone but valid latin-1 ('é')
        out = parse_to_text(b"caf\xe9", "snack.txt")
        # Either the latin-1 decode (preferred) or a replacement char
        # — both are acceptable; the contract is "no exception, non-empty
        # output, contains 'caf'".
        assert "caf" in out
        assert len(out) >= 3

    def test_unknown_extension_returns_empty_or_placeholder(self):
        # Could be either depending on whether LlamaIndex is installed
        out = parse_to_text(b"\x00\x01\x02", "thing.xyz123")
        assert out == "" or out.startswith("[binary content")


# ─── chunk_text ───────────────────────────────────────────────────────


class TestChunkText:
    def test_empty_returns_empty(self):
        assert chunk_text("") == []

    def test_whitespace_returns_empty(self):
        assert chunk_text("   \n\n  \t  ") == []

    def test_short_text_returns_at_least_one_chunk(self):
        chunks = chunk_text("A short sentence.")
        assert len(chunks) >= 1
        assert "short sentence" in chunks[0]

    def test_long_text_produces_multiple_chunks(self):
        # 600 sentences ≈ 7-8K words ≈ 10-12K tokens — well over the
        # 512-token default, so we expect several chunks.
        text = " ".join(f"Sentence number {i} with body text." for i in range(600))
        chunks = chunk_text(text)
        assert len(chunks) > 1

    def test_no_empty_chunks(self):
        text = " ".join(f"Sentence {i}." for i in range(50))
        chunks = chunk_text(text)
        for chunk in chunks:
            assert chunk.strip(), f"empty chunk in output: {chunk!r}"

    def test_custom_size_passed_through(self):
        # Large text with a small chunk size should produce many chunks
        text = " ".join(f"Sentence {i}." for i in range(100))
        few = chunk_text(text, chunk_size_tokens=512)
        many = chunk_text(text, chunk_size_tokens=64)
        assert len(many) >= len(few)


# ─── fallback path (LlamaIndex absent) ────────────────────────────────


class TestFallbackBehavior:
    """When LlamaIndex isn't importable, parse_to_text returns
    empty/placeholder for non-plain-text formats and chunk_text
    falls back to a paragraph split. We can't reliably uninstall
    LlamaIndex per-test, so these tests just verify the fallback
    helpers themselves work in isolation."""

    def test_fallback_paragraph_split(self):
        from _base.parsing import _fallback_paragraph_split
        out = _fallback_paragraph_split("Para 1.\n\nPara 2.\n\nPara 3.")
        assert out == ["Para 1.", "Para 2.", "Para 3."]

    def test_fallback_drops_empty_paragraphs(self):
        from _base.parsing import _fallback_paragraph_split
        out = _fallback_paragraph_split("A\n\n  \n\nB")
        assert out == ["A", "B"]
