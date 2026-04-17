"""Tests for Anthropic prompt caching helpers in pii-proxy.

Run: pytest pii-proxy/tests/test_cache.py
"""

import sys
import os

# Make the app package importable whether running from repo root or pii-proxy/
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here))

from app.main import _split_for_cache, _apply_anthropic_cache, _MIN_CACHE_CHARS


# ── _split_for_cache ─────────────────────────────────────────────────────────


def test_split_no_marker_returns_nosplit():
    """When no marker is given, return the prompt untouched."""
    prompt = "x" * 10000
    prefix, tail = _split_for_cache(prompt, marker=None)
    assert prefix is None
    assert tail == prompt


def test_split_marker_missing_returns_nosplit():
    """When the marker is not in the prompt, return the prompt untouched."""
    prompt = "x" * 10000
    prefix, tail = _split_for_cache(prompt, marker="USER TASK:")
    assert prefix is None
    assert tail == prompt


def test_split_marker_present_but_prefix_too_short():
    """Prefix under the minimum cache threshold — don't split."""
    prompt = "short prefix\n\nUSER TASK: what now?"
    prefix, tail = _split_for_cache(prompt, marker="USER TASK:")
    assert prefix is None
    assert tail == prompt


def test_split_marker_present_long_prefix():
    """Prefix is long enough — split at the marker."""
    stable = "x" * (_MIN_CACHE_CHARS + 100)
    marker = "\n\nUSER TASK:"
    tail_text = " what is the capital of France?"
    prompt = stable + marker + tail_text
    prefix, tail = _split_for_cache(prompt, marker=marker)
    assert prefix == stable
    assert tail == marker + tail_text
    # Reconstruction must match the input exactly.
    assert prefix + tail == prompt


# ── _apply_anthropic_cache ───────────────────────────────────────────────────


def test_apply_cache_short_system_noop():
    """Short system message — don't touch it."""
    data = {"system": "short", "messages": [{"role": "user", "content": "hi"}]}
    markers = _apply_anthropic_cache(data)
    assert markers == 0
    assert data["system"] == "short"  # untouched


def test_apply_cache_long_system_wraps_in_block():
    """Long system string — convert to structured content with cache_control."""
    long_sys = "memory context " * 500  # ~7500 chars
    data = {"system": long_sys, "messages": [{"role": "user", "content": "hi"}]}
    markers = _apply_anthropic_cache(data)
    assert markers == 1
    assert isinstance(data["system"], list)
    assert data["system"][0]["type"] == "text"
    assert data["system"][0]["text"] == long_sys
    assert data["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_apply_cache_structured_system_adds_marker():
    """System already in list form — add cache_control to the last block."""
    long_text = "x" * (_MIN_CACHE_CHARS + 100)
    data = {
        "system": [{"type": "text", "text": long_text}],
        "messages": [{"role": "user", "content": "hi"}],
    }
    markers = _apply_anthropic_cache(data)
    assert markers == 1
    assert data["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_apply_cache_history_marker_with_many_turns():
    """Long conversation — mark the penultimate message to cache history."""
    long_sys = "x" * (_MIN_CACHE_CHARS + 100)
    data = {
        "system": long_sys,
        "messages": [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer " * 50},  # >200 chars
            {"role": "user", "content": "second question"},
            {"role": "assistant", "content": "second answer " * 50},
            {"role": "user", "content": "latest question"},
        ],
    }
    markers = _apply_anthropic_cache(data)
    # One for system, one for history
    assert markers == 2

    # The message right before the final user turn should now have cache_control.
    target = data["messages"][3]  # the assistant just before final user
    assert isinstance(target["content"], list)
    assert target["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_apply_cache_few_messages_no_history_marker():
    """Only 2 messages — don't bother caching history."""
    long_sys = "x" * (_MIN_CACHE_CHARS + 100)
    data = {
        "system": long_sys,
        "messages": [{"role": "user", "content": "hi"}],
    }
    markers = _apply_anthropic_cache(data)
    assert markers == 1  # only system


if __name__ == "__main__":
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pytest", __file__, "-v"])
