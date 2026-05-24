"""Byte-stability tests for the PII redactor.

These are the hard invariants that protect Anthropic's prompt cache.
If any of these fails, the cached prefix would churn turn-over-turn
and every call would be a cache miss (= ~10x cost).

Invariants:
  A. Same input + same SessionId → same output bytes (determinism)
  B. Existing REF tokens NEVER change when new PII appears (append-only)
  C. Block dicts mutate only `.text`; cache_control + other keys preserved
  D. Pure substring replacement — no whitespace / Unicode normalization
  E. Rehydration round-trips losslessly for in-session tokens

Run:
    pytest pii/tests/test_byte_stability.py -v
"""

from __future__ import annotations

import hashlib
import os
import tempfile

import pytest


@pytest.fixture(autouse=True)
def isolated_lance_dir(tmp_path, monkeypatch):
    """Point the PII store at a fresh Lance dir per test.

    Resets the singleton so a fresh PIISessionStore is constructed
    against the new dir.
    """
    db_dir = tmp_path / "pii_db"
    db_dir.mkdir()
    monkeypatch.setenv("VEILGUARD_PII_DB_PATH", str(db_dir))

    # Reset singletons.
    from pii import session_store, redactor as redactor_mod
    session_store.PIISessionStore._instance = None
    redactor_mod.PIIRedactor._instance = None
    yield
    session_store.PIISessionStore._instance = None
    redactor_mod.PIIRedactor._instance = None


# ─────────────────────────────────────────────────────────────────────


def _sid(conv="conv-1", tenant="t1"):
    from pii import SessionId
    return SessionId(tenant_id=tenant, conv_id=conv)


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


# Invariant A — determinism ──────────────────────────────────────────


def test_same_input_same_output_50_calls():
    """Same text + same session = same bytes, every time."""
    from pii import get_redactor
    text = "Email Alice Johnson at alice@example.com or call +27 82 555 1234."
    sid = _sid()
    hashes = set()
    for _ in range(50):
        out = get_redactor().redact_text(text, sid)
        hashes.add(_hash(out))
    assert len(hashes) == 1, f"50 redactions produced {len(hashes)} distinct outputs"


def test_same_input_different_session_may_differ():
    """Different SessionId may produce different REF token numbers."""
    from pii import get_redactor
    text = "Contact Bob Smith."
    sid_a = _sid(conv="conv-a")
    sid_b = _sid(conv="conv-b")
    out_a = get_redactor().redact_text(text, sid_a)
    out_b = get_redactor().redact_text(text, sid_b)
    # Both should redact "Bob Smith" but counter spaces are per-session.
    # Within a fresh session each is REF_PERSON_1 — but that's an impl
    # detail; we don't pin it across sessions.
    assert "REF_PERSON_" in out_a
    assert "REF_PERSON_" in out_b


# Invariant B — append-only ──────────────────────────────────────────


def test_existing_token_never_changes_when_new_pii_appears():
    """Turn 1 sees Alice → REF_PERSON_1.  Turn 5 sees Bob (new).
    Alice's token stays REF_PERSON_1 in turn 5."""
    from pii import get_redactor
    sid = _sid()
    r = get_redactor()

    # Turn 1
    out1 = r.redact_text("Hello from Alice Johnson.", sid)
    alice_token_t1 = out1.split("from ")[1].split(".")[0]
    assert alice_token_t1.startswith("REF_PERSON_")

    # Turns 2-4 — different content
    r.redact_text("Today's weather is sunny.", sid)
    r.redact_text("The deployment succeeded.", sid)
    r.redact_text("Review the new feature.", sid)

    # Turn 5 — both Alice and Bob mentioned
    out5 = r.redact_text("Tell Bob Smith and Alice Johnson the news.", sid)
    # Alice's token from turn 1 must still appear unchanged
    assert alice_token_t1 in out5, (
        f"Alice's token {alice_token_t1!r} changed.  "
        f"Turn 5 output: {out5!r}"
    )


def test_same_pii_in_two_turns_gets_same_token():
    from pii import get_redactor
    sid = _sid()
    out1 = get_redactor().redact_text("Email alice@example.com", sid)
    out2 = get_redactor().redact_text("Reply to alice@example.com", sid)
    # Extract token
    tok1 = out1.split("Email ")[1].strip()
    tok2 = out2.split("Reply to ")[1].strip()
    assert tok1 == tok2, f"Same email produced different tokens: {tok1!r} vs {tok2!r}"


# Invariant C — block structure preserved ────────────────────────────


def test_redact_blocks_preserves_cache_control():
    """cache_control marker MUST survive block redaction unchanged."""
    from pii import get_redactor
    sid = _sid()
    blocks = [
        {
            "type": "text",
            "text": "Static preamble.  No PII here.",
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        },
        {
            "type": "text",
            "text": "User Alice asked a question.",
            "cache_control": {"type": "ephemeral"},
        },
    ]
    out = get_redactor().redact_blocks(blocks, sid)
    assert len(out) == 2
    # cache_control on block 0 unchanged
    assert out[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    # cache_control on block 1 unchanged
    assert out[1]["cache_control"] == {"type": "ephemeral"}
    # block 0 text unchanged (no PII)
    assert out[0]["text"] == "Static preamble.  No PII here."
    # block 1 text redacted (Alice → REF_PERSON_N)
    assert "Alice" not in out[1]["text"]
    assert "REF_PERSON_" in out[1]["text"]


def test_redact_blocks_preserves_arbitrary_extra_keys():
    """Custom keys like _veilguard, _skip_pii should pass through."""
    from pii import get_redactor
    sid = _sid()
    blocks = [{
        "type": "text",
        "text": "Hi Charlie Brown.",
        "_veilguard": {"connector": "sharepoint"},
        "custom_field": [1, 2, 3],
    }]
    out = get_redactor().redact_blocks(blocks, sid)
    assert out[0]["_veilguard"] == {"connector": "sharepoint"}
    assert out[0]["custom_field"] == [1, 2, 3]


def test_non_text_blocks_unchanged():
    from pii import get_redactor
    sid = _sid()
    blocks = [
        {"type": "tool_use", "name": "read_file", "input": {"path": "/tmp/x"}, "id": "tu_1"},
        {"type": "image", "source": {"type": "base64", "data": "..."}},
    ]
    out = get_redactor().redact_blocks(blocks, sid)
    assert out == blocks


# Invariant D — pure substring replacement ───────────────────────────


def test_no_whitespace_mutation_on_text_with_no_pii():
    """Text without detectable PII must be returned byte-identical.

    Watching for: stray .strip(), CRLF normalization, BOM injection,
    Unicode normalization — all of which break cache stability.
    """
    from pii import get_redactor
    sid = _sid()
    samples = [
        "Simple ASCII line.",
        "Line with    multiple   spaces.",
        "Trailing whitespace here.   ",
        "   Leading whitespace.",
        "Line1\nLine2\nLine3",
        "Line1\r\nLine2\r\nLine3",  # CRLF must NOT normalize
        "Unicode: café résumé naïve",
        "Quotes: “Hello” and ‘world’",
        "Tab\there.",
    ]
    for s in samples:
        out = get_redactor().redact_text(s, sid)
        # When no PII is detected, output MUST equal input byte-for-byte.
        # If Presidio detects something (e.g. PERSON false-positive on
        # "café résumé"), skip — we only assert on truly PII-free.
        if "REF_" not in out:
            assert out == s, (
                f"Byte mutation on PII-free input: "
                f"{s!r} → {out!r}"
            )


# Invariant E — rehydration round-trip ───────────────────────────────


def test_rehydrate_round_trip():
    from pii import get_redactor
    sid = _sid()
    original = "Alice Johnson emails alice@example.com from +27 82 555 1234."
    redacted = get_redactor().redact_text(original, sid)
    rehydrated = get_redactor().rehydrate_text(redacted, sid)
    assert rehydrated == original, (
        f"Round-trip failed:\n"
        f"  original:  {original!r}\n"
        f"  redacted:  {redacted!r}\n"
        f"  rehydrated:{rehydrated!r}"
    )


def test_rehydrate_blocks_preserves_structure():
    from pii import get_redactor
    sid = _sid()
    blocks = [
        {"type": "text", "text": "Email Alice at alice@example.com.",
         "cache_control": {"type": "ephemeral"}},
    ]
    redacted = get_redactor().redact_blocks(blocks, sid)
    rehydrated = get_redactor().rehydrate_blocks(redacted, sid)
    assert rehydrated[0]["cache_control"] == {"type": "ephemeral"}
    assert rehydrated[0]["text"] == "Email Alice at alice@example.com."


def test_rehydrate_tool_use_args():
    """tool_use blocks with REF tokens in input args must rehydrate."""
    from pii import get_redactor
    sid = _sid()
    r = get_redactor()
    # Seed the mapping with a token
    r.redact_text("Send to alice@example.com", sid)
    # Now simulate a tool_use the model emitted with the token
    blocks = [{
        "type": "tool_use",
        "name": "send_email",
        "id": "tu_1",
        "input": {"to": "REF_EMAIL_1", "subject": "Hi"},
    }]
    out = r.rehydrate_blocks(blocks, sid)
    assert out[0]["input"]["to"] == "alice@example.com"
    assert out[0]["input"]["subject"] == "Hi"
    # Structural fields unchanged
    assert out[0]["name"] == "send_email"
    assert out[0]["id"] == "tu_1"


# Cross-cutting — sub-cid parent resolution ──────────────────────────


def test_sub_cid_shares_parent_mapping():
    """sub-conv-1-researcher must share REF tokens with conv-1.

    This is the cache-stability property across agents in the same
    conversation.  Director redacts in `conv-1`, Researcher gets
    invoked in `sub-conv-1-researcher`; they must see Alice mapped
    to the SAME REF_PERSON_N.
    """
    from pii import get_redactor, SessionId
    parent = SessionId(tenant_id="t1", conv_id="conv-1")
    child = SessionId(tenant_id="t1", conv_id="sub-conv-1-researcher")

    out_parent = get_redactor().redact_text("Tell Alice the news.", parent)
    out_child = get_redactor().redact_text("Tell Alice the news.", child)

    # Both should produce identical output (same token for Alice)
    assert out_parent == out_child, (
        f"Sub-cid produced different tokens:\n"
        f"  parent: {out_parent!r}\n"
        f"  child:  {out_child!r}"
    )


# Persistence across process restarts ────────────────────────────────


def test_store_survives_redactor_singleton_reset():
    """Re-instantiating the redactor reads existing mappings from Lance.

    Simulates a process restart: the in-memory singleton is gone but
    the Lance table has the row.  New redactor must produce the SAME
    redacted output bytes for the same input + same SessionId.
    """
    from pii import redactor as redactor_mod
    from pii import session_store

    sid = _sid()
    text = "Contact me at alice@example.com please."

    out1 = redactor_mod.get_redactor().redact_text(text, sid)

    # Reset both singletons (simulate restart)
    redactor_mod.PIIRedactor._instance = None
    session_store.PIISessionStore._instance = None

    out2 = redactor_mod.get_redactor().redact_text(text, sid)

    # Same input + same SessionId must produce IDENTICAL bytes after
    # a singleton reset.  This is the cache-stability invariant for
    # multi-process / restart scenarios.
    assert out1 == out2, (
        f"Restart produced different bytes:\n"
        f"  before: {out1!r}\n"
        f"  after:  {out2!r}"
    )
