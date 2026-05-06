"""Veilguard provenance envelope for connector tool results.

Connectors wrap their tool output with this envelope so the TCMM adapter
can extract `acl`, `tool_ref`, `etag`, `title`, and the connector name
into the archive entry's `extras_json`. Later, `recall_structured()`
surfaces those fields as shadow-block provenance.

Wire shape::

    {
      "content": "<text the LLM would see>",
      "_veilguard": {
        "connector": "sharepoint",
        "tool_ref":  {"connector": "sharepoint", "tool": "read",
                      "args": {"item_id": "...", "site_id": "..."}},
        "acl":       ["group:engineering", "group:execs"],
        "etag":      "v3",
        "title":     "Q1 Project Status"
      }
    }

The LLM does **not** see the envelope: the PII proxy strips `_veilguard`
from `tool_result` content before forwarding to the LLM. Only `content`
makes it into the conversation context.

This module owns the connector-side `wrap_with_provenance`. The TCMM
adapter has its own parser that consumes the same shape — both sides
must agree on this format. The round-trip tests in
`tests/test_envelope.py` lock the contract.
"""
from __future__ import annotations

import json
from typing import Any


_VEILGUARD_KEY = "_veilguard"


def wrap_with_provenance(
    content: str,
    *,
    connector: str,
    tool_ref: dict | None = None,
    acl: list[str] | None = None,
    etag: str | None = None,
    title: str | None = None,
) -> str:
    """Return a `_veilguard`-tagged JSON envelope for a connector tool result.

    `content` is what the LLM will see (after the PII proxy strips the
    envelope). All other fields land in the TCMM archive entry's
    `extras_json` for later recall.

    Fields with value None or empty list are omitted from the envelope
    to keep the wire payload tight.
    """
    vg: dict[str, Any] = {"connector": connector}
    if tool_ref is not None:
        vg["tool_ref"] = tool_ref
    if acl:
        vg["acl"] = list(acl)
    if etag is not None:
        vg["etag"] = etag
    if title is not None:
        vg["title"] = title

    payload = {"content": content, _VEILGUARD_KEY: vg}
    return json.dumps(payload, ensure_ascii=False)


def parse_envelope(text: str) -> tuple[str, dict]:
    """Reverse of `wrap_with_provenance`.

    Returns `(content, metadata)`:
      * If `text` is a valid envelope: `metadata` carries the
        `_veilguard` dict; `content` is the inner content string.
      * Otherwise: `(text, {})` — caller treats it as plain content
        (existing tool-result behavior).

    The parser is permissive: malformed JSON, non-dict payloads, or
    JSON without a `_veilguard` key all pass through untouched. This
    keeps the adapter ingest path safe for any connector that hasn't
    been migrated to the envelope yet.
    """
    if not text or not text.lstrip().startswith("{"):
        return text, {}
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError, TypeError):
        return text, {}
    if not isinstance(parsed, dict) or _VEILGUARD_KEY not in parsed:
        return text, {}

    content = parsed.get("content", "")
    if not isinstance(content, str):
        content = str(content)
    metadata = parsed.get(_VEILGUARD_KEY, {})
    if not isinstance(metadata, dict):
        metadata = {}
    return content, metadata
