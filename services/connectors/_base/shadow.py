"""Shadow-block rendering for connector-sourced recall hits.

A shadow block is a candidate from an external system (SharePoint,
Slack, Jira, ...) injected into the LLM's context with a dereference
instruction. The LLM either uses the excerpt directly or calls the
named tool to fetch the live source.

The rendering convention is the wire contract between
`tcmm.recall_structured()` output and the LLM. The system-prompt
fragment :data:`SHADOW_BLOCK_SYSTEM_PROMPT` teaches the LLM how to
read these blocks; it should be inserted into every system prompt
that exposes connector tools.

Distinct from TCMM's existing "shadow_blocks" concept (recalled
memory staged into context). These shadow blocks are external
candidates with a tool_ref — pure-TCMM memory hits (no tool_ref)
are filtered out by :func:`render_shadow_blocks` and rendered
elsewhere by TCMM's own memory-block formatter.

Token budgeting is char-based (~4 chars/token). Both per-block and
total budgets are caller-controlled; defaults aim for ~2000 tokens
total and ~150 tokens per block.
"""
from __future__ import annotations

from typing import Any


SHADOW_BLOCK_SYSTEM_PROMPT = """\
## Shadow Memory Blocks (external connector candidates)

Your context may include `<shadow>` blocks. Each is a candidate document
or message from an external system (SharePoint, Slack, Jira, etc.) that
the recall layer judged relevant to the current request. You have NOT
retrieved this content yourself — it is a hint based on semantic
matching, presented for you to either use directly or dereference.

Shape:

    <shadow source="<connector>" score="<0.00-1.00>" title="<doc title>">
    <truncated excerpt>
    → <connector>.<tool>(<args>)
    </shadow>

When a shadow block is relevant:
  1. If the excerpt is enough to answer the user accurately, use it
     directly. Cite the title in your reply where natural.
  2. If you need the full source, call the exact tool shown after the
     arrow with the exact arguments. The tool will return the live
     content (which may differ from the excerpt if the source changed).
  3. With multiple relevant blocks, prefer higher score values. You
     may dereference more than one when needed.

When a shadow block is NOT relevant, ignore it. Do not mention shadow
blocks, score values, the connector source, or the dereference syntax
to the user — they are internal infrastructure.

Permissions are enforced before you see anything: shadow blocks only
appear for sources the current user is authorized to read.
"""


def render_shadow_blocks(
    hits: list[dict[str, Any]],
    *,
    max_blocks: int = 8,
    max_chars_per_block: int = 600,
    total_char_budget: int = 8000,
) -> str:
    """Render connector-sourced recall hits as shadow blocks.

    ``hits`` is the output of ``VeilguardTCMM.recall_structured()`` (or
    equivalent — any list of dicts carrying ``tool_ref`` / ``source`` /
    ``text`` / ``title`` / ``score``). Hits without ``tool_ref`` are
    skipped — they are pure-TCMM memory entries handled elsewhere.

    Sort order: descending by ``score`` (hits without a numeric score
    sort last). Output is the concatenation of rendered blocks
    separated by blank lines, capped by ``max_blocks`` and
    ``total_char_budget``.
    """
    if not hits:
        return ""

    candidates = [h for h in hits if isinstance(h, dict) and h.get("tool_ref")]

    def _sort_key(h: dict[str, Any]) -> float:
        s = h.get("score")
        if isinstance(s, (int, float)):
            return -float(s)
        return float("inf")

    candidates.sort(key=_sort_key)

    rendered: list[str] = []
    total = 0
    separator_len = 2  # blank line between blocks
    for hit in candidates[:max_blocks]:
        block = _render_one_shadow(hit, max_chars=max_chars_per_block)
        if not block:
            continue
        added = len(block) + (separator_len if rendered else 0)
        if total + added > total_char_budget:
            break
        rendered.append(block)
        total += added

    return "\n\n".join(rendered)


def _render_one_shadow(hit: dict[str, Any], *, max_chars: int) -> str:
    tool_ref = hit.get("tool_ref")
    if not isinstance(tool_ref, dict):
        return ""

    source = hit.get("source") or "unknown"
    score = hit.get("score")
    title = hit.get("title")
    content = hit.get("text") or ""

    # Defensive: prevent literal "</shadow>" inside the excerpt from
    # closing the wrapper prematurely. LLMs are tolerant of slight
    # mangling but a literal collision would split the block.
    if "</shadow>" in content:
        content = content.replace("</shadow>", "</ shadow>")

    if len(content) > max_chars:
        content = content[: max_chars - 3].rstrip() + "..."

    attrs: list[str] = [f'source="{_escape_attr(str(source))}"']
    if isinstance(score, (int, float)):
        attrs.append(f'score="{float(score):.2f}"')
    if isinstance(title, str) and title:
        attrs.append(f'title="{_escape_attr(title)}"')

    open_tag = "<shadow " + " ".join(attrs) + ">"
    tool_call = _render_tool_call(tool_ref)

    return f"{open_tag}\n{content}\n→ {tool_call}\n</shadow>"


def _render_tool_call(tool_ref: dict[str, Any]) -> str:
    """Render a tool_ref dict as a Python-style call signature.

    Mirrors `Ref.to_tool_call()` from types.py — kept independent
    because tool_ref arrives as a plain dict (after JSON round-trip
    through TCMM), not a typed Ref instance.
    """
    connector = tool_ref.get("connector") or "?"
    tool = tool_ref.get("tool") or "?"
    args = tool_ref.get("args") or {}
    if not isinstance(args, dict):
        args = {}
    kw_parts = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return f"{connector}.{tool}({kw_parts})"


def _escape_attr(value: str) -> str:
    """Minimal attribute-value escaping. Replaces double-quotes with
    single-quotes and strips backslashes — keeps the open tag parseable
    without pulling in a full XML escaper."""
    return value.replace('"', "'").replace("\\", "/")
