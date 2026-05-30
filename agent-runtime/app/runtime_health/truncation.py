"""Phase 6.5 — Truncated tool output marker.

Sibling of the observe-silent-failure bug closed in 2026-05-27.  When
a tool wrapper caps response size (read_file, web_search, inbox), the
caller sees a prefix that looks identical to a complete response.  The
LLM reasons over the prefix as if complete; that's the same epistemic
failure class as observe-silent-failure.

Phase 6.5 fix: every tool wrapper emits an explicit
`[TRUNCATED: <N> of <M> bytes shown — page or chunk before acting]`
tail.  Persona prompts include the one-line rule:

  > If a tool result ends with `[TRUNCATED: ...]`, the response is
  > incomplete.  Either call the tool again with pagination args,
  > or raise a `blocker_raised` comment and `submit_for_review` with
  > what you have.  Do not reason over a truncated response as if
  > complete.

This module is the pure helper used by tool wrappers.
"""

from __future__ import annotations

import re

# The exact marker shape.  Other modules (linters, persona prompts)
# look for this regex to verify the marker is present in tool wrappers.
TRUNCATION_MARKER_RE = re.compile(
    r"\[TRUNCATED:\s+\d+\s+of\s+\d+\s+bytes shown"
)


def truncate_with_marker(
    body: str,
    *,
    max_bytes: int,
    encoding: str = "utf-8",
) -> str:
    """Return `body` truncated to at most `max_bytes` bytes, with the
    TRUNCATED marker appended IF truncation occurred.

    The marker carries the original size + the truncation point so the
    LLM can decide whether to paginate, chunk, or raise a blocker.
    """
    if max_bytes <= 0:
        raise ValueError(f"max_bytes must be positive, got {max_bytes}")
    body_bytes = body.encode(encoding, errors="replace")
    total = len(body_bytes)
    if total <= max_bytes:
        return body
    # Reserve room for the marker.
    marker_template = (
        f"\n[TRUNCATED: {{shown}} of {total} bytes shown — page or chunk before acting]"
    )
    # Estimate marker size with a worst-case `shown` value.
    marker_size_estimate = len(marker_template.format(shown=total))
    show_bytes = max(0, max_bytes - marker_size_estimate)
    head = body_bytes[:show_bytes].decode(encoding, errors="replace")
    marker = marker_template.format(shown=show_bytes)
    return head + marker


def has_truncation_marker(body: str) -> bool:
    """True iff the body ends with a TRUNCATED marker."""
    return bool(TRUNCATION_MARKER_RE.search(body))
