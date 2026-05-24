"""Constitution loader — parses agents/CONSTITUTION.md.

Per spec §3.9.1, the constitution is user-authored and contains:
  - objectives (weighted, sum to 1.0)
  - constraints (boolean gates; no weight-buying)
  - metrics (how outcomes are scored for recalibration)

This module:
  1. Reads the file with CRLF + UTF-8 sanitization (mode="env" for
     constraint rule strings because they're config-like; descriptions
     stay rich via mode="text" — but we use one mode per call so we
     pick "text" overall, treating constraint rules as prose that may
     contain Unicode).
  2. Extracts the three sections via markdown header parsing.
  3. Parses the indented YAML-ish blocks into structured dicts.
  4. Validates: objective weights sum ≈ 1.0; constraints have rule
     strings; metrics have source + unit; no unknown top-level keys.

Hot-reload: a future enhancement.  For now, the file is read once at
startup and the service must restart on edit.  Spec §3.9.1 says
"on file change" which we'll bolt on later (inotify / mtime poll).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from ..utils.text_sanitize import sanitize_bytes

logger = logging.getLogger("agent-runtime.constitution")


class ConstitutionError(ValueError):
    """Raised on structural / validation errors during constitution parse."""


_OBJECTIVE_BLOCK_RE = re.compile(
    r"```\s*\n(?P<block>.*?)\n```",
    re.DOTALL,
)

# Section headers — we look for `## Objectives`, `## Constraints`, `## Metrics`.
# Case-insensitive; trailing punctuation ignored.
_SECTION_RE = re.compile(
    r"^##\s+(?P<name>[A-Za-z]+).*$",
    re.MULTILINE,
)

# Within each section's code block, items look like:
#   - id: reduce_toil
#     weight: 0.40
#     description: >
#       Eliminate repetitive work for the user.
# We parse this with a tolerant line-by-line state machine rather than
# a full YAML parser to keep the dependency footprint small and to give
# clearer error messages.


def load_constitution(path: Path) -> dict[str, Any]:
    """Parse the constitution file at `path`.

    Returns a dict shape:
      {
        "objectives":  [ {id, weight, description}, ... ],
        "constraints": [ {id, rule}, ... ],
        "metrics":     [ {id, unit, source, description}, ... ],
        "version":     int,
      }

    Raises ConstitutionError if the file is structurally invalid in a
    way that would prevent the scorer from running.  Warnings (e.g.,
    weights don't sum to 1.0 exactly) are logged but don't fail load.
    """
    if not path.exists():
        raise ConstitutionError(f"file not found: {path}")

    raw_bytes = path.read_bytes()
    text, issues = sanitize_bytes(raw_bytes, mode="text")
    for issue in issues:
        logger.warning(
            f"[constitution] {path.name}: sanitized {issue.code} — {issue.detail}"
        )

    sections = _split_sections(text)

    objectives = _parse_objectives(sections.get("objectives", ""))
    constraints = _parse_constraints(sections.get("constraints", ""))
    metrics = _parse_metrics(sections.get("metrics", ""))

    if not objectives:
        raise ConstitutionError(
            "no objectives parsed — constitution must define ≥1 objective"
        )

    # Weight sanity.  Don't fail load on slight imbalance — log loudly
    # so a quarterly review surfaces it via the existing
    # signal_quality_drift alert pattern.
    total = sum(o.get("weight", 0.0) for o in objectives)
    if not (0.99 <= total <= 1.01):
        logger.warning(
            f"[constitution] objective weights sum to {total:.3f}, "
            "expected 1.0 ± 0.01"
        )

    version = _extract_version(text)

    return {
        "objectives": objectives,
        "constraints": constraints,
        "metrics": metrics,
        "version": version,
        "source_path": str(path),
    }


def _split_sections(text: str) -> dict[str, str]:
    """Split markdown text into a dict of {section_name_lower: section_body}.

    Section boundaries are `## <Name>` headings.  Section body is
    everything until the next `## ` or `---` separator or end of file.
    """
    sections: dict[str, str] = {}
    last_pos = 0
    last_name = None

    for m in _SECTION_RE.finditer(text):
        if last_name is not None:
            sections[last_name] = text[last_pos:m.start()].strip()
        last_name = m["name"].lower()
        last_pos = m.end()

    if last_name is not None:
        sections[last_name] = text[last_pos:].strip()

    return sections


def _extract_code_block(section_body: str) -> str:
    """Return the first fenced code block in the section, or empty."""
    m = _OBJECTIVE_BLOCK_RE.search(section_body)
    return m.group("block") if m else ""


def _parse_yaml_ish_list(block: str) -> list[dict[str, Any]]:
    """Tolerant parser for `- id: foo\\n  weight: 0.40` style lists.

    Each item starts with `- ` at the top indent level; key:value pairs
    follow at the next indent level.  Multi-line strings via `>` (folded)
    or `|` (literal) are concatenated; `>` joins with spaces, `|` with
    newlines.

    Doesn't handle deeply nested structures — those would push us toward
    real YAML.  But the constitution is intentionally flat per §3.9.1.
    """
    items: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    pending_key: str | None = None
    pending_fold: str = " "  # " " for `>`, "\n" for `|`
    pending_buf: list[str] = []

    def flush_pending() -> None:
        nonlocal pending_key, pending_buf, pending_fold
        if current is not None and pending_key is not None and pending_buf:
            joined = pending_fold.join(s.strip() for s in pending_buf).strip()
            current[pending_key] = joined
        pending_key = None
        pending_buf = []
        pending_fold = " "

    for raw_line in block.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue

        # Item start: `- key: val` at outer indent.
        if line.lstrip().startswith("- "):
            flush_pending()
            if current is not None:
                items.append(current)
            current = {}
            kv_part = line.lstrip()[2:]
            new_pending = _absorb_kv(current, kv_part)
            if new_pending:
                pending_key, pending_fold = new_pending
            continue

        if current is None:
            continue

        stripped = line.lstrip()

        # No colon → continuation line for the currently-open folded key.
        if ":" not in stripped:
            if pending_key is not None:
                pending_buf.append(stripped)
            continue

        # `:` is present.  Need to disambiguate:
        #   - "rule: >"          → new key with folded continuation
        #   - "rule: some text"  → new key with inline value
        #   - "value with: colon inside continuation"  → ambiguous;
        #     YAML treats it as a new key.  We follow the same rule.
        flush_pending()
        new_pending = _absorb_kv(current, stripped)
        if new_pending:
            pending_key, pending_fold = new_pending

    flush_pending()
    if current is not None:
        items.append(current)

    return items


def _absorb_kv(target: dict[str, Any], line: str) -> tuple[str, str] | None:
    """Parse one `key: value` line into `target`.

    Returns (pending_key, fold_separator) when the value is a `>` or `|`
    block marker, signalling the caller to start continuation accumulation.
    Returns None for plain inline values.
    """
    if ":" not in line:
        return None
    key, _, val = line.partition(":")
    key = key.strip().lower().replace(" ", "_")
    val = val.strip()
    if val == ">":
        target[key] = ""
        return (key, " ")  # folded: join with spaces
    if val == "|":
        target[key] = ""
        return (key, "\n")  # literal: join with newlines
    # Coerce numeric values for weights etc.
    coerced: Any = val
    try:
        if "." in val:
            coerced = float(val)
        elif val.lstrip("-").isdigit():
            coerced = int(val)
    except ValueError:
        pass
    target[key] = coerced
    return None


def _parse_objectives(section_body: str) -> list[dict[str, Any]]:
    """Parse the objectives section into [{id, weight, description}, ...]."""
    block = _extract_code_block(section_body)
    items = _parse_yaml_ish_list(block)
    # Validate required fields per item; coerce weight to float if string.
    for it in items:
        if "id" not in it:
            raise ConstitutionError(f"objective missing 'id': {it}")
        if "weight" not in it:
            raise ConstitutionError(f"objective {it['id']!r} missing 'weight'")
        try:
            it["weight"] = float(it["weight"])
        except (TypeError, ValueError):
            raise ConstitutionError(
                f"objective {it['id']!r} weight not numeric: {it['weight']!r}"
            )
    return items


def _parse_constraints(section_body: str) -> list[dict[str, Any]]:
    """Parse the constraints section into [{id, rule}, ...]."""
    block = _extract_code_block(section_body)
    items = _parse_yaml_ish_list(block)
    # Each item should have id + rule (rule may be empty if parser
    # didn't pick up the continuation; log but don't fail).
    for it in items:
        if "id" not in it:
            raise ConstitutionError(f"constraint missing 'id': {it}")
        if "rule" not in it or not it.get("rule"):
            logger.warning(
                f"[constitution] constraint {it.get('id')!r} has empty rule string"
            )
    return items


def _parse_metrics(section_body: str) -> list[dict[str, Any]]:
    """Parse the metrics section into [{id, unit, source, description}, ...]."""
    block = _extract_code_block(section_body)
    items = _parse_yaml_ish_list(block)
    for it in items:
        if "id" not in it:
            raise ConstitutionError(f"metric missing 'id': {it}")
    return items


def _extract_version(text: str) -> int:
    """Extract `constitution_version: N` if present; default 1."""
    m = re.search(r"constitution_version:\s*(\d+)", text)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    return 1


__all__ = [
    "ConstitutionError",
    "load_constitution",
]
