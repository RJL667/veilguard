"""Server-side workspace filesystem tools (agent-runtime).

[WORKSPACE_FS_2026_05_29]  Spec §3.5 is explicit: "The org lives on the
VM ... Client-daemon as a tool, not a home."  Team-workspace artifacts
(`team/drafts/cost-analysis.md`, `agents/<aid>/...`) are SERVER-SIDE
state — they must NOT route to the user's client-daemon.

Before this module, `read_file`/`write_file`/`edit_file` were CLIENT
tools (see `agent/client_tools.py` + tool_dispatcher Path 2 → sub-agents
→ daemon WS bridge).  When sub-agents/daemon weren't running (the
default local-dev posture, and any prod moment the daemon is offline),
EVERY artifact read/write returned `isError: TOOL UNAVAILABLE`:

  * researcher calls write_file("team/drafts/x.md") → fails → no artifact
  * critic-prose calls read_file("team/drafts/x.md") → fails → can't review
  * critic loops ("I need to find the actual task ID") until max_turns

These handlers operate on a sandboxed workspace directory mounted into
the container (`VEILGUARD_WORKSPACE_ROOT`, default `/workspace`).  The
tool_dispatcher registers them in Path 1 (in-process), so they take
precedence over the daemon fallback.  Tool I/O contract matches
`agent/client_tools.CLIENT_TOOL_SCHEMAS` exactly so the LLM sees one
consistent schema regardless of which side executes.

Sandboxing: every path is resolved UNDER the workspace root.  A leading
`/` or `\\`, a drive letter, or any `..` segment that would escape the
root is rejected.  Agents cannot read/write outside the workspace.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("agent-runtime.tools.workspace_fs")


def _workspace_root() -> Path:
    root = os.environ.get("VEILGUARD_WORKSPACE_ROOT", "/workspace")
    p = Path(root)
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.warning(f"[workspace_fs] could not ensure root {root!r}: {e}")
    return p


def _ok(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "isError": False}


def _err(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": f"ERROR: {text}"}], "isError": True}


def _resolve(path: str) -> tuple[Path | None, str]:
    """Resolve `path` under the workspace root with sandbox enforcement.

    Returns (resolved_path, "") on success, or (None, error_message) if
    the path would escape the workspace.
    """
    if not path or not isinstance(path, str):
        return None, "path is required and must be a string"
    root = _workspace_root().resolve()
    # Normalise: strip a leading slash/backslash and any drive letter so
    # "team/drafts/x.md", "/team/drafts/x.md", and "C:\\team\\x.md" all
    # land under the workspace root rather than escaping it.
    cleaned = path.replace("\\", "/").lstrip("/")
    # Drop a Windows drive prefix like "C:" if the model emitted one.
    if len(cleaned) >= 2 and cleaned[1] == ":":
        cleaned = cleaned[2:].lstrip("/")
    candidate = (root / cleaned).resolve()
    # Sandbox: resolved path must stay within root.
    try:
        candidate.relative_to(root)
    except ValueError:
        return None, (
            f"path {path!r} escapes the workspace sandbox "
            f"(resolved to {candidate}); refused"
        )
    return candidate, ""


async def read_file(args: dict[str, Any]) -> dict[str, Any]:
    """Read a workspace file.  Returns `<lineno>\\t<line>` format to match
    the client-daemon read_file contract the personas expect.
    """
    path = args.get("path", "")
    target, err = _resolve(path)
    if err:
        return _err(err)
    if not target.exists():
        return _err(
            f"file not found at {path!r} (workspace: {target}). "
            f"If you intended to review an artifact, the producer may not "
            f"have written it yet — do NOT loop; report the missing file."
        )
    if target.is_dir():
        return _err(f"{path!r} is a directory, not a file")
    try:
        offset = int(args.get("offset", 0) or 0)
    except (TypeError, ValueError):
        offset = 0
    try:
        limit = int(args.get("limit", 500) or 500)
    except (TypeError, ValueError):
        limit = 500
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return _err(f"could not read {path!r}: {e}")
    lines = text.splitlines()
    window = lines[offset:offset + limit]
    numbered = "\n".join(
        f"{offset + i}\t{ln}" for i, ln in enumerate(window)
    )
    note = ""
    if offset + limit < len(lines):
        note = (
            f"\n\n[truncated — showing lines {offset}..{offset + limit} "
            f"of {len(lines)}; re-call with a larger offset to continue]"
        )
    return _ok(numbered + note if numbered else "(empty file)")


async def write_file(args: dict[str, Any]) -> dict[str, Any]:
    """Create or overwrite a workspace file.  Atomic (temp + rename).
    Creates parent directories as needed.
    """
    path = args.get("path", "")
    target, err = _resolve(path)
    if err:
        return _err(err)
    content = args.get("content")
    if content is None:
        return _err("write_file: both 'path' and 'content' are required")
    if not isinstance(content, str):
        content = str(content)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(target)
    except Exception as e:
        return _err(f"could not write {path!r}: {e}")
    return _ok(
        f"wrote {len(content)} bytes to {path!r} "
        f"({content.count(chr(10)) + 1} lines)"
    )


async def edit_file(args: dict[str, Any]) -> dict[str, Any]:
    """Exact-string replace in a workspace file.  old_string must match
    exactly and be unique.
    """
    path = args.get("path", "")
    target, err = _resolve(path)
    if err:
        return _err(err)
    if not target.exists():
        return _err(f"edit_file: file not found at {path!r} — write it first")
    old = args.get("old_string")
    new = args.get("new_string")
    if old is None or new is None:
        return _err("edit_file: 'old_string' and 'new_string' are required")
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return _err(f"could not read {path!r}: {e}")
    count = text.count(old)
    if count == 0:
        return _err(
            f"edit_file: old_string not found in {path!r}. "
            f"Read the file first to copy the exact text (including whitespace)."
        )
    if count > 1:
        return _err(
            f"edit_file: old_string appears {count}× in {path!r}; "
            f"it must be unique. Include more surrounding context."
        )
    updated = text.replace(old, new, 1)
    try:
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(updated, encoding="utf-8")
        tmp.replace(target)
    except Exception as e:
        return _err(f"could not write {path!r}: {e}")
    return _ok(f"edited {path!r} (1 replacement)")


async def list_directory(args: dict[str, Any]) -> dict[str, Any]:
    """List entries under a workspace directory."""
    path = args.get("path", "") or "."
    target, err = _resolve(path)
    if err:
        return _err(err)
    if not target.exists():
        return _err(f"directory not found at {path!r}")
    if not target.is_dir():
        return _err(f"{path!r} is a file, not a directory")
    try:
        entries = sorted(
            (("d " if c.is_dir() else "f ") + c.name) for c in target.iterdir()
        )
    except Exception as e:
        return _err(f"could not list {path!r}: {e}")
    return _ok("\n".join(entries) if entries else "(empty directory)")


# Name → async handler.  tool_dispatcher imports this and merges it into
# its Path-1 registry so these execute server-side BEFORE the daemon
# fallback.  Names match agent/client_tools.CLIENT_TOOL_SCHEMAS.
HANDLERS: dict[str, Callable] = {
    "read_file":      read_file,
    "write_file":     write_file,
    "edit_file":      edit_file,
    "list_directory": list_directory,
}


__all__ = ["HANDLERS", "read_file", "write_file", "edit_file", "list_directory"]
