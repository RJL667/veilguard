"""Pattern-match common tool errors → produce actionable suggestions.

Tools currently return bare error strings ("[Errno 2] No such file or
directory: 'foo'") that the LLM has to interpret. With a small pattern
library we can append concrete next-step suggestions:

  > [Errno 2] No such file or directory: 'foo'
  >
  > _Hint: try `list_directory(".")` to see what files exist, or check
  > the path is relative to the workspace (/workspace), not your local
  > Windows filesystem._

The library is intentionally small — bigger libraries become a
maintenance burden and risk over-triggering on unrelated text. Each
pattern carries (a) a regex/substring matcher and (b) a one-paragraph
hint. Tools call ``enrich_error(tool_name, raw_error)`` at the error
return site. If no pattern matches, the raw error is returned unchanged.

Usage:
    from error_hints import enrich_error

    try:
        ...
    except Exception as e:
        return enrich_error("read_pdf", str(e), context={"path": path})

Patterns can be tool-scoped or global. Tool-scoped takes precedence so a
filesystem tool can have a more specific suggestion than the generic
"file not found" hint.
"""

import re
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class _Pattern:
    matcher: re.Pattern
    hint: Callable[[re.Match, dict], str]
    # If non-empty, only fires for this set of tool names. Empty = global.
    tool_names: tuple = ()


# ── Global patterns (all tools) ─────────────────────────────────────────

_GLOBAL_PATTERNS: list[_Pattern] = [
    _Pattern(
        # POSIX: "[Errno 2] No such file or directory: '...'"
        # Windows: "[WinError 2] The system cannot find the file specified"
        matcher=re.compile(r"(no such file or directory|cannot find the (?:file|path) specified)", re.IGNORECASE),
        hint=lambda m, ctx: (
            "the path doesn't exist. Try `list_directory(\".\")` to see "
            "what's in the current workspace, or check the path is "
            "workspace-relative (e.g. `Documents/foo.pdf`) — paths like "
            "`C:\\Users\\...` from the user's Windows machine don't "
            "exist inside the server's container."
        ),
    ),
    _Pattern(
        matcher=re.compile(r"permission denied|errno 13|eacces", re.IGNORECASE),
        hint=lambda m, ctx: (
            "the OS refused the operation. Common causes: (a) the path "
            "isn't under the workspace root, (b) the path mixes drive "
            "letters with POSIX separators (e.g. `/workspace/C:\\...`), "
            "or (c) the parent directory doesn't exist yet. Verify the "
            "target path is workspace-relative and the parent dir exists."
        ),
    ),
    _Pattern(
        matcher=re.compile(r"workspace must be within|access denied: path", re.IGNORECASE),
        hint=lambda m, ctx: (
            "the path resolved outside the workspace root. Don't use "
            "`..` to escape, and use workspace-relative paths "
            "(e.g. `subdir/foo.txt`) rather than absolute ones — the "
            "container's filesystem is sandboxed."
        ),
    ),
    _Pattern(
        matcher=re.compile(r"timeout|timed out|deadline exceeded", re.IGNORECASE),
        hint=lambda m, ctx: (
            "the operation took too long. For network tools, the upstream "
            "may be slow or blocked; retry once before escalating. For "
            "background tasks, use `wait_for_tasks(...)` with no explicit "
            "timeout — the auto-pick layer adapts to recent history."
        ),
    ),
    _Pattern(
        matcher=re.compile(r"json(?:\.decoder)?\.jsondecode|invalid json|expecting (value|property name|',' delimiter)", re.IGNORECASE),
        hint=lambda m, ctx: (
            "the input wasn't valid JSON. Common gotchas: trailing "
            "commas, single-quoted strings, unescaped newlines. Pass a "
            "raw string and let the tool parse it, or run the input "
            "through `parse_json` first to surface the exact error line."
        ),
    ),
    _Pattern(
        matcher=re.compile(r"connection (?:refused|reset|aborted)|name or service not known|nodename nor servname", re.IGNORECASE),
        hint=lambda m, ctx: (
            "the upstream host couldn't be reached. Check the URL/host "
            "is correct; some MCP services run as bare host processes "
            "and use `host.docker.internal` from inside containers."
        ),
    ),
    _Pattern(
        matcher=re.compile(r"unauthori[sz]ed|401\b|invalid (?:api )?key|authentication failed", re.IGNORECASE),
        hint=lambda m, ctx: (
            "credentials are missing or wrong. Check the relevant env "
            "var on the server (e.g. ANTHROPIC_API_KEY, GOOGLE_CLOUD_PROJECT). "
            "Don't ask the user to paste a key into chat — direct them "
            "to set it server-side."
        ),
    ),
]


# ── Tool-scoped patterns (override globals when matched) ────────────────

_TOOL_PATTERNS: list[_Pattern] = [
    _Pattern(
        tool_names=("check_task", "get_result", "wait_for_tasks", "cancel_task"),
        matcher=re.compile(r"unknown task id|task .* not found", re.IGNORECASE),
        hint=lambda m, ctx: (
            "the task ID doesn't exist (or has been pruned). "
            "Use `list_tasks()` to see currently-tracked task IDs. "
            "Task IDs are short UUIDs (e.g. `abc12345`) that come back "
            "from `start_task`/`smart_task`/`start_parallel_tasks`."
        ),
    ),
    _Pattern(
        tool_names=("daemon_log", "daemon_wait", "stop_daemon"),
        matcher=re.compile(r"daemon .* not found", re.IGNORECASE),
        hint=lambda m, ctx: (
            "no daemon by that name. Use `list_daemons()` to see "
            "currently-running daemons. Names are slugified — "
            "`Inbox Watcher` becomes `inbox-watcher`."
        ),
    ),
    _Pattern(
        tool_names=("plan_get", "plan_update_step"),
        matcher=re.compile(r"plan .* not found", re.IGNORECASE),
        hint=lambda m, ctx: (
            "no plan by that ID. Use `plan_list()` to see recent plans. "
            "Plan IDs include a timestamp prefix (e.g. `1777381737_refactor_auth`) "
            "— plans older than 7 days are auto-pruned."
        ),
    ),
    _Pattern(
        tool_names=("notebook_edit", "notebook_read"),
        matcher=re.compile(r"not a notebook|\.ipynb required", re.IGNORECASE),
        hint=lambda m, ctx: (
            "this tool only operates on Jupyter `.ipynb` files. For "
            "plain-text JSON, use `read_file` / `parse_json`."
        ),
    ),
    _Pattern(
        tool_names=("read_xlsx", "edit_xlsx_cell", "edit_xlsx_range", "create_xlsx"),
        matcher=re.compile(r"badzipfile|invalid file|not a zip file", re.IGNORECASE),
        hint=lambda m, ctx: (
            "the file isn't a valid `.xlsx` (XLSX is a ZIP container). "
            "Possible causes: file is `.xls` (old Excel format — convert "
            "first), file is corrupt, or you're pointing at a non-Excel "
            "file. `read_file` can confirm what's actually there."
        ),
    ),
]


def enrich_error(tool_name: str, raw_error: str,
                  context: Optional[dict] = None) -> str:
    """Return ``raw_error`` plus a hint paragraph if any pattern matches.

    Tool-scoped patterns are checked first; the first match wins. If no
    pattern matches, the raw error is returned unchanged so we never
    obscure the original message.

    Args:
        tool_name: The MCP tool name. Used to select tool-scoped patterns.
        raw_error: The error string the tool would have returned.
        context: Optional dict of extra context (e.g. ``{"path": "..."}``).
            Hint generators can use it but most ignore it — included for
            future patterns that need it.
    """
    ctx = context or {}

    # Tool-scoped first.
    for p in _TOOL_PATTERNS:
        if tool_name not in p.tool_names:
            continue
        m = p.matcher.search(raw_error)
        if m:
            return f"{raw_error}\n\n_Hint: {p.hint(m, ctx)}_"

    # Then globals.
    for p in _GLOBAL_PATTERNS:
        m = p.matcher.search(raw_error)
        if m:
            return f"{raw_error}\n\n_Hint: {p.hint(m, ctx)}_"

    return raw_error


def list_known_patterns() -> list[str]:
    """Return a human-readable list of all (scope, pattern) pairs.

    Useful for an admin/debug endpoint to see what's covered.
    """
    out = []
    for p in _TOOL_PATTERNS:
        out.append(f"[{','.join(p.tool_names)}] {p.matcher.pattern}")
    for p in _GLOBAL_PATTERNS:
        out.append(f"[*] {p.matcher.pattern}")
    return out
