"""Client tool schemas — daemon-routed tools whose impl lives in sub-agents.

When an LLM emits a tool_use for one of these, the agent-runtime
tool_dispatcher.Path 2 falls back to
  POST {SUB_AGENTS_URL}/api/agent_runtime/dispatch_tool
which then runs the approval gate and forwards over the WS bridge to
the user's daemon for execution on the host machine.

This module is the SINGLE SOURCE OF TRUTH for the schemas — both
`agent/base.py` (live mode, Agent.tools()) and
`agent-runtime/app/runtime.py` (scripted-or-other code paths) import
from here.  Keep names in sync with sub-agents/core/agentic.handle_tool's
dispatch switch.

If you add a new client tool to sub-agents, also add its schema here
and (optionally) reference it in the personas that should be allowed
to call it.  Personas that don't list a tool name don't see it in
their LLM call.
"""

from __future__ import annotations

CLIENT_TOOL_SCHEMAS: dict[str, dict] = {
    "read_file": {
        "name": "read_file",
        "description": (
            "Read a file from the user's host machine.  Returns content "
            "with line numbers in `<line_no>\\t<line>` format.  Reads at "
            "MOST `limit` lines starting at `offset` (default 500 lines "
            "from line 0 — sufficient for any sub-2000-line file in one "
            "call).  IMPORTANT: if the returned content is shorter than "
            "`limit` lines, **the file is fully read — the response is "
            "complete, not truncated**.  Do not re-call to 'get the rest' "
            "just because the response is short.  Re-call only if you "
            "explicitly want a different window (different offset or "
            "larger limit for a multi-thousand-line file).  Critics: a "
            "200-300 word deliverable is typically 30-50 lines; if you "
            "got 30-50 lines back, you have the full document."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path":   {"type": "string", "description": "Absolute or project-relative path."},
                "offset": {"type": "integer", "description": "Start line (0-indexed). Default 0."},
                "limit":  {"type": "integer", "description": "Max lines per call. Default 500 — almost always leave at default unless the file is >2000 lines."},
            },
            "required": ["path"],
        },
    },
    "write_file": {
        "name": "write_file",
        "description": (
            "Create or overwrite a file on the user's host machine.  "
            "OVERWRITES — read first if modifying.  BOTH `path` AND "
            "`content` are required (use content=\"\" for empty file)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path":    {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    "edit_file": {
        "name": "edit_file",
        "description": (
            "Targeted edit of an existing file via exact-string "
            "replace.  old_string must match exactly (including "
            "whitespace)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path":       {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
    "run_command": {
        "name": "run_command",
        "description": (
            "Run a shell command on the user's host machine.  Gated by "
            "an approval toast.  Pass `command` as a single string "
            "(NOT a list).  Returns stdout + stderr + exit code."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Full shell command line."},
            },
            "required": ["command"],
        },
    },
    "search_files": {
        "name": "search_files",
        "description": (
            "Glob-search files under `path`.  Returns up to 50 most-"
            "recently-modified matches.  Include a file extension in "
            "the pattern (e.g. **/*.py)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path":    {"type": "string"},
            },
            "required": ["pattern"],
        },
    },
    "grep": {
        "name": "grep",
        "description": (
            "ripgrep-style content search.  Returns matching files or "
            "lines depending on `output_mode`."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern":     {"type": "string"},
                "path":        {"type": "string"},
                "output_mode": {"type": "string", "enum": ["files_with_matches", "content", "count"]},
            },
            "required": ["pattern"],
        },
    },
    "web_search": {
        "name": "web_search",
        "description": (
            "Search the web for a query.  Returns top results as a list "
            "of {title, url, snippet}.  Use for open-ended research "
            "where you need to discover sources you don't already have "
            "URLs for.  Quote ALL claims you write from search results "
            "with the source URL."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        },
    },
    "web_fetch": {
        "name": "web_fetch",
        "description": (
            "Fetch a URL and return its text content (capped at ~3000 "
            "chars).  Use AFTER web_search to read a specific result, "
            "or when the user gives you a URL directly.  Strips HTML "
            "to readable text."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
            },
            "required": ["url"],
        },
    },
}


__all__ = ["CLIENT_TOOL_SCHEMAS"]
