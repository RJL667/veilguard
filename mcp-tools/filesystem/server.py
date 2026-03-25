"""Filesystem MCP Tool Server for Veilguard.

Tools: read_file, write_file, list_directory, search_files, grep
"""

import fnmatch
import os
import re
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("filesystem", instructions="File system tools for reading, writing, listing, searching, and grepping files.")

# Workspace root — configurable via env var, defaults to /workspace
WORKSPACE = os.environ.get("WORKSPACE_ROOT", "/workspace")


def _safe_path(path: str) -> Path:
    """Resolve path relative to workspace, prevent escape."""
    p = Path(path)
    if not p.is_absolute():
        p = Path(WORKSPACE) / p
    resolved = p.resolve()
    # Restrict to workspace — prevent directory traversal
    workspace_resolved = Path(WORKSPACE).resolve()
    if not str(resolved).startswith(str(workspace_resolved)):
        raise ValueError(f"Access denied: path must be within {WORKSPACE}")
    return resolved


@mcp.tool()
def read_file(path: str, offset: int = 0, limit: int = 500) -> str:
    """Read the contents of a file.

    Args:
        path: File path (absolute or relative to workspace)
        offset: Line number to start from (0-indexed)
        limit: Maximum number of lines to return (default 500)
    """
    p = _safe_path(path)
    if not p.exists():
        return f"Error: File not found: {p}"
    if not p.is_file():
        return f"Error: Not a file: {p}"

    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        total = len(lines)
        selected = lines[offset:offset + limit]
        numbered = [f"{i + offset + 1:>6}  {line}" for i, line in enumerate(selected)]
        header = f"# {p.name} ({total} lines total, showing {offset + 1}-{offset + len(selected)})\n"
        return header + "\n".join(numbered)
    except Exception as e:
        return f"Error reading file: {e}"


@mcp.tool()
def write_file(path: str, content: str) -> str:
    """Write content to a file. Creates parent directories if needed.

    Args:
        path: File path (absolute or relative to workspace)
        content: The content to write
    """
    p = _safe_path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Written {len(content)} bytes to {p}"
    except Exception as e:
        return f"Error writing file: {e}"


@mcp.tool()
def edit_file(path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    """Make a precise edit to a file by replacing an exact string match.

    Use this instead of write_file when you only need to change part of a file.
    The old_string must match exactly (including whitespace/indentation).

    Args:
        path: File path (absolute or relative to workspace)
        old_string: The exact text to find and replace. Must be unique in the file unless replace_all=True.
        new_string: The replacement text. Can be empty to delete the old_string.
        replace_all: If True, replace ALL occurrences. If False (default), old_string must be unique.
    """
    p = _safe_path(path)
    if not p.exists():
        return f"Error: File not found: {p}"

    try:
        content = p.read_text(encoding="utf-8", errors="replace")

        count = content.count(old_string)
        if count == 0:
            return f"Error: old_string not found in {p.name}. Read the file first to get exact text."

        if not replace_all and count > 1:
            return (
                f"Error: old_string found {count} times in {p.name}. "
                f"Provide more surrounding context to make it unique, or set replace_all=True."
            )

        new_content = content.replace(old_string, new_string) if replace_all else content.replace(old_string, new_string, 1)
        p.write_text(new_content, encoding="utf-8")

        replaced = count if replace_all else 1
        return f"Edited {p.name}: replaced {replaced} occurrence(s)"
    except Exception as e:
        return f"Error editing file: {e}"


@mcp.tool()
def edit_lines(path: str, start_line: int, end_line: int, new_content: str) -> str:
    """Replace a range of lines in a file. Lines are 1-indexed.

    Use after read_file to identify the exact line numbers to change.

    Args:
        path: File path (absolute or relative to workspace)
        start_line: First line to replace (1-indexed, inclusive)
        end_line: Last line to replace (1-indexed, inclusive)
        new_content: Replacement text for those lines. Use newlines for multiple lines.
    """
    p = _safe_path(path)
    if not p.exists():
        return f"Error: File not found: {p}"

    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        total = len(lines)

        if start_line < 1 or end_line < start_line or start_line > total:
            return f"Error: Invalid line range {start_line}-{end_line} (file has {total} lines)"

        # Clamp end_line
        end_line = min(end_line, total)

        # Replace lines
        new_lines = new_content.splitlines(keepends=True)
        # Ensure last line has newline
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines[-1] += "\n"

        result = lines[:start_line - 1] + new_lines + lines[end_line:]
        p.write_text("".join(result), encoding="utf-8")

        removed = end_line - start_line + 1
        added = len(new_lines)
        return f"Edited {p.name}: replaced lines {start_line}-{end_line} ({removed} lines → {added} lines)"
    except Exception as e:
        return f"Error editing lines: {e}"


@mcp.tool()
def insert_lines(path: str, after_line: int, content: str) -> str:
    """Insert new lines after a specific line number. Use after_line=0 to insert at the top.

    Args:
        path: File path (absolute or relative to workspace)
        after_line: Insert after this line number (0 = beginning of file)
        content: Text to insert. Use newlines for multiple lines.
    """
    p = _safe_path(path)
    if not p.exists():
        return f"Error: File not found: {p}"

    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)

        if after_line < 0 or after_line > len(lines):
            return f"Error: after_line {after_line} out of range (file has {len(lines)} lines)"

        new_lines = content.splitlines(keepends=True)
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines[-1] += "\n"

        result = lines[:after_line] + new_lines + lines[after_line:]
        p.write_text("".join(result), encoding="utf-8")

        return f"Inserted {len(new_lines)} lines after line {after_line} in {p.name}"
    except Exception as e:
        return f"Error inserting lines: {e}"


@mcp.tool()
def delete_lines(path: str, start_line: int, end_line: int) -> str:
    """Delete a range of lines from a file. Lines are 1-indexed.

    Args:
        path: File path (absolute or relative to workspace)
        start_line: First line to delete (1-indexed, inclusive)
        end_line: Last line to delete (1-indexed, inclusive)
    """
    p = _safe_path(path)
    if not p.exists():
        return f"Error: File not found: {p}"

    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        total = len(lines)

        if start_line < 1 or end_line < start_line or start_line > total:
            return f"Error: Invalid line range {start_line}-{end_line} (file has {total} lines)"

        end_line = min(end_line, total)
        deleted = end_line - start_line + 1
        result = lines[:start_line - 1] + lines[end_line:]
        p.write_text("".join(result), encoding="utf-8")

        return f"Deleted lines {start_line}-{end_line} ({deleted} lines) from {p.name}"
    except Exception as e:
        return f"Error deleting lines: {e}"


@mcp.tool()
def apply_diff(path: str, diff: str) -> str:
    """Apply a unified diff patch to a file.

    The diff format uses - for removed lines and + for added lines:
        @@ -start,count +start,count @@
        -old line
        +new line
         context line

    Args:
        path: File path (absolute or relative to workspace)
        diff: Unified diff text (only the hunks, no file headers needed)
    """
    p = _safe_path(path)
    if not p.exists():
        return f"Error: File not found: {p}"

    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        result = list(lines)

        # Parse hunks
        hunk_pattern = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
        offset = 0  # Track line number shift from previous hunks

        diff_lines = diff.splitlines()
        i = 0
        hunks_applied = 0

        while i < len(diff_lines):
            match = hunk_pattern.match(diff_lines[i])
            if not match:
                i += 1
                continue

            old_start = int(match.group(1)) - 1  # 0-indexed
            i += 1

            # Collect hunk lines
            removals = []
            additions = []
            context_before = 0
            in_prefix_context = True

            while i < len(diff_lines) and not diff_lines[i].startswith("@@"):
                line = diff_lines[i]
                if line.startswith("-"):
                    removals.append(line[1:])
                    in_prefix_context = False
                elif line.startswith("+"):
                    additions.append(line[1:])
                    in_prefix_context = False
                elif line.startswith(" ") or line == "":
                    if in_prefix_context:
                        context_before += 1
                i += 1

            # Apply: find and replace
            apply_at = old_start + context_before + offset
            if removals:
                # Verify removals match
                for j, rem in enumerate(removals):
                    idx = apply_at + j
                    if idx >= len(result) or result[idx].rstrip() != rem.rstrip():
                        return (
                            f"Error: Diff mismatch at line {idx + 1}. "
                            f"Expected: '{rem.rstrip()}', "
                            f"Got: '{result[idx].rstrip() if idx < len(result) else '(EOF)'}'"
                        )
                result[apply_at:apply_at + len(removals)] = additions
                offset += len(additions) - len(removals)
            else:
                # Pure addition
                for j, add in enumerate(additions):
                    result.insert(apply_at + j, add)
                offset += len(additions)

            hunks_applied += 1

        if hunks_applied == 0:
            return "Error: No valid diff hunks found. Use unified diff format: @@ -line,count +line,count @@"

        p.write_text("\n".join(result) + "\n", encoding="utf-8")
        return f"Applied {hunks_applied} diff hunk(s) to {p.name}"
    except Exception as e:
        return f"Error applying diff: {e}"


@mcp.tool()
def list_directory(path: str = ".", show_hidden: bool = False) -> str:
    """List files and directories.

    Args:
        path: Directory path (absolute or relative to workspace)
        show_hidden: Include hidden files (starting with .)
    """
    p = _safe_path(path)
    if not p.exists():
        return f"Error: Directory not found: {p}"
    if not p.is_dir():
        return f"Error: Not a directory: {p}"

    try:
        entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        lines = []
        for entry in entries:
            if not show_hidden and entry.name.startswith("."):
                continue
            if entry.is_dir():
                lines.append(f"  {entry.name}/")
            else:
                size = entry.stat().st_size
                if size < 1024:
                    size_str = f"{size}B"
                elif size < 1024 * 1024:
                    size_str = f"{size / 1024:.1f}KB"
                else:
                    size_str = f"{size / (1024 * 1024):.1f}MB"
                lines.append(f"  {entry.name}  ({size_str})")

        header = f"# {p} ({len(lines)} items)\n"
        return header + "\n".join(lines) if lines else header + "(empty)"
    except Exception as e:
        return f"Error listing directory: {e}"


@mcp.tool()
def search_files(pattern: str, path: str = ".", max_results: int = 50) -> str:
    """Search for files by name pattern (glob).

    Args:
        pattern: Glob pattern (e.g. '*.py', '**/*.json', 'test_*')
        path: Directory to search in
        max_results: Maximum results to return
    """
    p = _safe_path(path)
    if not p.exists():
        return f"Error: Directory not found: {p}"

    try:
        matches = []
        for match in p.rglob(pattern):
            rel = match.relative_to(p)
            matches.append(str(rel))
            if len(matches) >= max_results:
                break

        if not matches:
            return f"No files matching '{pattern}' in {p}"

        header = f"# Found {len(matches)} files matching '{pattern}'\n"
        return header + "\n".join(f"  {m}" for m in matches)
    except Exception as e:
        return f"Error searching: {e}"


@mcp.tool()
def grep(pattern: str, path: str = ".", file_pattern: str = "*", max_results: int = 50) -> str:
    """Search file contents using regex.

    Args:
        pattern: Regular expression to search for
        path: File or directory to search in
        file_pattern: Only search files matching this glob (e.g. '*.py')
        max_results: Maximum matching lines to return
    """
    p = _safe_path(path)
    if not p.exists():
        return f"Error: Path not found: {p}"

    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return f"Invalid regex: {e}"

    results = []

    def search_file(filepath: Path):
        try:
            text = filepath.read_text(encoding="utf-8", errors="replace")
            for i, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    rel = filepath.relative_to(p) if p.is_dir() else filepath.name
                    results.append(f"  {rel}:{i}: {line.rstrip()}")
                    if len(results) >= max_results:
                        return
        except Exception:
            pass

    if p.is_file():
        search_file(p)
    else:
        for filepath in sorted(p.rglob(file_pattern)):
            if filepath.is_file() and not any(part.startswith(".") for part in filepath.parts):
                search_file(filepath)
                if len(results) >= max_results:
                    break

    if not results:
        return f"No matches for '{pattern}'"

    header = f"# {len(results)} matches for '{pattern}'\n"
    return header + "\n".join(results)


# ── Data Tools ───────────────────────────────────────────────────────────────


@mcp.tool()
def parse_csv(path: str, delimiter: str = ",", max_rows: int = 200) -> str:
    """Read a CSV file and return as formatted text table.

    Args:
        path: CSV file path (absolute or relative to workspace)
        delimiter: Column delimiter (default comma)
        max_rows: Maximum rows to return (default 200)
    """
    import csv

    p = _safe_path(path)
    if not p.exists():
        return f"Error: File not found: {p}"

    try:
        with open(p, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f, delimiter=delimiter)
            rows = []
            for i, row in enumerate(reader):
                if i >= max_rows + 1:  # +1 for header
                    break
                rows.append(row)

        if not rows:
            return f"Empty CSV: {p}"

        # Calculate column widths
        widths = [0] * len(rows[0])
        for row in rows:
            for j, cell in enumerate(row):
                if j < len(widths):
                    widths[j] = max(widths[j], len(str(cell)))

        # Format output
        output = [f"# {p.name} ({len(rows) - 1} data rows)\n"]
        for i, row in enumerate(rows):
            line = " | ".join(str(cell).ljust(widths[j]) for j, cell in enumerate(row) if j < len(widths))
            output.append(line)
            if i == 0:
                output.append("-+-".join("-" * w for w in widths))

        if len(rows) > max_rows:
            output.append(f"\n... truncated at {max_rows} rows")

        return "\n".join(output)
    except Exception as e:
        return f"Error parsing CSV: {e}"


@mcp.tool()
def parse_json(path: str, max_depth: int = 5) -> str:
    """Read and pretty-print a JSON file.

    Args:
        path: JSON file path (absolute or relative to workspace)
        max_depth: Maximum nesting depth to display (default 5)
    """
    import json

    p = _safe_path(path)
    if not p.exists():
        return f"Error: File not found: {p}"

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        formatted = json.dumps(data, indent=2, ensure_ascii=False, default=str)

        # Truncate if too long
        lines = formatted.splitlines()
        if len(lines) > 500:
            formatted = "\n".join(lines[:500]) + f"\n\n... truncated ({len(lines)} total lines)"

        header = f"# {p.name}"
        if isinstance(data, list):
            header += f" (array, {len(data)} items)"
        elif isinstance(data, dict):
            header += f" (object, {len(data)} keys: {list(data.keys())[:10]})"

        return f"{header}\n\n{formatted}"
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON in {p}: {e}"
    except Exception as e:
        return f"Error parsing JSON: {e}"


@mcp.tool()
def transform_csv(path: str, operations: str, output_path: str = "") -> str:
    """Filter, sort, or select columns from a CSV file.

    Args:
        path: Input CSV file path
        operations: JSON string with operations:
            {"columns": ["name", "age"]} — select specific columns
            {"sort": "age"} — sort by column (prefix with - for descending: "-age")
            {"filter": "column:operator:value"} — filter rows. Operators: eq, ne, gt, lt, gte, lte, contains
            {"limit": 50} — limit output rows
            Can combine: {"columns": ["name"], "sort": "-age", "filter": "age:gt:30", "limit": 10}
        output_path: If given, write result to this file. Otherwise return as text.
    """
    import csv
    import json as json_mod

    p = _safe_path(path)
    if not p.exists():
        return f"Error: File not found: {p}"

    try:
        ops = json_mod.loads(operations)
    except json_mod.JSONDecodeError:
        return "Error: 'operations' must be valid JSON"

    try:
        with open(p, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            headers = reader.fieldnames or []

        if not rows:
            return f"Empty CSV: {p}"

        # Filter
        filter_str = ops.get("filter", "")
        if filter_str:
            parts = filter_str.split(":")
            if len(parts) == 3:
                col, op, val = parts
                filtered = []
                for row in rows:
                    cell = row.get(col, "")
                    try:
                        cell_num = float(cell)
                        val_num = float(val)
                        is_numeric = True
                    except (ValueError, TypeError):
                        cell_num = val_num = 0
                        is_numeric = False

                    match = False
                    if op == "eq":
                        match = cell == val
                    elif op == "ne":
                        match = cell != val
                    elif op == "contains":
                        match = val.lower() in cell.lower()
                    elif is_numeric:
                        if op == "gt":
                            match = cell_num > val_num
                        elif op == "lt":
                            match = cell_num < val_num
                        elif op == "gte":
                            match = cell_num >= val_num
                        elif op == "lte":
                            match = cell_num <= val_num
                    if match:
                        filtered.append(row)
                rows = filtered

        # Sort
        sort_col = ops.get("sort", "")
        if sort_col:
            descending = sort_col.startswith("-")
            col_name = sort_col.lstrip("-")
            try:
                rows.sort(key=lambda r: float(r.get(col_name, 0)), reverse=descending)
            except (ValueError, TypeError):
                rows.sort(key=lambda r: r.get(col_name, ""), reverse=descending)

        # Select columns
        selected_cols = ops.get("columns", headers)
        selected_cols = [c for c in selected_cols if c in headers]
        if not selected_cols:
            selected_cols = headers

        # Limit
        limit = ops.get("limit", len(rows))
        rows = rows[:limit]

        # Output
        if output_path:
            out = _safe_path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=selected_cols, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
            return f"Wrote {len(rows)} rows to {out}"
        else:
            lines = [" | ".join(selected_cols)]
            lines.append("-+-".join("-" * len(c) for c in selected_cols))
            for row in rows:
                lines.append(" | ".join(str(row.get(c, "")) for c in selected_cols))
            return f"# Result: {len(rows)} rows\n\n" + "\n".join(lines)

    except Exception as e:
        return f"Error transforming CSV: {e}"


@mcp.tool()
def json_query(path: str, expression: str) -> str:
    """Query a JSON file using JMESPath expressions.

    Args:
        path: JSON file path (absolute or relative to workspace)
        expression: JMESPath expression, e.g.:
            "people[?age > `30`].name" — filter and select
            "items[0:5]" — slice
            "length(items)" — count
            "sort_by(people, &age)" — sort
    """
    import json as json_mod
    import jmespath

    p = _safe_path(path)
    if not p.exists():
        return f"Error: File not found: {p}"

    try:
        data = json_mod.loads(p.read_text(encoding="utf-8"))
        result = jmespath.search(expression, data)
        formatted = json_mod.dumps(result, indent=2, ensure_ascii=False, default=str)

        lines = formatted.splitlines()
        if len(lines) > 500:
            formatted = "\n".join(lines[:500]) + f"\n\n... truncated ({len(lines)} total lines)"

        return f"# Query: {expression}\n\n{formatted}"
    except jmespath.exceptions.ParseError as e:
        return f"Error: Invalid JMESPath expression: {e}"
    except Exception as e:
        return f"Error querying JSON: {e}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
