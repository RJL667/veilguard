"""Direct file/search/web tools — exposed as top-level MCP tools.

These let the LLM read files, search, edit, and browse directly
without needing to spawn a sub-agent. For single operations this
is faster and more natural than going through spawn_agentic.
"""

from core.agentic import handle_tool


def register(mcp):
    @mcp.tool()
    async def read_file(path: str, offset: int = 0, limit: int = 500) -> str:
        """Read a file from the host filesystem. Returns content with line numbers.

        Args:
            path: File path (absolute or relative to project root)
            offset: Start reading from this line number (0-indexed)
            limit: Max lines to read (default 500)
        """
        return await handle_tool("read_file", {"path": path, "offset": offset, "limit": limit})

    @mcp.tool()
    async def write_file(path: str, content: str) -> str:
        """Create or overwrite a file. OVERWRITES entire file — read first if modifying.

        BOTH `path` AND `content` are REQUIRED in every call. Calling
        with only `path` is a schema error — the tool runtime will reject
        it before any work is done. To create an empty file, pass
        `content=""` explicitly; never omit the field.

        Args:
            path: File path to write to. Required.
            content: Full file content. Required. Pass "" for an empty file.
                     Never omit this argument.
        """
        return await handle_tool("write_file", {"path": path, "content": content})

    @mcp.tool()
    async def edit_file(path: str, old_string: str, new_string: str) -> str:
        """Make a targeted edit to an existing file using exact string replacement.

        Args:
            path: File path to edit
            old_string: Exact string to find (must match exactly including whitespace)
            new_string: Replacement string
        """
        return await handle_tool("edit_file", {"path": path, "old_string": old_string, "new_string": new_string})

    @mcp.tool()
    async def search_files(pattern: str, path: str = "") -> str:
        """Find files by glob pattern under ``path``. Returns up to 50
        most-recently-modified matches.

        STRONG GUIDANCE: include a file extension or specific name in
        the pattern. Daemon caps scans at 5000 entries / 8s wall-clock,
        so a bare ``**/*`` on a large tree (e.g. a user home dir) will
        return capped results plus a truncation notice — not what you
        want. Good patterns:
          - ``**/*.py`` — every Python file
          - ``**/server.py`` — every file literally named server.py
          - ``src/**/*.ts`` — TypeScript under src/
          - ``.config*`` — dotfiles starting with .config
        Bad patterns:
          - ``**/*`` — too broad, triggers cap, useless on user home dir
          - ``*`` (with no path) — only the immediate dir, you probably
            wanted ``**`` recursion

        Skips noise dirs by default: ``.git``, ``node_modules``,
        ``__pycache__``, ``AppData``, ``.venv``, ``site-packages``,
        ``dist``, ``build``, ``Cache``, etc.

        Args:
            pattern: Glob pattern. Extension- or name-anchored beats
                     bare ``**/*``. Required.
            path: Directory to search in (absolute or relative to
                  project root). Defaults to project root.
        """
        return await handle_tool("search_files", {"pattern": pattern, "path": path})

    @mcp.tool()
    async def grep(pattern: str, path: str = "", include: str = "") -> str:
        """Search file contents for a regex pattern. Returns matching lines with file paths.

        Args:
            pattern: Regex pattern to search for
            path: Directory or file to search in (default: project root)
            include: File pattern to include (e.g. '*.py')
        """
        return await handle_tool("grep", {"pattern": pattern, "path": path, "include": include})

    @mcp.tool()
    async def run_command(command: str) -> str:
        """Run a shell command on the user's local machine via their client daemon.

        IMPORTANT: The shell is the user's native OS shell — could be
        Windows CMD, macOS zsh, or Linux bash. The first result includes
        a `[Host: ...]` hint identifying the platform; match your command
        syntax to that. If you haven't seen the hint yet, prefer
        platform-neutral commands or probe first:
          - Windows CMD: `cd` (current dir), `dir` (list), `echo %OS%`
          - Unix/macOS:  `pwd`, `ls -la`, `uname -a`
        Avoid Unix-only commands like `pwd`/`ls -la` on an unknown host —
        they fail on Windows. Dangerous commands are blocked. Max 30s.

        Args:
            command: Shell command to execute
        """
        return await handle_tool("run_command", {"command": command})

    @mcp.tool()
    async def web_search(query: str) -> str:
        """Search the web and return results.

        Args:
            query: Search query
        """
        return await handle_tool("web_search", {"query": query})

    @mcp.tool()
    async def web_fetch(url: str) -> str:
        """Fetch a URL and return its text content (max 3000 chars).

        Args:
            url: URL to fetch
        """
        return await handle_tool("web_fetch", {"url": url})
