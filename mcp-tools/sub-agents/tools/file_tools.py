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

        Args:
            path: File path to write to
            content: Full file content
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
        """Search for files by glob pattern. Returns matching file paths.

        Args:
            pattern: Glob pattern (e.g. '**/*.py', 'src/**/*.ts')
            path: Directory to search in (default: project root)
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
        """Run a shell command on the host. Dangerous commands are blocked.

        Args:
            command: Shell command to execute (max 30s timeout)
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
