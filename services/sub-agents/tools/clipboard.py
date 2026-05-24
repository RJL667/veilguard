"""Clipboard tools — in-memory named data slots."""

from core import state


def register(mcp):
    @mcp.tool()
    async def clipboard_copy(name: str, content: str) -> str:
        """Copy content to a named clipboard slot. Use to pass data between tools."""
        state.clipboard[name] = content
        preview = content[:100] + "..." if len(content) > 100 else content
        return f"Copied to clipboard '{name}' ({len(content)} chars): {preview}"

    @mcp.tool()
    async def clipboard_paste(name: str) -> str:
        """Retrieve content from a named clipboard slot."""
        if name not in state.clipboard:
            available = ", ".join(state.clipboard.keys()) if state.clipboard else "(empty)"
            return f"Error: Clipboard '{name}' not found. Available: {available}"
        return state.clipboard[name]

    @mcp.tool()
    async def clipboard_list() -> str:
        """List all clipboard slots and their content preview."""
        if not state.clipboard:
            return "Clipboard is empty."
        lines = ["# Clipboard Slots\n"]
        for name, content in state.clipboard.items():
            preview = content[:80].replace("\n", " ") + "..." if len(content) > 80 else content.replace("\n", " ")
            lines.append(f"- **{name}** ({len(content)} chars): {preview}")
        return "\n".join(lines)
