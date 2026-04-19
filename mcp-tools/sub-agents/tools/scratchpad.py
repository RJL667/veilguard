"""Scratchpad tools — persistent file-based data exchange.

These are for INTER-AGENT use (pipeline steps, coordinator workers).
NOT for storing user conversation data — TCMM handles that.
"""

import re
import time
from config import SCRATCHPAD_DIR


def register(mcp):
    @mcp.tool()
    async def scratchpad_write(name: str, content: str) -> str:
        """Write intermediate data to scratchpad for agent pipeline steps.

        ONLY use this inside pipelines or coordinator workflows to pass data between agents.
        Do NOT use for storing user information — TCMM memory handles that automatically.
        Do NOT use when the user is just chatting or providing information.
        """
        SCRATCHPAD_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r'[^\w\-.]', '_', name)
        path = SCRATCHPAD_DIR / f"{safe_name}.txt"
        path.write_text(content, encoding="utf-8")
        return f"Written {len(content)} chars to scratchpad `{safe_name}`"

    @mcp.tool()
    async def scratchpad_read(name: str) -> str:
        """Read data from the shared scratchpad."""
        safe_name = re.sub(r'[^\w\-.]', '_', name)
        path = SCRATCHPAD_DIR / f"{safe_name}.txt"
        if not path.exists():
            available = [f.stem for f in SCRATCHPAD_DIR.glob("*.txt")] if SCRATCHPAD_DIR.exists() else []
            return f"Error: Scratchpad `{safe_name}` not found. Available: {', '.join(available) or '(empty)'}"
        return path.read_text(encoding="utf-8")

    @mcp.tool()
    async def scratchpad_list() -> str:
        """List all files in the shared scratchpad. Auto-prunes files older than 24h."""
        if not SCRATCHPAD_DIR.exists(): return "Scratchpad is empty."
        now = time.time()
        lines = ["# Scratchpad Files\n"]
        for f in sorted(SCRATCHPAD_DIR.glob("*.txt")):
            age_hours = (now - f.stat().st_mtime) / 3600
            if age_hours > 24:
                f.unlink()
                continue
            preview = f.read_text(encoding="utf-8")[:80].replace("\n", " ")
            lines.append(f"- **{f.stem}** ({f.stat().st_size}B, {age_hours:.1f}h ago): {preview}...")
        return "\n".join(lines) if len(lines) > 1 else "Scratchpad is empty."
