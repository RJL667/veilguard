"""Transcript tools — audit trail for agent conversations."""

import json
import time
from config import TRANSCRIPT_DIR


def register(mcp):
    @mcp.tool()
    async def transcript_list() -> str:
        """List all saved agent transcripts."""
        if not TRANSCRIPT_DIR.exists(): return "No transcripts."
        files = sorted(TRANSCRIPT_DIR.glob("*.jsonl"), key=lambda f: f.stat().st_mtime, reverse=True)
        if not files: return "No transcripts."
        lines = ["# Agent Transcripts\n"]
        for f in files[:20]:
            lines.append(f"- **{f.stem}** ({f.stat().st_size}B, {(time.time()-f.stat().st_mtime)/3600:.1f}h ago)")
        return "\n".join(lines)

    @mcp.tool()
    async def transcript_read(agent_id: str) -> str:
        """Read a saved agent transcript."""
        path = TRANSCRIPT_DIR / f"{agent_id}.jsonl"
        if not path.exists(): return f"Transcript '{agent_id}' not found"
        lines = [f"# Transcript: {agent_id}\n"]
        for raw in path.read_text(encoding="utf-8").strip().split("\n")[-20:]:
            try:
                entry = json.loads(raw)
                t = time.strftime("%H:%M:%S", time.localtime(entry.get("time", 0)))
                lines.append(f"**[{t}] {entry.get('role','?')}:** {entry.get('content','')[:300]}")
            except Exception:
                continue
        return "\n".join(lines)
