"""Utility tools — list_roles, list_skills, afk_status, agent_stats, system_health."""

import time
import httpx

from config import ROLES, SKILLS_DIR, SCRATCHPAD_DIR
from core import state
from core.state import TaskStatus
from utils.safety import TOOL_RISK
from utils.afk import is_afk, touch_activity


def register(mcp):
    @mcp.tool()
    async def list_roles() -> str:
        """List all available predefined agent roles and tool risk levels."""
        output = ["# Available Agent Roles\n"]
        for key, role in ROLES.items():
            output.append(f"## `{key}` — {role['name']}")
            output.append(f"{role['system'][:200]}...\n")
        output.append("\n## Tool Risk Levels\n")
        for level in ("LOW", "MEDIUM", "HIGH"):
            tools = [t for t, r in TOOL_RISK.items() if r == level]
            output.append(f"**{level}:** {', '.join(tools)}")
        return "\n".join(output)

    @mcp.tool()
    async def list_skills() -> str:
        """List available agent skills."""
        if not SKILLS_DIR.exists(): return "No skills directory found."
        files = sorted(SKILLS_DIR.glob("*.md"))
        if not files: return "No skills found."
        lines = ["# Available Agent Skills\n"]
        for f in files:
            first_line = f.read_text(encoding="utf-8").strip().split("\n")[0].strip("# ").strip()
            lines.append(f"- **{f.stem}**: {first_line} ({f.stat().st_size} chars)")
        lines.append(f'\nUsage: `spawn_agent(task, skill="{files[0].stem}", role="analyst")`')
        return "\n".join(lines)

    @mcp.tool()
    async def afk_status() -> str:
        """Check AFK mode status."""
        touch_activity()
        idle = time.time() - state.last_user_activity
        afk = is_afk()
        return "\n".join([
            f"**AFK Mode:** {'ACTIVE' if afk else 'INACTIVE'}",
            f"**Idle time:** {idle/60:.1f} min",
            f"**Active daemons:** {len([d for d in state.daemons.values() if d.enabled])}",
        ])

    @mcp.tool()
    async def agent_stats() -> str:
        """Show agent call statistics: calls, costs, errors, per-backend and per-role breakdown."""
        lines = ["# Agent Statistics\n"]
        lines.append(f"**Total calls:** {state.agent_stats['total_calls']}")
        lines.append(f"**Total errors:** {state.agent_stats['total_errors']}")
        lines.append(f"**Estimated cost:** ${state.estimated_cost_usd:.4f} USD")
        lines.append(f"**AFK mode:** {'ACTIVE' if is_afk() else 'inactive'}\n")

        lines.append("## By Backend\n")
        for name, stats in state.agent_stats["by_backend"].items():
            avg_ms = stats["total_ms"] / max(stats["calls"], 1)
            lines.append(f"- **{name}**: {stats['calls']} calls, {stats['errors']} err, avg {avg_ms:.0f}ms")

        if state.agent_stats["by_role"]:
            lines.append("\n## By Role\n")
            for role, stats in sorted(state.agent_stats["by_role"].items()):
                lines.append(f"- **{role}**: {stats['calls']} calls, {stats['errors']} errors")

        if state.call_log:
            lines.append(f"\n## Recent Calls (last {min(len(state.call_log), 10)})\n")
            for entry in state.call_log[-10:]:
                lines.append(f"- [{entry['time']}] {entry['backend']}/{entry['model']} role={entry['role']} {entry['elapsed_ms']}ms")

        lines.append(f"\n## Resources\n")
        lines.append(f"- Daemons: {len([d for d in state.daemons.values() if d.enabled])}")
        lines.append(f"- Running tasks: {len([t for t in state.tasks.values() if t.status == TaskStatus.RUNNING])}")
        lines.append(f"- Scratchpad: {len(list(SCRATCHPAD_DIR.glob('*.txt'))) if SCRATCHPAD_DIR.exists() else 0} files")
        return "\n".join(lines)

    @mcp.tool()
    async def system_health() -> str:
        """Check health of all Veilguard services."""
        services = [
            ("Sub-Agents (8809)", "http://localhost:8809/api/stats"),
            ("TCMM (8811)", "http://localhost:8811/health"),
            ("Host-Exec (8808)", "http://localhost:8808/sse"),
            ("Forge (8810)", "http://localhost:8810/sse"),
            ("PII Proxy (4000)", "http://localhost:4000/health"),
        ]
        lines = ["# Veilguard System Health\n"]
        async with httpx.AsyncClient(timeout=5) as client:
            for name, url in services:
                try:
                    resp = await client.get(url)
                    lines.append(f"- **{name}**: {'UP' if resp.status_code < 400 else 'DEGRADED'} ({resp.status_code})")
                except Exception:
                    lines.append(f"- **{name}**: DOWN")

        lines.append(f"\n**Calls:** {state.agent_stats['total_calls']} | **Cost:** ${state.estimated_cost_usd:.4f}")
        return "\n".join(lines)
