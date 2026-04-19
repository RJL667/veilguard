"""Agent messaging tools — inter-agent communication."""

import time
from core import state


def register(mcp):
    @mcp.tool()
    async def agent_send(to: str, message: str, from_agent: str = "user") -> str:
        """Send a message to another agent's mailbox."""
        state.agent_mailbox.setdefault(to, []).append({"from": from_agent, "message": message, "time": time.strftime("%H:%M:%S")})
        return f"Message sent to `{to}` from `{from_agent}` ({len(message)} chars)"

    @mcp.tool()
    async def agent_inbox(agent_id: str) -> str:
        """Check an agent's inbox for messages."""
        msgs = state.agent_mailbox.get(agent_id, [])
        if not msgs: return f"Inbox for `{agent_id}` is empty."
        lines = [f"# Inbox for `{agent_id}` — {len(msgs)} messages\n"]
        for m in msgs:
            lines.append(f"**[{m['time']}] From {m['from']}:** {m['message']}")
        return "\n".join(lines)

    @mcp.tool()
    async def agent_broadcast(message: str, from_agent: str = "coordinator") -> str:
        """Broadcast a message to all agents with active mailboxes."""
        count = 0
        for agent_id in list(state.agent_mailbox.keys()):
            state.agent_mailbox[agent_id].append({"from": from_agent, "message": message, "time": time.strftime("%H:%M:%S")})
            count += 1
        return f"Broadcast to {count} agents from `{from_agent}`"
