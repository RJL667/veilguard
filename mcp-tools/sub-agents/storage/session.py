"""Veilguard Sub-Agents: Session persistence."""

import json
import logging
import time

from config import SESSION_DIR, TRANSCRIPT_DIR
from core import state
from core.state import TaskStatus

logger = logging.getLogger("veilguard.storage")


def save_session():
    """Persist current state to disk."""
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    try:
        saved = {
            "stats": state.agent_stats,
            "cost": state.estimated_cost_usd,
            "schedules": state.schedules,
            "managed_tasks": {
                tid: {
                    "id": t.id, "title": t.title, "description": t.description[:200],
                    "status": t.status, "depends_on": t.depends_on,
                    "assigned_to": t.assigned_to,
                }
                for tid, t in state.managed_tasks.items()
            },
        }
        path = SESSION_DIR / "state.json"
        path.write_text(json.dumps(saved, default=str, indent=2), encoding="utf-8")
        logger.debug("Session saved")
    except Exception as e:
        logger.error(f"Session save failed: {e}")


def load_session():
    """Restore state from disk on startup."""
    path = SESSION_DIR / "state.json"
    if not path.exists():
        return
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        state.agent_stats.update(saved.get("stats", {}))
        state.estimated_cost_usd = float(saved.get("cost", 0))
        state.schedules.update(saved.get("schedules", {}))
        logger.info(f"Session restored: {state.agent_stats['total_calls']} calls, ${state.estimated_cost_usd:.4f}")
    except Exception as e:
        logger.error(f"Session load failed: {e}")


def log_transcript(agent_id: str, role: str, content: str):
    """Append an entry to the transcript log."""
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    path = TRANSCRIPT_DIR / f"{agent_id}.jsonl"
    entry = {"role": role, "agent": agent_id, "time": time.time(), "content": content[:2000]}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
