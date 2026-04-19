"""Veilguard Sub-Agents: Utility helpers."""

import json
import os
import re
import time
import uuid
from pathlib import Path

from config import OFFLOAD_DIR, OFFLOAD_THRESHOLD, SKILLS_DIR, PLAYBOOKS_DIR


def short_id() -> str:
    """Generate a short readable task ID."""
    return uuid.uuid4().hex[:8]


def elapsed(start: float, end: float | None = None) -> str:
    """Human-readable elapsed time."""
    secs = (end or time.time()) - start
    if secs < 60:
        return f"{secs:.1f}s"
    return f"{secs / 60:.1f}m"


def maybe_offload(result: str, task_id: str = "") -> str:
    """If result exceeds threshold, save to disk and return summary + path."""
    if len(result) <= OFFLOAD_THRESHOLD:
        return result
    OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)
    fname = task_id or uuid.uuid4().hex[:8]
    path = OFFLOAD_DIR / f"{fname}.md"
    path.write_text(result, encoding="utf-8")
    return result[:500] + f"\n\n... ({len(result)} chars total)\n\nFull result saved to: {path}"


def load_skill(name: str) -> str:
    """Load a skill markdown file by name."""
    safe = re.sub(r'[^\w\-.]', '_', name)
    path = SKILLS_DIR / f"{safe}.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def load_playbook(name: str) -> dict:
    """Load a playbook JSON file by name."""
    safe = re.sub(r'[^\w\-.]', '_', name)
    path = PLAYBOOKS_DIR / f"{safe}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}
