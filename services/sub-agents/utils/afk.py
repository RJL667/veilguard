"""Veilguard Sub-Agents: AFK mode tracking."""

import time
from config import AFK_THRESHOLD
from core import state


def is_afk() -> bool:
    return (time.time() - state.last_user_activity) > AFK_THRESHOLD


def touch_activity():
    state.last_user_activity = time.time()
