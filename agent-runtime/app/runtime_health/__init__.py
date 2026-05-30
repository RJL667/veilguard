"""Phase 6.3 + 6.4 + 6.5 + 6.7 — runtime-health helpers.

Pure-Python utilities the inbox-poller + tool wrappers use to enforce
the Phase 6 invariants.  Each sub-module is independently testable:

  heartbeats.py    — lease TTL + heartbeat detection (6.3)
  revision_lane.py — revision-first claim ordering    (6.4)
  truncation.py    — TRUNCATED marker on tool output  (6.5)
  apr.py           — Artifact Production Ratio + breaker (6.7)
"""

from .heartbeats import (
    HeartbeatRow,
    is_stale,
    select_orphans,
    DEFAULT_LEASE_TTL_S,
)
from .revision_lane import (
    sort_for_revision_priority,
    is_revision_candidate,
)
from .truncation import (
    truncate_with_marker,
    has_truncation_marker,
    TRUNCATION_MARKER_RE,
)
from .apr import (
    APRWindow,
    HEALTHY_LOW,
    HEALTHY_HIGH,
    CIRCUIT_BREAKER_FLOOR,
    CIRCUIT_BREAKER_WINDOW_S,
    GLOBAL_APR,
    apr_record_artifact,
    apr_record_tokens,
    apr_should_pause_dispatch,
    apr_resume,
    apr_snapshot,
)

__all__ = [
    # 6.3
    "HeartbeatRow", "is_stale", "select_orphans", "DEFAULT_LEASE_TTL_S",
    # 6.4
    "sort_for_revision_priority", "is_revision_candidate",
    # 6.5
    "truncate_with_marker", "has_truncation_marker", "TRUNCATION_MARKER_RE",
    # 6.7
    "APRWindow", "HEALTHY_LOW", "HEALTHY_HIGH",
    "CIRCUIT_BREAKER_FLOOR", "CIRCUIT_BREAKER_WINDOW_S",
    "GLOBAL_APR",
    "apr_record_artifact", "apr_record_tokens",
    "apr_should_pause_dispatch", "apr_resume", "apr_snapshot",
]
