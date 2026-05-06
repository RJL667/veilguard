"""Smart timeout / polling defaults for blocking tools.

The problem this solves: a tool like ``daemon_wait`` takes a
``timeout_seconds`` arg, and the LLM has no signal for what's right.
It guesses based on the docstring default (300s) and crosses its
fingers — sometimes too short (timeout fires for a daemon that runs
every 30min), sometimes too long (wastes a request slot for a job
that completes in 5s).

This module provides a three-layer default-picker so tools and the
LLM share the load:

  1. **Tool-derived default** — the tool itself knows context the LLM
     doesn't (e.g. ``daemon_wait`` knows the daemon's interval, so
     ``2 * interval`` is a smart default). The tool passes a callable
     to ``resolve_timeout`` that computes this.

  2. **History-tuned default** — every call records its actual
     duration to a JSONL log. Subsequent calls read the recent p95
     and prefer it over the tool-derived default once enough samples
     exist (≥3).

  3. **Explicit override** — if the LLM passed a value explicitly,
     that wins (subject to hard min/max clamps), but the response
     gets a hint comparing it against history.

Hints flow back to the LLM in the tool's response text — so within a
single conversation, the LLM learns "this daemon usually fires in
~600s" and adjusts subsequent calls without us having to teach it
out-of-band.

History is stored as JSONL at ``tcmm-data/tool-timing/<tool>.jsonl``,
one line per observation. Pruned to the last 200 records per (tool,
key) on read.
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

# Match the rest of the codebase — config.PROJECT_ROOT is the canonical
# anchor for tcmm-data/. We import lazily so this module is testable
# without dragging in the full sub-agents config (which pulls LLM creds).
try:
    from config import PROJECT_ROOT  # type: ignore
    _DEFAULT_DIR = Path(PROJECT_ROOT) / "tcmm-data" / "tool-timing"
except ImportError:
    _DEFAULT_DIR = None  # set via TIMING_DIR env or set_timing_dir()

logger = logging.getLogger("veilguard.tool_timing")

# Last 200 records per file is enough for stable p95 without
# accumulating forever. ~30KB per file at this cap.
_MAX_HISTORY_PER_KEY = 200

# Buffer on the p95 — real-world cadence has tail latency that a
# 95th-percentile estimator from 20 samples will under-shoot. 1.2x is
# arbitrary but consistently cheaper than the alternative (timing out
# a job that would have completed in p99).
_P95_BUFFER = 1.2


def _timing_dir() -> Optional[Path]:
    """Resolve the timing-data dir from env override or default."""
    env = os.environ.get("VEILGUARD_TOOL_TIMING_DIR", "").strip()
    if env:
        return Path(env)
    return _DEFAULT_DIR


def set_timing_dir(path: Path) -> None:
    """Override the timing-data dir (used by tests)."""
    global _DEFAULT_DIR
    _DEFAULT_DIR = Path(path)


@dataclass
class ResolvedTimeout:
    """The chosen timeout + provenance, for use by the calling tool."""
    timeout_seconds: int
    source: str            # "explicit" | "history" | "derived" | "fallback"
    history_size: int      # samples available for this (tool, key)
    history_p50: Optional[float] = None
    history_p95: Optional[float] = None
    derived_value: Optional[int] = None  # what the tool would have picked


def _safe_key(s: str) -> str:
    """Sanitise a key for use in a filename."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)[:60] or "default"


def _history_path(tool_name: str) -> Optional[Path]:
    """Return the JSONL file for a tool's history, or None if no dir is set."""
    base = _timing_dir()
    if base is None:
        return None
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{_safe_key(tool_name)}.jsonl"


def _read_history(tool_name: str, key: str) -> list[dict]:
    """Read the recent successful observations for one (tool, key) pair."""
    path = _history_path(tool_name)
    if path is None or not path.is_file():
        return []
    safe_k = _safe_key(key)
    out: list[dict] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("key") == safe_k and rec.get("success"):
                    out.append(rec)
    except OSError:
        return []
    # Keep only the most recent N — older samples are stale (system load,
    # daemon config, network conditions all drift).
    return out[-_MAX_HISTORY_PER_KEY:]


def _percentile(sorted_vals: list[float], pct: float) -> float:
    """Cheap linear-interp percentile. Assumes input is sorted ascending."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = pct / 100 * (len(sorted_vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def resolve_timeout(
    tool_name: str,
    key: str,
    explicit: Optional[int],
    derived_default: Callable[[], Optional[int]],
    hard_min: int = 1,
    hard_max: int = 1800,
    fallback: int = 300,
) -> ResolvedTimeout:
    """Pick a timeout using explicit > history > derived > fallback.

    Args:
        tool_name: Logical tool name (e.g. ``"daemon_wait"``). Used for the
            history filename — keep stable across releases.
        key: Sub-key within the tool (e.g. daemon name). Allows separate
            history per resource — a "fast" daemon and a "slow" daemon get
            independent p95s.
        explicit: The value the LLM passed. ``None`` or ``0`` means
            "auto-pick", anything else is treated as a deliberate override.
        derived_default: Callable returning the tool's own preferred
            default (e.g. ``lambda: 2 * daemon.interval_seconds``). Called
            only if no explicit value AND no history. May return ``None``
            to fall back to ``fallback``.
        hard_min: Floor (inclusive). Ignored values are clamped up.
        hard_max: Ceiling (inclusive). Ignored values are clamped down.
        fallback: Last-resort value if everything else fails.
    """
    history = _read_history(tool_name, key)
    durations = sorted(float(r["elapsed_s"]) for r in history if "elapsed_s" in r)
    p50 = _percentile(durations, 50) if durations else None
    p95 = _percentile(durations, 95) if durations else None

    # Compute what the derived default WOULD be, even if we don't use it
    # — the hint formatter wants this for context.
    try:
        derived_val = derived_default()
        if derived_val is not None:
            derived_val = int(derived_val)
    except Exception as e:
        logger.warning(f"derived_default for {tool_name}/{key} raised: {e}")
        derived_val = None

    def _clamp(v: int) -> int:
        return max(hard_min, min(int(v), hard_max))

    if explicit is not None and explicit > 0:
        return ResolvedTimeout(
            timeout_seconds=_clamp(explicit),
            source="explicit",
            history_size=len(history),
            history_p50=p50,
            history_p95=p95,
            derived_value=derived_val,
        )

    # Need ≥3 samples before history is statistically meaningful — below
    # that, single outliers dominate p95 and we'd thrash defaults.
    if p95 is not None and len(history) >= 3:
        return ResolvedTimeout(
            timeout_seconds=_clamp(int(p95 * _P95_BUFFER)),
            source="history",
            history_size=len(history),
            history_p50=p50,
            history_p95=p95,
            derived_value=derived_val,
        )

    if derived_val is not None:
        return ResolvedTimeout(
            timeout_seconds=_clamp(derived_val),
            source="derived",
            history_size=len(history),
            history_p50=p50,
            history_p95=p95,
            derived_value=derived_val,
        )

    return ResolvedTimeout(
        timeout_seconds=_clamp(fallback),
        source="fallback",
        history_size=len(history),
        history_p50=p50,
        history_p95=p95,
        derived_value=derived_val,
    )


def record_duration(
    tool_name: str,
    key: str,
    elapsed_s: float,
    success: bool,
) -> None:
    """Append one observation to the JSONL history.

    Failures (timeouts) are logged with success=false so they don't skew
    the p95 used for future defaults — but they ARE preserved so an
    operator can audit "is this tool timing out a lot?"
    """
    path = _history_path(tool_name)
    if path is None:
        return
    rec = {
        "key": _safe_key(key),
        "elapsed_s": round(float(elapsed_s), 2),
        "success": bool(success),
        "ts": int(time.time()),
    }
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError as e:
        logger.warning(f"Could not write timing history for {tool_name}: {e}")


def format_hint(
    resolved: ResolvedTimeout,
    actual_elapsed: Optional[float] = None,
    hit_timeout: bool = False,
    param_name: str = "timeout_seconds",
) -> str:
    """Format a one-line hint to append to the tool's response.

    The LLM reads this and adjusts subsequent calls. Three modes:
      - completed in time: shows where the actual landed vs. history
      - hit the timeout: recommends a higher value based on history
      - first-call-no-history: tells the LLM the system is learning

    Args:
        resolved: The ResolvedTimeout returned by resolve_timeout.
        actual_elapsed: Wall-clock duration if the wait completed. None
            if it timed out.
        hit_timeout: True if the wait expired without the event arriving.
        param_name: The kwarg name the tool actually accepts. Defaults to
            "timeout_seconds" (matches daemon_wait), but tools like
            wait_for_tasks use "timeout" — pass that here so the hint's
            recommended-call snippet is copy-pastable for the LLM.

    Returns "" if there's nothing meaningful to say (avoids noisy hints
    on every call).
    """
    if hit_timeout:
        # Recommend a higher value. If we have history, base it on p95;
        # otherwise just suggest doubling.
        if resolved.history_p95:
            suggest = int(resolved.history_p95 * _P95_BUFFER * 1.5)
            return (
                f"\n\n_[timing] No event within {resolved.timeout_seconds}s. "
                f"History (n={resolved.history_size}): p50={int(resolved.history_p50 or 0)}s, "
                f"p95={int(resolved.history_p95)}s. "
                f"Try `{param_name}={suggest}` next call._"
            )
        suggest = min(resolved.timeout_seconds * 2, 1800)
        return (
            f"\n\n_[timing] No event within {resolved.timeout_seconds}s and no history yet. "
            f"Try `{param_name}={suggest}` next call._"
        )

    if actual_elapsed is None:
        return ""

    # Completed. Worth showing the LLM where this run landed vs. history
    # — but only when there's enough data to draw a contrast worth seeing.
    if resolved.history_size < 3:
        # First couple of calls: just say "system is learning, defaults will
        # tune themselves" so the LLM doesn't have to second-guess.
        return (
            f"\n\n_[timing] Completed in {actual_elapsed:.1f}s "
            f"(source: {resolved.source}, n={resolved.history_size} samples — "
            f"defaults will auto-tune as more data lands)._"
        )

    # We have history. Flag drift if the actual landed far from p50.
    p50 = resolved.history_p50 or 0
    p95 = resolved.history_p95 or 0
    if p50 > 0 and (actual_elapsed > p95 * 1.5 or actual_elapsed < p50 * 0.3):
        return (
            f"\n\n_[timing] Completed in {actual_elapsed:.1f}s — outside the "
            f"usual range (p50={p50:.0f}s, p95={p95:.0f}s, n={resolved.history_size}). "
            f"Worth noting if this becomes a pattern._"
        )
    # Within normal range — minimal hint.
    return (
        f"\n\n_[timing] Completed in {actual_elapsed:.1f}s "
        f"(typical: {p50:.0f}s p50, {p95:.0f}s p95, n={resolved.history_size})._"
    )
