"""Cost telemetry for MCP tools.

Same shape as the smart-timeout layer in
``mcp-tools/sub-agents/utils/tool_timing.py``, but tracking *cost*
rather than time. Tools record a per-call cost signal — bytes for
fetch tools, tokens for LLM-backed tools, wall-clock for compute —
and ``format_cost_hint`` turns that history into a one-liner the LLM
sees in the response:

  > _[cost] this fetch returned 47KB; recent similar URLs averaged
  > 8KB (n=12). Worth noting if you're working a budget._

The point is to give the LLM a signal it currently lacks. Today, every
``web_fetch`` is the same to it whether the page is 1KB or 10MB. With
this layer the model can make resource-aware decisions: if a URL
typically returns 2MB, maybe summarise the user's question into a more
specific search rather than dumping the whole page.

Storage: JSONL at ``tcmm-data/tool-cost/<tool_name>.jsonl``, one
record per call. Pruned to the last 200 records per (tool, key) on
read. Sharing-as-much-shape-as-possible-with-tool_timing means
operators only have to learn one log format.
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger("veilguard.tool_cost")

_DEFAULT_DIR: Optional[Path] = None
_MAX_HISTORY_PER_KEY = 200


def _resolve_dir() -> Optional[Path]:
    """Find the cost-data dir from env override, set_cost_dir, or sibling tcmm-data."""
    env = os.environ.get("VEILGUARD_TOOL_COST_DIR", "").strip()
    if env:
        return Path(env)
    if _DEFAULT_DIR is not None:
        return _DEFAULT_DIR
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        candidate = ancestor / "tcmm-data"
        if candidate.is_dir():
            return candidate / "tool-cost"
    return None


def set_cost_dir(path: Path) -> None:
    """Override the cost-data dir (used by tests)."""
    global _DEFAULT_DIR
    _DEFAULT_DIR = Path(path)


def _safe_key(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)[:60] or "default"


def _history_path(tool_name: str) -> Optional[Path]:
    base = _resolve_dir()
    if base is None:
        return None
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{_safe_key(tool_name)}.jsonl"


def _read_history(tool_name: str, key: str) -> list[dict]:
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
                if rec.get("key") == safe_k:
                    out.append(rec)
    except OSError:
        return []
    return out[-_MAX_HISTORY_PER_KEY:]


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = pct / 100 * (len(sorted_vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def record_cost(
    tool_name: str,
    key: str,
    *,
    bytes_out: Optional[int] = None,
    tokens: Optional[int] = None,
    wall_seconds: Optional[float] = None,
    extra: Optional[dict] = None,
) -> None:
    """Append one call's cost signals to the JSONL history.

    Pass whichever of (bytes_out, tokens, wall_seconds) the tool can
    measure — None means "not applicable" and is excluded from records.
    ``extra`` is a free-form dict for tool-specific signals (e.g. a web
    tool might pass ``{"status_code": 200}``).
    """
    path = _history_path(tool_name)
    if path is None:
        return
    rec = {"key": _safe_key(key), "ts": int(time.time())}
    if bytes_out is not None:
        rec["bytes_out"] = int(bytes_out)
    if tokens is not None:
        rec["tokens"] = int(tokens)
    if wall_seconds is not None:
        rec["wall_seconds"] = round(float(wall_seconds), 3)
    if extra:
        rec["extra"] = extra
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError as e:
        logger.warning(f"Could not write cost history for {tool_name}: {e}")


@dataclass
class _Stats:
    n: int
    p50: float
    p95: float


def _stats_for(history: list[dict], field: str) -> Optional[_Stats]:
    vals = sorted(float(r[field]) for r in history if field in r)
    if not vals:
        return None
    return _Stats(n=len(vals), p50=_percentile(vals, 50), p95=_percentile(vals, 95))


def _human_bytes(n: float) -> str:
    """Format byte count compactly: 1024 → '1KB', 1_500_000 → '1.4MB'."""
    if n < 1024:
        return f"{int(n)}B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f}KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f}MB"
    return f"{n / (1024 * 1024 * 1024):.2f}GB"


def format_cost_hint(
    tool_name: str,
    key: str,
    *,
    bytes_out: Optional[int] = None,
    tokens: Optional[int] = None,
    wall_seconds: Optional[float] = None,
) -> str:
    """Format a one-line cost hint from observed history.

    Pass the same cost signals you just passed to ``record_cost`` —
    this function compares THIS call against recent history for the
    same (tool, key) and produces:

      - First few calls: minimal hint, "system is learning"
      - In normal range: brief comparison line
      - Outside normal range: flag the divergence so the LLM notes it

    Returns "" if there's nothing meaningful to say (no measurable
    cost passed in, no history available).
    """
    history = _read_history(tool_name, key)

    parts: list[str] = []

    if bytes_out is not None:
        s = _stats_for(history, "bytes_out")
        if s and s.n >= 3:
            this = bytes_out
            if this > s.p95 * 1.5 or this < s.p50 * 0.3:
                parts.append(
                    f"this call: {_human_bytes(this)} — outside the usual "
                    f"range for this tool (p50={_human_bytes(s.p50)}, "
                    f"p95={_human_bytes(s.p95)}, n={s.n})"
                )
            else:
                parts.append(
                    f"{_human_bytes(this)} returned (typical: "
                    f"{_human_bytes(s.p50)} p50, {_human_bytes(s.p95)} p95, n={s.n})"
                )
        elif s:
            parts.append(f"{_human_bytes(this := bytes_out)} returned (n={s.n} samples — learning)")
        else:
            parts.append(f"{_human_bytes(bytes_out)} returned (no history yet)")

    if tokens is not None:
        s = _stats_for(history, "tokens")
        if s and s.n >= 3:
            parts.append(f"~{tokens} tokens (typical {int(s.p50)} p50, {int(s.p95)} p95)")
        else:
            parts.append(f"~{tokens} tokens (no history yet)")

    if wall_seconds is not None and (bytes_out is None and tokens is None):
        # Only surface wall-time if no other signal was recorded —
        # otherwise it's already in the timing layer's hints.
        parts.append(f"{wall_seconds:.1f}s wall")

    if not parts:
        return ""

    return f"\n\n_[cost] {' | '.join(parts)}._"
