"""Auto-offload huge tool results to disk.

The problem: a single ``web_fetch`` of a long article or a
``transcript_read`` of a multi-megabyte session can dump 50KB-2MB of
text directly into the LLM's context window. That's a 10-30% bite out
of a single tool call's budget — and the LLM probably only needed a
few paragraphs.

This helper provides:
  - ``maybe_offload(result, tool_name)`` — pure function, called by a
    tool at its return site. If ``len(result)`` exceeds the threshold,
    writes the full text to disk and returns a summary + path. Otherwise
    returns the input unchanged.
  - ``@auto_offload(threshold=N, tool_name="...")`` — decorator form
    for tools whose entire return value should always be considered.

Offloaded results live at ``$VEILGUARD_OFFLOAD_DIR`` (default
``tcmm-data/offload/``) as ``.txt`` files. The path is included in the
summary so the LLM can subsequently call ``read_file`` to read a slice
if it needs more, without forcing the whole blob into context first.

Why this isn't auto-applied to *every* tool:
  - Some tool outputs are structured (JSON, tables) where mid-blob
    truncation breaks downstream parsing.
  - Some are intentionally tiny (status messages) — the threshold check
    is wasted overhead at scale.
  - Per-tool opt-in keeps the LLM's expectations stable: a tool that
    can offload always offloads above N; a tool that doesn't never does.
"""

import functools
import hashlib
import inspect
import logging
import os
import time
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("veilguard.result_offload")

# Default threshold — tuned for the typical agent context budget
# (~200KB total useful budget, 30 tool calls, so each call >6KB starts
# eating real budget). 50KB is the point where I'd rather pay an extra
# read_file round-trip than burn the whole budget on one fetch.
DEFAULT_THRESHOLD = 50_000

# Default storage location. Falls back to /tmp if no env hint and we
# can't find a sensible project root.
def _default_offload_dir() -> Path:
    env = os.environ.get("VEILGUARD_OFFLOAD_DIR", "").strip()
    if env:
        return Path(env)
    # Try to mirror the rest of the codebase by writing under tcmm-data/.
    # Walk up from this file until we find a tcmm-data/ sibling.
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        candidate = ancestor / "tcmm-data"
        if candidate.is_dir():
            return candidate / "offload"
    # Last resort — system temp.
    return Path(os.environ.get("TMPDIR", "/tmp")) / "veilguard-offload"


_OFFLOAD_DIR: Optional[Path] = None


def set_offload_dir(path: Path) -> None:
    """Override the offload directory (used by tests)."""
    global _OFFLOAD_DIR
    _OFFLOAD_DIR = Path(path)


def _resolve_dir() -> Path:
    if _OFFLOAD_DIR is not None:
        return _OFFLOAD_DIR
    return _default_offload_dir()


def _safe_tool_slug(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:40] or "tool"


def maybe_offload(result: str, tool_name: str = "tool",
                   threshold: int = DEFAULT_THRESHOLD,
                   preview_chars: int = 800) -> str:
    """If ``result`` exceeds ``threshold`` chars, write to disk and return a summary.

    The summary contains:
      - First ``preview_chars`` of the original output
      - Total char count
      - Absolute path of the offloaded file
      - A hint pointing at ``filesystem.read_file`` for retrieval

    Non-string inputs are returned unchanged. The threshold check is in
    chars, not bytes — close enough for the budget heuristic and avoids
    a redundant utf-8 encode pass on every call.
    """
    if not isinstance(result, str):
        return result
    if len(result) <= threshold:
        return result

    base_dir = _resolve_dir()
    try:
        base_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warning(f"Could not create offload dir {base_dir}: {e} — returning untruncated result")
        return result

    # Filename: <tool>_<short-hash>_<timestamp>.txt — short-hash so
    # repeated identical results don't all collide on the same file
    # (overwrite would lose history); timestamp keeps human-friendly
    # ordering when listing the dir.
    digest = hashlib.sha256(result.encode("utf-8", errors="replace")).hexdigest()[:10]
    ts = int(time.time())
    fname = f"{_safe_tool_slug(tool_name)}_{ts}_{digest}.txt"
    path = base_dir / fname
    try:
        path.write_text(result, encoding="utf-8")
    except OSError as e:
        logger.warning(f"Could not write offload file {path}: {e} — returning untruncated result")
        return result

    preview = result[:preview_chars].rstrip()
    return (
        f"{preview}\n\n"
        f"... [offloaded — full result is {len(result):,} chars, "
        f"first {preview_chars} shown above]\n\n"
        f"_Full output saved to_ `{path}`. To read more, use:\n"
        f"  - `read_file(\"{path}\", offset=N, limit=M)` for a slice, or\n"
        f"  - `grep(pattern, \"{path}\")` to find specific lines."
    )


def auto_offload(threshold: int = DEFAULT_THRESHOLD,
                  tool_name: Optional[str] = None,
                  preview_chars: int = 800) -> Callable:
    """Decorator: pipe the tool's return value through ``maybe_offload``.

    Args:
        threshold: Char count above which offload kicks in.
        tool_name: Override the tool slug used in offload filenames.
            Defaults to the wrapped function's ``__name__``.
        preview_chars: How many leading chars to keep inline as a
            preview before "..." and the offload pointer.
    """
    def decorator(fn: Callable) -> Callable:
        slug = tool_name or fn.__name__
        is_async = inspect.iscoroutinefunction(fn)

        if is_async:
            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                result = await fn(*args, **kwargs)
                return maybe_offload(result, slug, threshold, preview_chars)
            return async_wrapper

        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)
            return maybe_offload(result, slug, threshold, preview_chars)
        return sync_wrapper
    return decorator
