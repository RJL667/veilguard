"""Phase 4 — document workflow versioning (§Phase 4).

Spec: "Document workflow.  Task outputs in workspace get versioning:
append-only `.history/` for files not under git, native git for files
in repo-tracked dirs.  Canonical paths only (no worktrees, per §3.5).
Conflict resolution = the Task whose `outputs[]` first listed the
path holds merge authority; peer Tasks open subtasks for changes."

This module:
  * detects repo-tracked vs non-repo paths
  * snapshots non-repo files into `.history/<rel>/<ts>.snap` before edit
  * stamps an audit entry on the agent_task (extras_json.document_history)
  * resolves merge authority via earliest-outputs lookup against
    agent_tasks
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


HISTORY_DIRNAME = ".history"


def _now() -> float:
    return time.time()


def _is_repo_tracked(path: Path) -> bool:
    """True if ``path`` lives inside a git work-tree.  Best-effort:
    falls back to False on any git error so non-repo callers don't
    crash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=path.parent if path.is_file() else path,
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0 and result.stdout.strip() == "true"
    except Exception:
        return False


def _git_workspace_root(path: Path) -> Optional[Path]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=path.parent if path.is_file() else path,
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return Path(result.stdout.strip())
    except Exception:
        pass
    return None


def snapshot_before_edit(
    target: str | Path,
    *,
    by_agent_id: str = "",
    note: str = "",
) -> dict[str, Any]:
    """Create a snapshot of ``target`` before an in-place edit.

    - If ``target`` is in a git work-tree: skip the .history snapshot
      (git already tracks it).  Just record the current HEAD hash in
      the audit entry.
    - Otherwise: copy the file into
      ``<target.parent>/.history/<target.name>/<ts>.snap``.
      Append-only — never overwrites prior snapshots.

    Returns an audit-record dict that callers can persist
    (typically into ``agent_tasks.extras_json.document_history``):

        {
          "target":            str (canonical path),
          "snapshot_path":     str | None,
          "git_head_sha":      str | None,
          "is_repo_tracked":   bool,
          "ts":                float,
          "by_agent_id":       str,
          "note":              str,
        }

    Missing target is OK — the snapshot is a no-op but the audit
    still records the "snapshot attempted" event with a None
    snapshot_path.  Callers can use this to mark "fresh file
    creation, no prior state to preserve".
    """
    p = Path(target).resolve()
    is_repo = _is_repo_tracked(p) if p.exists() else False
    head_sha: Optional[str] = None
    snap_path: Optional[Path] = None

    if is_repo:
        try:
            r = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=p.parent, capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                head_sha = r.stdout.strip()
        except Exception:
            pass
    elif p.is_file():
        try:
            hist_dir = p.parent / HISTORY_DIRNAME / p.name
            hist_dir.mkdir(parents=True, exist_ok=True)
            ts_str = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            snap_path = hist_dir / f"{ts_str}.snap"
            shutil.copy2(p, snap_path)
        except Exception as e:
            logger.warning(f"[documents] snapshot failed for {p}: {e}")
            snap_path = None

    return {
        "target":          str(p),
        "snapshot_path":   str(snap_path) if snap_path else None,
        "git_head_sha":    head_sha,
        "is_repo_tracked": bool(is_repo),
        "ts":              _now(),
        "by_agent_id":     by_agent_id,
        "note":            note,
    }


def list_history(target: str | Path) -> list[dict[str, Any]]:
    """List prior snapshots for a non-repo file, newest first.

    Empty list for repo-tracked files (use ``git log -- <path>``) or
    files with no `.history/<name>/` directory yet.
    """
    p = Path(target).resolve()
    if not p.parent.is_dir():
        return []
    hist_dir = p.parent / HISTORY_DIRNAME / p.name
    if not hist_dir.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for entry in hist_dir.iterdir():
        if not entry.is_file() or not entry.name.endswith(".snap"):
            continue
        try:
            st = entry.stat()
            out.append({
                "snapshot_path": str(entry),
                "size_bytes":    st.st_size,
                "ts":            st.st_mtime,
                "filename":      entry.name,
            })
        except Exception:
            continue
    out.sort(key=lambda r: r["ts"], reverse=True)
    return out


def merge_authority_holder(
    *,
    target_path: str,
    tenant_id: str,
    user_id: str,
) -> Optional[dict[str, Any]]:
    """Return the Task that holds merge authority for ``target_path``.

    Per spec: "the Task whose `outputs[]` first listed the path holds
    merge authority".  We scan `agent_tasks` for the earliest
    `created_ts` where `outputs` contains the path; that's the holder.

    Peer Tasks should open subtasks rooted at this holder for changes
    (Phase 4 dispatch logic — not yet enforced; this is the lookup
    helper that the enforcement will read from).
    """
    try:
        from .ledger.store import LedgerStore
        tbl = LedgerStore.get().table("agent_tasks")
        arr = (
            tbl.search()
            .where(
                f"tenant_id = '{tenant_id}' AND user_id = '{user_id}'"
            )
            .select(["id", "owner_id", "created_ts", "outputs", "status"])
            .limit(2000)
            .to_arrow()
        )
    except Exception as e:
        logger.warning(f"[documents] authority lookup failed: {e}")
        return None
    candidates = []
    for i in range(arr.num_rows):
        outs = arr.column("outputs")[i].as_py() or []
        if target_path in outs:
            candidates.append({
                "task_id":    arr.column("id")[i].as_py(),
                "owner_id":   arr.column("owner_id")[i].as_py(),
                "created_ts": arr.column("created_ts")[i].as_py(),
                "status":     arr.column("status")[i].as_py(),
            })
    if not candidates:
        return None
    candidates.sort(key=lambda r: r["created_ts"])
    return candidates[0]


__all__ = [
    "snapshot_before_edit",
    "list_history",
    "merge_authority_holder",
    "HISTORY_DIRNAME",
]
