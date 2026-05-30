"""Tests for documents.py — snapshot/history/authority helpers."""

import os
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from app import documents as D


@pytest.fixture
def tmp_text_file(tmp_path):
    p = tmp_path / "doc.md"
    p.write_text("original content\nline 2\n")
    return p


# ── snapshot_before_edit ───────────────────────────────────────────────


def test_snapshot_creates_history_file_for_non_repo_target(tmp_text_file):
    rec = D.snapshot_before_edit(
        tmp_text_file, by_agent_id="agent:test", note="pre-edit",
    )
    assert rec["is_repo_tracked"] is False
    assert rec["snapshot_path"] is not None
    snap = Path(rec["snapshot_path"])
    assert snap.exists()
    assert ".history" in str(snap)
    assert snap.read_text() == tmp_text_file.read_text()
    assert rec["by_agent_id"] == "agent:test"
    assert rec["note"] == "pre-edit"


def test_snapshot_records_git_head_for_repo_tracked(tmp_path):
    """Mock subprocess so we don't actually need a git repo."""
    target = tmp_path / "file.py"
    target.write_text("x = 1")
    with patch("app.documents.subprocess.run") as mock_run:
        # First call: is-inside-work-tree returns true
        # Second call: rev-parse HEAD returns a hex sha
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="true\n"),
            MagicMock(returncode=0, stdout="abc123def\n"),
        ]
        rec = D.snapshot_before_edit(target)
    assert rec["is_repo_tracked"] is True
    assert rec["git_head_sha"] == "abc123def"
    # repo-tracked → NO .history copy
    assert rec["snapshot_path"] is None


def test_snapshot_handles_missing_target(tmp_path):
    """Target doesn't exist → no snapshot, no crash."""
    rec = D.snapshot_before_edit(
        tmp_path / "does-not-exist.md", note="fresh-create",
    )
    assert rec["snapshot_path"] is None
    assert rec["is_repo_tracked"] is False
    assert rec["note"] == "fresh-create"


def test_snapshot_each_call_creates_new_file(tmp_text_file):
    """Append-only: two snapshots = two .history files."""
    r1 = D.snapshot_before_edit(tmp_text_file)
    time.sleep(1.1)   # ensure ts-string differs (second-resolution)
    tmp_text_file.write_text("edit 1")
    r2 = D.snapshot_before_edit(tmp_text_file)
    assert r1["snapshot_path"] != r2["snapshot_path"]
    assert Path(r1["snapshot_path"]).exists()
    assert Path(r2["snapshot_path"]).exists()


# ── list_history ───────────────────────────────────────────────────────


def test_list_history_returns_empty_for_no_history(tmp_text_file):
    assert D.list_history(tmp_text_file) == []


def test_list_history_sorts_newest_first(tmp_text_file):
    D.snapshot_before_edit(tmp_text_file)
    time.sleep(1.1)
    tmp_text_file.write_text("v2")
    D.snapshot_before_edit(tmp_text_file)
    hist = D.list_history(tmp_text_file)
    assert len(hist) == 2
    assert hist[0]["ts"] >= hist[1]["ts"]
    for h in hist:
        assert h["snapshot_path"].endswith(".snap")
        assert h["size_bytes"] > 0


def test_list_history_returns_empty_for_nonexistent_parent():
    assert D.list_history("/totally/fake/path/file.md") == []


# ── merge_authority_holder ─────────────────────────────────────────────


def test_authority_returns_earliest_task_listing_path():
    """Among tasks whose outputs contain target → earliest created_ts wins."""
    class FakeColumn:
        def __init__(self, vs): self._v = vs
        def __getitem__(self, i):
            class C:
                def __init__(self, v): self.v = v
                def as_py(self): return self.v
            return C(self._v[i])
    class FakeArrow:
        def __init__(self, rows):
            self._rows = rows
            self.num_rows = len(rows)
        def column(self, name):
            return FakeColumn([r.get(name) for r in self._rows])

    class FakeTable:
        def __init__(self, rows):
            self._rows = rows
        def search(self): return self
        def where(self, _e): return self
        def select(self, *_a, **_kw): return self
        def limit(self, _n): return self
        def to_arrow(self):
            return FakeArrow(self._rows)

    target = "team/drafts/popia-card.md"
    tbl = FakeTable([
        {"id": "task-A", "owner_id": "researcher",
         "created_ts": 100.0, "outputs": [target], "status": "done"},
        {"id": "task-B", "owner_id": "critic-prose",
         "created_ts": 50.0,  "outputs": [target], "status": "in_progress"},
        {"id": "task-C", "owner_id": "builder",
         "created_ts": 75.0,  "outputs": ["other.md"], "status": "open"},
    ])
    fake_store = MagicMock()
    fake_store.table.return_value = tbl
    with patch("app.documents.LedgerStore.get", return_value=fake_store) if False else patch.dict("os.environ", {}):
        # Use the real import path
        import importlib
        with patch("app.ledger.store.LedgerStore.get", return_value=fake_store):
            holder = D.merge_authority_holder(
                target_path=target, tenant_id="t", user_id="u",
            )
    assert holder is not None
    assert holder["task_id"] == "task-B"   # earliest created_ts
    assert holder["owner_id"] == "critic-prose"


def test_authority_returns_none_when_no_tasks_match():
    class FakeArrow:
        def __init__(self): self.num_rows = 0
        def column(self, _n):
            class C:
                def __getitem__(self, i):
                    raise IndexError
            return C()
    class FakeTable:
        def search(self): return self
        def where(self, _e): return self
        def select(self, *_a, **_kw): return self
        def limit(self, _n): return self
        def to_arrow(self): return FakeArrow()
    fake_store = MagicMock()
    fake_store.table.return_value = FakeTable()
    with patch("app.ledger.store.LedgerStore.get", return_value=fake_store):
        out = D.merge_authority_holder(
            target_path="x.md", tenant_id="t", user_id="u",
        )
    assert out is None
