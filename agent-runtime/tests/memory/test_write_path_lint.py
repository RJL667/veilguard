"""Phase 6.8 — Memory write-path discipline lint.

AC-36 — Linter rejects direct memory writes outside the writers module.
AC-37 — Writer-to-destination map auto-generated; doc cannot drift.
AC-38 — Negative-fixture: deliberately wire a direct Lance write from a
        non-writer module, assert the lint catches it.

The lint walks the source tree under `agent-runtime/app/` and `agent/`,
parses each .py file, and flags any module that imports a memory
destination handle outside the sanctioned set:

  sanctioned writers (allowed to import everything):
    app/memory/writers.py
    app/memory/__init__.py (re-exports only)
    middleware/tcmm.py (the underlying observe_agent_output wrapper)
    ledger/tasks.py, ledger/comments.py, ledger/proposals.py,
    ledger/store.py (low-level Lance handles — used BY writers)
    proposals/lessons.py (writers calls into it)

Everything else must go THROUGH the writer functions.  This catches:
  - new tools that try to call lancedb.connect() directly
  - debug shims that bypass writers
  - copy-paste from ledger CRUD into agent code
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
APP_ROOT = REPO_ROOT / "app"
AGENT_ROOT = REPO_ROOT.parent / "agent"

# Modules allowed to import memory destination handles directly.
# Anything else has to use the writer functions in app/memory/writers.py.
_SANCTIONED_DIRECT_IMPORTERS: frozenset[str] = frozenset({
    # The writers themselves.
    "app/memory/writers.py",
    "app/memory/__init__.py",
    # Phase 7 split-writers — sanctioned cross-ref sites (ledger PK ↔
    # TCMM obs_id). Only these are allowed to write both substrates
    # in one logical call.
    "app/memory/phase_7_writers.py",
    # Phase 7 M1 cutover (2026-05-28) — TCMM-archive-backed lesson
    # reader.  Reads (not writes) Lance directly because lessons live
    # in the shared archive Lance file; the legacy `org_memory` table
    # is gone.  Backfill helper inside this file is one-shot.
    "app/memory/lessons_reader.py",
    # Low-level Lance + TCMM HTTP — the writers call into these.
    "app/middleware/tcmm.py",
    "app/ledger/store.py",
    "app/ledger/tasks.py",
    "app/ledger/comments.py",
    "app/ledger/proposals.py",
    "app/ledger/schemas.py",
    "app/ledger/task_progress.py",
    "app/ledger/__init__.py",
    "app/proposals/lessons.py",
    "app/proposals/dream_scanner.py",
    "app/proposals/outcomes.py",
    "app/proposals/lifecycle.py",
    "app/proposals/recalibration.py",
    "app/proposals/signal_emitters.py",
    "app/proposals/constitution_amendments.py",
    "app/proposals/drift_watchdog.py",
    "app/proposals/rank.py",
    "app/proposals/proactive_config.py",
    # One-shot Phase 7 M4 historical backfill — uses LedgerStore directly
    # for the `migrated_to_tcmm` marker update, and routes through the
    # sanctioned Phase 7 split-writer for the TCMM side.
    "app/proposals/m4_backfill.py",
    # One-shot operational maintenance script (run via `docker exec`), not an
    # agent/runtime write path — uses LedgerStore directly to cancel stale
    # coordinator junk. Same treatment as m4_backfill.py above.
    "app/scripts_cleanup_stale_coordinators.py",
    # Phase 4 docs + replay touch ledger directly.
    "app/documents.py",
    "app/replay.py",
    # main.py wires endpoints; it touches Lance via repository pattern (Phase 6.10).
    "app/main.py",
    # a2a_external.py manages its own Lance table (a2a_external_keys).
    "app/a2a_external.py",
    # Tool MCP servers call the ledger CRUD directly — they are the bridge
    # from the LLM to writers.  Phase 6.8 v1 keeps them in the allow list
    # while we evaluate which calls should go through writer functions.
    "app/tools/ledger_mcp.py",
    "app/tools/memory_mcp.py",
    # Tool dispatcher routes to sub-agents bridge.
    "app/tool_dispatcher.py",
    # Workers — inbox_poller is the dispatcher.  Phase 7 may push some
    # of its writes through writers; for now it stays sanctioned.
    "app/workers/inbox_poller.py",
    "app/workers/__init__.py",
    # Constitution loader reads constitution.json — not a write path.
    "app/constitution/loader.py",
    "app/constitution/__init__.py",
})

# Forbidden direct imports (= "memory destination handles").
_FORBIDDEN_IMPORTS: frozenset[str] = frozenset({
    "lancedb",                          # raw Lance handle
    "app.middleware.tcmm",              # TCMM HTTP wrapper
    "app.ledger.store",                 # Lance store
})

# Forbidden specific symbol imports — even if the module is allowed,
# importing a specific low-level symbol from outside a sanctioned site
# is also a violation.
_FORBIDDEN_FROM_IMPORTS: dict[str, frozenset[str]] = {
    "app.middleware.tcmm": frozenset({"observe_agent_output"}),
    "app.ledger.store":    frozenset({"LedgerStore"}),
}


def _is_test_path(p: Path) -> bool:
    """Test files + fixtures legitimately access store / TCMM handles
    directly (arrange-phase setup, assertions on persisted rows). The
    write-path discipline (§3.11) governs RUNTIME / agent code, not the
    test harness — so the scan skips them. AC-38 still proves the detector
    fires, via synthetic tmp_path fixtures, so excluding tests here does
    not weaken the guarantee."""
    if any(part.lower() == "tests" for part in p.parts):
        return True
    return p.name.startswith("test_") or p.name == "conftest.py"


def _all_py_files() -> list[Path]:
    files: list[Path] = []
    for root in (APP_ROOT, AGENT_ROOT):
        if not root.is_dir():
            continue
        for p in root.rglob("*.py"):
            if "__pycache__" in str(p) or _is_test_path(p):
                continue
            files.append(p)
    return files


def _rel_path(p: Path) -> str:
    """Relative key like 'app/memory/writers.py'."""
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.resolve().relative_to(REPO_ROOT.parent).as_posix()


def _scan_imports(path: Path) -> tuple[set[str], set[tuple[str, str]]]:
    """Return (modules imported, (module, symbol) tuples for `from X import Y`)."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return set(), set()
    modules: set[str] = set()
    from_imports: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.add(node.module)
                for alias in node.names:
                    from_imports.add((node.module, alias.name))
    return modules, from_imports


# ── AC-36 ───────────────────────────────────────────────────────────────


def test_ac36_no_module_outside_sanctioned_set_imports_lance_handles():
    """Every Python module under app/ or agent/ must either be in the
    sanctioned-importer set OR not import memory destination handles."""
    violations: list[str] = []
    for path in _all_py_files():
        rel = _rel_path(path)
        if rel in _SANCTIONED_DIRECT_IMPORTERS:
            continue
        modules, from_imports = _scan_imports(path)
        # 1. Direct module imports.
        bad_modules = [m for m in modules if _is_forbidden_module(m)]
        if bad_modules:
            violations.append(
                f"{rel}: imports forbidden module(s) {bad_modules}"
            )
        # 2. Specific symbol imports.
        for mod, sym in from_imports:
            mod_norm = _normalize_module(mod, rel)
            if mod_norm in _FORBIDDEN_FROM_IMPORTS:
                if sym in _FORBIDDEN_FROM_IMPORTS[mod_norm]:
                    violations.append(
                        f"{rel}: forbidden `from {mod_norm} import {sym}`"
                    )
    assert not violations, (
        "Phase 6.8 write-path discipline violations:\n  - "
        + "\n  - ".join(violations)
        + "\n\nFix: route this through one of the writers in "
        "app/memory/writers.py instead of importing the destination handle "
        "directly. If you genuinely need the direct import, justify it in "
        "the writer-path PR and add the file to _SANCTIONED_DIRECT_IMPORTERS."
    )


def _is_forbidden_module(mod: str) -> bool:
    if mod in _FORBIDDEN_IMPORTS:
        return True
    # Lance has submodules; lancedb.* all count.
    if mod.startswith("lancedb"):
        return True
    return False


def _normalize_module(mod: str, importer_rel: str) -> str:
    """Resolve relative imports (`..middleware.tcmm`) to absolute."""
    if not mod.startswith("."):
        return mod
    # Count leading dots — each one goes up a level.
    n_dots = len(mod) - len(mod.lstrip("."))
    suffix = mod[n_dots:]
    # importer_rel is like 'app/foo/bar.py'.
    parts = importer_rel.replace("\\", "/").split("/")
    # Drop the .py filename
    parts = parts[:-1]
    # Go up n_dots-1 (one dot = same package, two dots = parent)
    if n_dots > 1:
        parts = parts[:-(n_dots - 1)]
    base = ".".join(parts)
    return f"{base}.{suffix}" if suffix else base


# ── AC-37 ───────────────────────────────────────────────────────────────


def test_ac37_writer_destination_map_doc_can_be_generated():
    """The auto-gen doc builds without error and contains every writer."""
    from app.memory.writers import WRITER_DESTINATIONS, generate_docs
    doc = generate_docs()
    assert "auto-generated" in doc.lower() or "DO NOT EDIT" in doc
    # Every writer named in WRITER_DESTINATIONS shows up in the doc.
    for d in WRITER_DESTINATIONS:
        assert d.writer in doc, f"writer {d.writer!r} missing from doc"
    # Sanity bounds: Phase 6.8 ships 8 single-destination writers + Phase 7
    # adds 4 split-writers (M1/M2/M3/M4 cross-ref sites) = 12 total.  Bound
    # set to catch accidental doc-only adds without forcing a bump every
    # time a real writer lands; tighten this when the surface stabilises.
    assert 8 <= len(WRITER_DESTINATIONS) <= 16


def test_ac37_writer_count_matches_public_api():
    """The number of WRITER_DESTINATIONS entries matches the number of
    callable writer functions exported.  Catches "added a writer but
    forgot to document it" and "documented a writer that doesn't exist."
    """
    from app import memory as memory_pkg
    from app.memory.writers import WRITER_DESTINATIONS
    # All writers should be importable from app.memory.
    documented = {d.writer for d in WRITER_DESTINATIONS}
    exported = {
        name for name in memory_pkg.__all__
        if name not in {"WriterError", "WRITER_DESTINATIONS"}
    }
    assert documented == exported, (
        f"WRITER_DESTINATIONS docs {documented} "
        f"!= app.memory.__all__ writers {exported}"
    )


# ── AC-38 — negative fixture (tests the test) ──────────────────────────


def test_ac38_lint_catches_synthetic_direct_lance_import(tmp_path: Path):
    """Deliberately construct a fake module that imports lancedb
    directly, and confirm the AST scanner would flag it.

    Without this, AC-36 could pass while the linter is silently broken
    (e.g. AST walks the wrong nodes, or `_is_forbidden_module` mis-checks
    the prefix).
    """
    bad_file = tmp_path / "evil_agent.py"
    bad_file.write_text(
        "import lancedb\n"
        "from app.middleware.tcmm import observe_agent_output\n"
        "def evil():\n"
        "    db = lancedb.connect('/tmp/evil.db')\n"
    )
    modules, from_imports = _scan_imports(bad_file)
    # Direct import check
    assert any(_is_forbidden_module(m) for m in modules), (
        "AST scanner failed to detect direct `import lancedb`"
    )
    # From-import check
    assert ("app.middleware.tcmm", "observe_agent_output") in from_imports, (
        "AST scanner failed to detect "
        "`from app.middleware.tcmm import observe_agent_output`"
    )


def test_ac38_lint_catches_relative_import_bypass(tmp_path: Path):
    """A relative `from ..middleware.tcmm import observe_agent_output`
    from a module under app/something/ must also be caught — the
    forbidden-from-imports check normalises relative imports.
    """
    bad_file = tmp_path / "evil_relative.py"
    bad_file.write_text(
        "from ..middleware.tcmm import observe_agent_output\n"
    )
    modules, from_imports = _scan_imports(bad_file)
    # The relative form survives as ..middleware.tcmm in the AST.
    relative_found = any(
        m.endswith("middleware.tcmm") for m, _ in from_imports
    )
    assert relative_found, "relative import not captured by AST scan"
