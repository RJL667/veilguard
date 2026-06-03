"""W4-finish codemod: route the direct-`lancedb.connect()` ledger/proposal
modules through the backend-aware open_ledger_db() helper. Exact regex sub with
a per-file count report (skips files where it doesn't match)."""
import re, io, os
ROOT = r"C:\Users\rudol\Documents\veilguard\agent-runtime\app"
FILES = [
    "memory/lessons_reader.py",
    "proposals/constitution_amendments.py",
    "proposals/dream_scanner.py",
    "proposals/m4_backfill.py",
    "proposals/signal_emitters.py",
    "proposals/drift_watchdog.py",
    "proposals/recalibration.py",
    "proposals/lessons.py",
    "proposals/outcomes.py",
]
PAT = re.compile(r'^(\s*)db = lancedb\.connect\(([^)]*)\)[ \t]*$', re.M)
REPL = r'\1from ..ledger.store import open_ledger_db; db = open_ledger_db(\2)'
for f in FILES:
    p = os.path.join(ROOT, f)
    with io.open(p, encoding="utf-8") as fh:
        txt = fh.read()
    new, n = PAT.subn(REPL, txt)
    if n:
        with io.open(p, "w", encoding="utf-8", newline="") as fh:
            fh.write(new)
    print(f"  {f}: {n} replaced")
print("DONE")
