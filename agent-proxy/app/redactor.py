"""DEPRECATED shim — kept for one release.

[PII_UNIFY_2026_05_29]  There is now exactly ONE live redactor:
``veilguard.pii``.  The proxy-local Presidio engine + in-memory session
store that used to live here have been removed in favour of the shared,
fail-closed, Lance-backed, block-cached implementation that the
agent-runtime already uses.  This module only re-exports the shared
symbols so any straggling ``from app.redactor import ...`` keeps working.

New code MUST import from ``pii`` directly.  This file will be deleted
once no importer references it.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `from pii import ...` resolve from both local dev (repo root) and
# docker (/pii sibling of /app) — same bootstrap as main.py.
_HERE = Path(__file__).resolve()
for _cand in [*_HERE.parents, Path("/")]:
    if (_cand / "pii" / "__init__.py").is_file():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

from pii import (  # noqa: E402
    PIIRedactor,
    get_redactor,
    PII_ENTITIES,
    RedactionUnavailable,
)
from pii.redactor import USER_CONTENT_KEYS, SKIP_KEYS  # noqa: E402,F401

__all__ = [
    "PIIRedactor",
    "get_redactor",
    "PII_ENTITIES",
    "RedactionUnavailable",
    "USER_CONTENT_KEYS",
    "SKIP_KEYS",
]
