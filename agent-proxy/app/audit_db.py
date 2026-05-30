"""LanceDB-backed PII audit log — thin re-export shim.

The actual writer + schema live at ``llm/audit_db.py`` (shared by
pii-proxy and agent-runtime).  This module exists for backward-
compatibility with any code that still does
``from app.audit_db import record`` or ``from app import audit_db``.

See ``llm/audit_db.py`` for the full implementation and rationale.
"""

from llm.audit_db import (
    SCHEMA as _SCHEMA,  # legacy private-name alias
    AuditDB,
    record,
)

# Some legacy callers reach for `_SCHEMA` (the original lower-case
# underscore-prefix name).  Keep the alias so they don't break.
__all__ = ["AuditDB", "record", "_SCHEMA"]
