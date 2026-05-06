"""Connector framework type contract.

Pure dataclasses, no behavior, no external imports beyond stdlib. This module
is the single source of truth for the data shapes that flow between
connectors, Veilguard's recall step, and TCMM.

Lifecycle of a connector candidate:

    Connector.hint(prompt) -> list[Snippet]            (recall-time, transient)
                          \\
                           -> survives fuse + ACL filter
                          /
    list[RecallHit]  <-- emitted to agent layer
                    -> rendered as shadow block in LLM context
                    -> LLM may call connector.read(ref)
                          |
                          v
    Connector.read(ref) -> Content
                       -> chunked, embedded (raw, no PII redaction at TCMM
                          boundary), upserted as TCMMEntries
                       -> next turn, the same content may surface as a
                          RecallHit with is_live_hint=False
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class Capability(str, Enum):
    """What a connector supports. Declared on each Connector subclass."""

    HINT = "hint"
    SEARCH = "search"
    READ = "read"
    LIST = "list"
    WRITE = "write"
    PERMISSIONS = "permissions"


@dataclass(frozen=True)
class Ref:
    """Opaque-to-Veilguard dereference token.

    The connector emits Refs inside hint snippets and accepts them in
    read/list/permissions calls. Veilguard never inspects ``args``; only the
    owning connector knows what they mean.
    """

    connector: str
    tool: str
    args: dict[str, Any]

    def to_tool_call(self) -> str:
        kw = ", ".join(f"{k}={v!r}" for k, v in self.args.items())
        return f"{self.connector}.{self.tool}({kw})"


@dataclass
class UserContext:
    """Per-request identity used to scope storage and filter ACLs.

    ``principals`` is the union of group/user IDs the user belongs to in the
    source system; recall filters entries where this set intersects
    ``entry.acl``.
    """

    tenant_id: str
    user_id: str
    principals: list[str] = field(default_factory=list)


@dataclass
class Snippet:
    """Recall-time hint candidate produced by Connector.hint().

    Never persisted as-is. May become a RecallHit if it survives fusion and
    the ACL filter; may seed TCMMEntries only after a successful read of the
    referenced resource.
    """

    content: str
    score: float
    ref: Ref
    acl: list[str]
    source: str
    title: str | None = None
    last_modified: datetime | None = None
    etag: str | None = None


@dataclass
class Content:
    """Result of Connector.read(). Raw text — TCMM stores raw, embeddings
    are computed over raw, redaction happens only at the LLM-egress boundary
    via the existing PII proxy."""

    text: str
    ref: Ref
    acl: list[str]
    title: str | None = None
    last_modified: datetime | None = None
    etag: str | None = None
    mime_type: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """One unit of Content after Connector.chunk(). Position is a stable
    ordinal inside the source resource so chunks can be re-ordered or
    grouped at recall time."""

    text: str
    position: int
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RecallHit:
    """Returned by the recall pipeline to the agent layer.

    ``is_live_hint`` distinguishes:
      * True  — candidate from Connector.hint() this turn (live, fresh ACL).
      * False — entry from TCMM storage (cached, ACL re-checked at recall).

    The shadow-block renderer uses ``tool_ref`` to emit the dereference
    instruction. ``tool_ref`` is None only for pure-TCMM memory entries.
    """

    content: str
    score: float
    source: str
    tool_ref: Ref | None
    title: str | None = None
    etag: str | None = None
    last_modified: datetime | None = None
    is_live_hint: bool = False
