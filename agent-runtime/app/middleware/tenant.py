"""Per-request tenant context.

The Claude Agent SDK is tenant-blind by design — every `query()` call is
a fresh session with no notion of `cid`, `user_id`, or `tenant_id`.  We
own the routing entirely.  This module holds the per-request context
that every middleware + hook needs, threaded through `contextvars` so
async code naturally inherits the right values.

Why contextvars rather than function arguments:
  - The SDK's hooks (PreToolUse, PostToolUse, etc.) don't receive
    arbitrary user data; they get a fixed dict shape.  To know which
    tenant a hook fires for, the hook must read from contextvars.
  - Middleware that wraps `query()` also relies on contextvars so it
    doesn't have to thread tenant_id through every async call.
  - asyncio's contextvar propagation ensures concurrent requests don't
    cross-contaminate, even when one request fans out subagents in
    parallel.

Discipline:
  - Every entry point (the /agent/query route handler) MUST call
    `set_tenant_context(...)` at the start of the request and ensure
    it's reset on exit.  The TenantContext context-manager helper
    handles this idiom.
  - Hooks + middleware read via `current()`.  If `current()` returns
    None mid-execution, something has gone wrong with propagation;
    we log loudly and fail closed (the approval gate denies; the audit
    writes with NULL tenant — still useful for forensics).
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator, Optional


@dataclass(frozen=True)
class TenantContext:
    """All the identifiers a hook or middleware might need to know about
    the request currently in flight.

    Frozen so accidental mid-request mutation can't happen.  Fields:
      conversation_id  — the LibreChat cid; the cache memoization key
      user_id          — pii-proxy verified via x-user-id header
      tenant_id        — derived from user_id at proxy boundary
      agent_id         — which persona is driving this turn
      parent_cid       — for sub-calls; the parent conversation's cid
      is_background    — true if origin is a scheduled daemon, not chat
    """

    conversation_id: str
    user_id: str
    tenant_id: str
    agent_id: str
    parent_cid: Optional[str] = None
    is_background: bool = False


_current: contextvars.ContextVar[Optional[TenantContext]] = (
    contextvars.ContextVar("agent_runtime_tenant", default=None)
)


def current() -> Optional[TenantContext]:
    """Return the TenantContext for the in-flight request, or None.

    None means we're outside any request — typically a programming bug
    or a background task that didn't establish context.  Callers should
    fail closed (deny tool calls, write audit with tenant=unknown).
    """
    return _current.get()


def require() -> TenantContext:
    """Like `current()` but raises if context isn't set.

    Useful in code that should never run outside a request — failing
    loudly here is better than a downstream NoneType error.
    """
    ctx = _current.get()
    if ctx is None:
        raise RuntimeError(
            "agent-runtime: no TenantContext set; "
            "make sure /agent/query establishes context before dispatching"
        )
    return ctx


@contextmanager
def set_tenant_context(
    *,
    conversation_id: str,
    user_id: str,
    tenant_id: str,
    agent_id: str,
    parent_cid: Optional[str] = None,
    is_background: bool = False,
) -> Generator[TenantContext, None, None]:
    """Set the TenantContext for the duration of a with-block.

    Resets to the previous value on exit (so nested calls work cleanly).
    Use this at the top of /agent/query and at every sub-agent boundary
    where the agent_id should change (Director → Researcher etc.).
    """
    ctx = TenantContext(
        conversation_id=conversation_id,
        user_id=user_id,
        tenant_id=tenant_id,
        agent_id=agent_id,
        parent_cid=parent_cid,
        is_background=is_background,
    )
    token = _current.set(ctx)
    try:
        yield ctx
    finally:
        _current.reset(token)


def is_subagent_cid(cid: str) -> bool:
    """True if the cid is a sub-agent call (background-origin).

    Per spec §3.8: cids starting with `sub-` come from spawn_scope and
    indicate non-foreground origin.  The approval gate uses this to
    decide whether to gate client-daemon tool calls.

    Pattern: `sub-<parent7>-<kind3>-<uuid8>`.
    """
    return cid.startswith("sub-")


def parent_cid_from_sub(sub_cid: str) -> Optional[str]:
    """Extract the parent cid from a sub-agent cid, or None.

    `sub-abc1234-bgt-deadbeef` → `abc1234`.  Used by the cache-control
    middleware: sibling sub-calls memoize on (parent_cid, tcmm_version)
    so they share the rendered system prefix.
    """
    if not sub_cid.startswith("sub-"):
        return None
    parts = sub_cid.split("-", 3)
    if len(parts) < 4:
        return None
    return parts[1]


__all__ = [
    "TenantContext",
    "current",
    "require",
    "set_tenant_context",
    "is_subagent_cid",
    "parent_cid_from_sub",
]
