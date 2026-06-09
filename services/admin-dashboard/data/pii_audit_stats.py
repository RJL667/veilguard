"""Stats over the PII proxy audit log (LanceDB ``pii_audit`` table).

Schema (see pii-proxy/app/audit_db.py):
  aid (int64)
  user_id (str)
  conversation_id (str)
  direction (str: "TO_LLM" | "FROM_LLM")
  model (str)
  stream (bool)
  content (str)
  created_at (float64)
  tokens_input / tokens_output / cache_create / cache_read (int64)
  extra (str, optional JSON)

The dashboard never displays the ``content`` column verbatim; it only
counts redaction tokens within it. That keeps the UI a stats view, not
a content viewer.

# Time window contract
Every public function takes ``start_ts`` / ``end_ts`` (Unix seconds).
``end_ts=None`` means "now". Convenience: ``window_hours`` is still
accepted as a fallback when no explicit range is provided, so existing
callers keep working.

# Empty-state contract
Each function returns a fully-shaped result even when the table is
missing (fresh dev environment) or the query window is empty, so the
frontend never sees ``undefined`` keys.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import Counter
from typing import Any, Optional

logger = logging.getLogger("veilguard.admin.pii_audit")


LANCE_DIR = os.environ.get(
    "ADMIN_LANCE_DIR",
    "/home/rudol/veilguard/tcmm-data/veilguard/tcmm.db",
)
TOK_RE = re.compile(
    r"REF_(?:PERSON|EMAIL|PHONE|IP|LOCATION|URL|CREDIT_CARD|ID|"
    r"IBAN_CODE|IBAN|ORG|DATE|API_KEY|CARD|BANK_ACCOUNT|SA_ID|SA_PHONE)_\d+"
)
KIND_RE = re.compile(r"REF_([A-Z_]+?)_\d+")


_AUDIT_BACKEND = os.environ.get(
    "VEILGUARD_AUDIT_BACKEND", os.environ.get("TCMM_STORAGE", "postgres")
).lower()
_PG = _AUDIT_BACKEND in ("postgres", "postgresql")
_AUDIT_DSN = os.environ.get(
    "VEILGUARD_AUDIT_DATABASE_URL",
    os.environ.get("TCMM_DATABASE_URL", "postgresql://tcmm:tcmm@localhost:5432/tcmm"),
)


def _open_table():
    import lancedb  # lazy: only the Lance backend needs it
    db = lancedb.connect(LANCE_DIR)
    return db.open_table("pii_audit")


def _pg_table(where_sql: str, params: list, cols: list):
    """Run a pii_audit query on Postgres and return a REAL pyarrow.Table with
    ALL `cols` present (even when empty), so every downstream aggregator's
    `.column(name).to_pylist()` / `.num_rows` / `.slice()` works unchanged."""
    import psycopg2
    import pyarrow as pa
    sel = ", ".join(cols)
    conn = psycopg2.connect(_AUDIT_DSN)
    conn.autocommit = True   # read-only stats: don't sit in an open transaction
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT {sel} FROM pii_audit WHERE {where_sql}", params)
            rows = cur.fetchall()
    finally:
        conn.close()
    return pa.table({c: [r[i] for r in rows] for i, c in enumerate(cols)})


def _resolve_window(
    start_ts: Optional[float],
    end_ts: Optional[float],
    window_hours: Optional[int],
) -> tuple[float, float]:
    """Normalize the three ways callers ask for a time window."""
    now = time.time()
    if end_ts is None:
        end_ts = now
    if start_ts is None:
        # Fall back to window_hours; default 24h
        h = window_hours if window_hours is not None else 24
        start_ts = end_ts - h * 3600
    return float(start_ts), float(end_ts)


# ── 2026-05-19: cache the windowed-rows result for ~8s ─────────────
# The admin dashboard auto-refreshes every 10s and fires 4 endpoints
# in parallel (overview, recent_redactions, messages_per_user,
# cache_overview) that each called _windowed_rows() → t.to_arrow() →
# full pii_audit scan. 4× redundant scans every refresh.
#
# Cache key is (start_ts, end_ts, user_id) — identical across the
# parallel calls within one refresh, so all 4 share ONE scan. TTL is
# slightly less than the dashboard refresh interval so freshness
# stays acceptable while load drops 4×.
import time as _t_cache

_WINDOWED_CACHE: dict = {}
_WINDOWED_CACHE_TTL = 8.0
_WINDOWED_CACHE_MAX_KEYS = 16


def _windowed_rows(
    start_ts: Optional[float] = None,
    end_ts: Optional[float] = None,
    window_hours: Optional[int] = None,
    user_id: Optional[str] = None,
):
    """Open the table and apply window + optional user filter.

    Returns ``(start_ts, end_ts, sub)`` where ``sub`` is a pyarrow Table
    or ``None`` when the table is missing.

    Cached for ``_WINDOWED_CACHE_TTL`` seconds per (s, e, user_id) key —
    the dashboard fires 4 parallel callers per refresh that all share
    the same window; first wins the scan, others reuse.
    """
    # Cache key uses the INPUT parameters (rounded) — NOT the
    # resolved (s, e) timestamps. _resolve_window computes its own
    # "now" each call, so window_hours=24 drifts s by milliseconds
    # every call and misses the cache. Rounding explicit timestamps
    # to the nearest 5s prevents the same issue for custom windows.
    def _round5(x):
        return None if x is None else int(round(float(x) / 5) * 5)
    key = (
        _round5(start_ts),
        _round5(end_ts),
        window_hours,
        user_id or "",
    )
    now = _t_cache.monotonic()
    cached = _WINDOWED_CACHE.get(key)
    s = e = None  # filled either from cache or from _resolve_window below
    if cached is not None and (now - cached[0]) < _WINDOWED_CACHE_TTL:
        return cached[1]

    # Cache miss — resolve the window now and do the actual scan.
    s, e = _resolve_window(start_ts, end_ts, window_hours)
    # [LEAK_FIX_2026_06_01] Push the time-window + user filter AND column
    # projection INTO Lance (DataFusion) so it materialises only the windowed
    # rows + the columns the dashboard reads — instead of to_arrow()-ing the
    # ENTIRE audit history (every prompt's full `content`) into RAM on every
    # ~10s refresh, copying it through a pyarrow mask, and caching it. That
    # full-history load — repeated and retained — is what pinned admin at ~3GB.
    _COLS = [
        "aid", "created_at", "direction", "user_id", "conversation_id",
        "model", "tokens_input", "tokens_output", "cache_create",
        "cache_read", "extra", "content",
    ]
    try:
        if _PG:
            _where = "created_at >= %s AND created_at < %s"
            _params: list = [float(s), float(e)]
            if user_id:
                _where += " AND user_id = %s"
                _params.append(str(user_id))
            arr = _pg_table(_where, _params, _COLS)
        else:
            t = _open_table()
            _where = f"created_at >= {float(s)} AND created_at < {float(e)}"
            if user_id:
                _u = str(user_id).replace("'", "''")
                _where += f" AND user_id = '{_u}'"
            # search() with no vector = plain filtered scan; .where() + .select()
            # push predicate + projection to the engine (no Python-side mask copy).
            arr = t.search().where(_where).select(_COLS).to_arrow()
    except Exception as ex:
        logger.info("pii_audit windowed scan failed: %s", ex)
        return s, e, None

    result = (s, e, arr)

    _WINDOWED_CACHE[key] = (now, result)
    # Evict oldest if we've accumulated too many distinct keys
    # (custom windows + per-user filters can balloon otherwise).
    if len(_WINDOWED_CACHE) > _WINDOWED_CACHE_MAX_KEYS:
        oldest_key = min(
            _WINDOWED_CACHE.keys(),
            key=lambda k: _WINDOWED_CACHE[k][0],
        )
        del _WINDOWED_CACHE[oldest_key]

    return result


def overview(
    window_hours: Optional[int] = 24,
    *,
    start_ts: Optional[float] = None,
    end_ts: Optional[float] = None,
    user_id: Optional[str] = None,
) -> dict[str, Any]:
    """Aggregated stats over a window, optionally scoped to one user.

    Returns:
      total_rows, by_direction, tokens_by_kind, conversations_seen,
      users_seen, time_series_hourly, start_ts, end_ts.
    """
    empty = {
        "start_ts": 0.0,
        "end_ts": 0.0,
        "window_hours": window_hours,
        "total_rows": 0,
        "by_direction": {},
        "tokens_by_kind": {},
        "conversations_seen": 0,
        "users_seen": 0,
        "time_series_hourly": [],
        "user_id": user_id,
    }
    s, e, sub = _windowed_rows(start_ts, end_ts, window_hours, user_id)
    if sub is None:
        return {**empty, "start_ts": s, "end_ts": e,
                "error": "pii_audit table not found"}

    dirs = sub.column("direction").to_pylist()
    contents = sub.column("content").to_pylist()
    convs = sub.column("conversation_id").to_pylist()
    users = sub.column("user_id").to_pylist()
    timestamps = sub.column("created_at").to_pylist()

    tok_kinds: Counter[str] = Counter()
    for d, c in zip(dirs, contents):
        if d != "TO_LLM" or not c:
            continue
        for m in KIND_RE.finditer(c):
            tok_kinds[m.group(1)] += 1

    # Bucket size scales with window so we always get ~30-60 buckets.
    span = max(e - s, 1)
    bucket_size = _pick_bucket_size(span)
    buckets: Counter[int] = Counter()
    for ts in timestamps:
        b = int(ts // bucket_size) * bucket_size
        buckets[b] += 1
    time_series = [
        {"ts": int(b), "count": n} for b, n in sorted(buckets.items())
    ]

    return {
        "start_ts": s,
        "end_ts": e,
        "window_hours": window_hours,
        "bucket_seconds": bucket_size,
        "total_rows": sub.num_rows,
        "by_direction": dict(Counter(dirs)),
        "tokens_by_kind": dict(tok_kinds),
        "conversations_seen": len({c for c in convs if c}),
        "users_seen": len({u for u in users if u}),
        "time_series_hourly": time_series,
        "user_id": user_id,
    }


def _pick_bucket_size(span_seconds: float) -> int:
    """Pick a bucket size aiming for 30-60 buckets across the window."""
    target = span_seconds / 48
    candidates = [
        60,            # 1 minute
        5 * 60,        # 5 min
        15 * 60,       # 15 min
        30 * 60,       # 30 min
        60 * 60,       # 1 hour
        3 * 3600,      # 3 hour
        6 * 3600,      # 6 hour
        12 * 3600,     # 12 hour
        24 * 3600,     # 1 day
        7 * 86400,     # 1 week
    ]
    for c in candidates:
        if c >= target:
            return c
    return candidates[-1]


def per_user(
    window_hours: Optional[int] = 24,
    top_n: int = 20,
    *,
    start_ts: Optional[float] = None,
    end_ts: Optional[float] = None,
) -> list[dict[str, Any]]:
    """Per-user redaction + message breakdown over the window.

    Aggregates across ALL users; the ``user_id`` filter is intentionally
    not exposed here since this view exists to compare users.
    """
    s, e, sub = _windowed_rows(start_ts, end_ts, window_hours, None)
    if sub is None or sub.num_rows == 0:
        return []

    users = sub.column("user_id").to_pylist()
    dirs = sub.column("direction").to_pylist()
    contents = sub.column("content").to_pylist()
    convs = sub.column("conversation_id").to_pylist()
    timestamps = sub.column("created_at").to_pylist()
    tokens_in = sub.column("tokens_input").to_pylist()
    tokens_out = sub.column("tokens_output").to_pylist()
    extras = sub.column("extra").to_pylist()

    by_user: dict[str, dict[str, Any]] = {}
    for uid, d, c, conv, ts, ti, to, ex_raw in zip(
        users, dirs, contents, convs, timestamps, tokens_in, tokens_out, extras
    ):
        if not uid:
            continue
        # [DEDUP_2026_06_01] Skip the per-turn ``turn_summary`` roll-up
        # rows (record_turn) — their tokens are already accounted for by
        # the per-call ``agent_event`` rows (record_event). Counting both
        # inflated user totals ~2.5x and made them irreconcilable with the
        # per-agent rows. Same filter per_agent() uses.
        try:
            ex = json.loads(ex_raw) if ex_raw else {}
        except Exception:
            ex = {}
        if ex.get("kind") == "turn_summary":
            continue
        u = by_user.setdefault(
            uid,
            {
                "user_id": uid,
                "calls": 0,
                "to_llm": 0,
                "from_llm": 0,
                "to_llm_bytes": 0,
                "from_llm_bytes": 0,
                "tokens_in": 0,
                "tokens_out": 0,
                "conversations": set(),
                "redactions": Counter(),
                "last_seen": 0,
            },
        )
        if conv:
            u["conversations"].add(conv)
        cl = len(c) if c else 0
        if d == "TO_LLM":
            u["to_llm"] += 1
            u["to_llm_bytes"] += cl
            if c:
                for m in KIND_RE.finditer(c):
                    u["redactions"][m.group(1)] += 1
        elif d == "FROM_LLM":
            # Tokens live on the response side. Summing tokens_in on
            # TO_LLM too would double-count input (both rows carry it).
            u["from_llm"] += 1
            u["calls"] += 1
            u["from_llm_bytes"] += cl
            u["tokens_in"] += ti or 0
            u["tokens_out"] += to or 0
        if ts and ts > u["last_seen"]:
            u["last_seen"] = ts

    rows = []
    for u in by_user.values():
        u["redactions"] = dict(u["redactions"])
        u["redactions_total"] = sum(u["redactions"].values())
        u["conversations"] = len(u["conversations"])
        rows.append(u)
    rows.sort(key=lambda r: r["calls"], reverse=True)
    return rows[:top_n]


def per_agent(
    window_hours: Optional[int] = 24,
    top_n: int = 30,
    *,
    start_ts: Optional[float] = None,
    end_ts: Optional[float] = None,
    user_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Per-agent (agent-runtime IC) call + token rollup over the window.

    ``per_user`` collapses every background dispatch under the libreuser
    id the agents inherit, so the operator can't see which agent spent
    the quota.  This groups FROM_LLM rows by ``extra.agent_id`` instead
    — director / researcher / builder / critic-claim / critic-prose —
    answering "which agent burned my usage".

    Emits the SAME column shape as ``per_user`` (to_llm / from_llm /
    conversations / tokens_in / tokens_out / to_llm_bytes /
    redactions_total / last_seen) so the dashboard can list agents and
    users in ONE table.  ``owner_user_id`` is the user the agent most
    often acted for (so the endpoint can show its email).

    Token totals come from the per-call ``agent_event`` rows; the
    per-turn ``turn_summary`` rows are skipped so tokens aren't
    double-counted (the summary equals the sum of its events).
    """
    s, e, sub = _windowed_rows(start_ts, end_ts, window_hours, user_id)
    if sub is None or sub.num_rows == 0:
        return []

    dirs = sub.column("direction").to_pylist()
    extras = sub.column("extra").to_pylist()
    ti = sub.column("tokens_input").to_pylist()
    to_ = sub.column("tokens_output").to_pylist()
    cr = sub.column("cache_read").to_pylist()
    times = sub.column("created_at").to_pylist()
    contents = sub.column("content").to_pylist()
    convs = sub.column("conversation_id").to_pylist()
    users = sub.column("user_id").to_pylist()

    by_agent: dict[str, dict[str, Any]] = {}
    for i in range(sub.num_rows):
        try:
            ex = json.loads(extras[i]) if extras[i] else {}
        except Exception:
            ex = {}
        if ex.get("kind") == "turn_summary":
            continue  # avoid double-count — agent_event rows carry the tokens
        agent = ex.get("agent_id")
        if not agent:
            continue
        a = by_agent.setdefault(agent, {
            "agent_id": agent,
            "calls": 0,
            "to_llm": 0,
            "from_llm": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            "to_llm_bytes": 0,
            "cache_read": 0,
            "conversations": set(),
            "redactions": Counter(),
            "owner_ids": Counter(),
            "last_seen": 0,
        })
        c = contents[i] or ""
        if dirs[i] == "TO_LLM":
            a["to_llm"] += 1
            a["to_llm_bytes"] += len(c)
            for m in KIND_RE.finditer(c):
                a["redactions"][m.group(1)] += 1
        elif dirs[i] == "FROM_LLM":
            a["from_llm"] += 1
            a["calls"] += 1
            a["tokens_in"] += ti[i] or 0
            a["tokens_out"] += to_[i] or 0
            a["cache_read"] += cr[i] or 0
        if convs[i]:
            a["conversations"].add(convs[i])
        if users[i]:
            a["owner_ids"][users[i]] += 1
        if times[i] and times[i] > a["last_seen"]:
            a["last_seen"] = times[i]

    rows: list[dict[str, Any]] = []
    for a in by_agent.values():
        a["conversations"] = len(a["conversations"])
        a["redactions"] = dict(a["redactions"])
        a["redactions_total"] = sum(a["redactions"].values())
        a["owner_user_id"] = (
            a["owner_ids"].most_common(1)[0][0] if a["owner_ids"] else ""
        )
        del a["owner_ids"]
        rows.append(a)
    rows.sort(key=lambda r: r["tokens_in"], reverse=True)
    return rows[:top_n]


def recent_redactions(
    limit: int = 30,
    *,
    start_ts: Optional[float] = None,
    end_ts: Optional[float] = None,
    window_hours: Optional[int] = None,
    user_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Last N TO_LLM rows in window, summarised. NEVER returns raw content.

    Each emitted row is also enriched with the cache_create / cache_read /
    tokens_input / tokens_output values from the matching FROM_LLM
    response. Pii-proxy writes the request audit, then the response audit
    in the same conversation, so we pair each TO_LLM with the *next*
    FROM_LLM in that conversation during a single ascending-aid pass.

    We deliberately pair on ``conversation_id`` ALONE, not on
    ``(conversation_id, model)``. The model column doesn't survive the
    request/response round-trip cleanly: the request side stores the
    short alias the user picked (``claude-haiku-4-5``) while the response
    side stores the dated model id Anthropic returned
    (``claude-haiku-4-5-20251001``). Keying on the pair would orphan
    every haiku title-gen row in the window. Conversation-id alone is
    safe because pii-proxy serialises audit writes per conversation —
    requests and responses can't interleave within one conversation.

    Unpaired TO_LLM rows (response landed outside the window, or the call
    errored before audit) get ``None`` for those four fields — the
    dashboard renders an em-dash.
    """
    s, e, sub = _windowed_rows(start_ts, end_ts, window_hours, user_id)
    if sub is None or sub.num_rows == 0:
        return []

    # Materialize columns once so the inner loop is plain Python indexing
    n = sub.num_rows
    aids = sub.column("aid").to_pylist()
    dirs = sub.column("direction").to_pylist()
    convs = sub.column("conversation_id").to_pylist()
    models = sub.column("model").to_pylist()
    users = sub.column("user_id").to_pylist()
    contents = sub.column("content").to_pylist()
    times = sub.column("created_at").to_pylist()
    cc = sub.column("cache_create").to_pylist()
    cr = sub.column("cache_read").to_pylist()
    ti = sub.column("tokens_input").to_pylist()
    to_ = sub.column("tokens_output").to_pylist()
    extras = sub.column("extra").to_pylist()

    # Pass 1 (asc): build TO_LLM aid → FROM_LLM idx mapping.
    pair: dict[int, int] = {}
    pending: dict[str, int] = {}
    asc = sorted(range(n), key=lambda i: aids[i])
    for i in asc:
        conv = convs[i] or ""
        if dirs[i] == "TO_LLM":
            pending[conv] = i
        elif dirs[i] == "FROM_LLM":
            pi = pending.pop(conv, None)
            if pi is not None:
                pair[aids[pi]] = i

    # Pass 2 (desc): emit the latest N TO_LLM rows with paired stats
    out: list[dict[str, Any]] = []
    desc = sorted(range(n), key=lambda i: -aids[i])
    for i in desc:
        if dirs[i] != "TO_LLM":
            continue
        c = contents[i] or ""
        kinds = Counter(KIND_RE.findall(c))
        pi = pair.get(aids[i])
        # Pull the agent-runtime identity out of the extra JSON so the
        # dashboard can ATTRIBUTE each row to the IC that made it
        # (researcher / builder / critic / director).  Without this, every
        # background dispatch is invisible-as-an-agent: it inherits the
        # libreuser's user_id, so the operator sees "the libreuser" doing
        # everything instead of seeing the agents.  Best-effort parse.
        try:
            _ex = json.loads(extras[i]) if extras[i] else {}
        except Exception:
            _ex = {}
        out.append(
            {
                "aid": aids[i],
                "user_id": users[i],
                "conversation_id": convs[i],
                "model": models[i],
                "ts": times[i],
                "bytes": len(c),
                "redactions": dict(kinds),
                "cache_read": (cr[pi] if pi is not None else None),
                "cache_write": (cc[pi] if pi is not None else None),
                "tokens_input": (ti[pi] if pi is not None else None),
                "tokens_output": (to_[pi] if pi is not None else None),
                "turn_type":     _classify_turn(convs[i], c),
                "agent_id":      _ex.get("agent_id"),
                "task_id":       _ex.get("task_id"),
                "source":        _ex.get("source"),
            }
        )
        if len(out) >= limit:
            break
    return out


def _classify_turn(conv_id: str | None, content: str) -> str:
    """Classify a TO_LLM row so the dashboard can distinguish:

    - ``sub`` — sub-agent LLM call (conv_id prefixed ``sub-``, e.g.
      ``sub-69df785-age-abc12345`` set by sub-agents/core/request_ctx.py
      ``spawn_scope``)
    - ``tool`` — mid-flow tool-result follow-up. The body has no
      ``--- MEMORY CONTEXT ---`` preamble (TCMM short-circuits these)
      and the messages chain starts with a real user/assistant turn
      with a tool_result block embedded — usually visible as a
      ``[TOOL]`` marker in the audit text.
    - ``turn`` — main user turn. Carries the full TCMM preamble.

    The pii_audit schema doesn't have an explicit turn-type column, so
    we sniff from content. Cheap heuristic, ~zero false positives in
    practice because:
      * sub-agents are gated on conv_id prefix (no content needed)
      * the preamble starts with a distinctive ``--- MEMORY CONTEXT
        (from previous conversations) ---`` banner that only the
        ``_VEILGUARD_PREAMBLE_TEXT`` ever produces
      * everything else is by elimination a tool follow-up
    """
    cid = (conv_id or "")
    if cid.startswith("sub-"):
        return "sub"
    head = content[:200] if content else ""
    if "--- MEMORY CONTEXT (from previous conversations) ---" in head:
        return "turn"
    if "[TOOL]" in content[:2000]:
        return "tool"
    # Fallback: anything without the preamble looks like a tool /
    # continuation under the current proxy convention.
    return "tool"


def request_detail(aid: int) -> dict[str, Any]:
    """Return one audit row's full metadata + REDACTED content.

    The ``content`` column has already had PII replaced with REF_X
    tokens by the pii-proxy before being persisted, so returning it
    here exposes nothing the LLM didn't already see. The redaction
    counts are recomputed from the content for display.
    """
    try:
        if _PG:
            sub = _pg_table("aid = %s", [int(aid)],
                            ["aid", "user_id", "conversation_id", "direction", "model",
                             "created_at", "tokens_input", "tokens_output", "content"])
        else:
            # [LEAK_FIX_2026_06_01] Predicate-pushdown the single aid instead of
            # to_arrow()-ing the ENTIRE audit history just to find one row — that
            # full load made every popup click pull ~600MB and lag the dashboard.
            t = _open_table()
            sub = t.search().where(f"aid = {int(aid)}").to_arrow()
    except Exception as e:
        return {"error": f"open pii_audit failed: {e}"}

    if sub.num_rows == 0:
        return {"error": f"aid {aid} not found"}
    row = sub.slice(0, 1)
    content = row.column("content")[0].as_py() or ""
    kinds = Counter(KIND_RE.findall(content))
    return {
        "aid": row.column("aid")[0].as_py(),
        "user_id": row.column("user_id")[0].as_py(),
        "conversation_id": row.column("conversation_id")[0].as_py(),
        "direction": row.column("direction")[0].as_py(),
        "model": row.column("model")[0].as_py(),
        "ts": row.column("created_at")[0].as_py(),
        "tokens_input": row.column("tokens_input")[0].as_py(),
        "tokens_output": row.column("tokens_output")[0].as_py(),
        "bytes": len(content),
        "redactions": dict(kinds),
        "content": content,  # REDACTED — REF_PERSON_X / REF_EMAIL_X / etc.
    }


# ── [SPOT_CHECK_2026-06-09] block-class view + redaction-gap scanner ──
# UNAMBIGUOUS PII shapes only (no name guessing) so false positives stay
# low. A match in POST-redaction audit content means the redactor missed it.
_MISS_PATTERNS = [
    ("EMAIL",       re.compile(r"[\w.+-]+@[\w-]+\.[\w][\w.-]*")),
    ("SA_PHONE",    re.compile(r"(?<!\d)(?:\+27|0)\d{9}(?!\d)")),
    ("SA_ID",       re.compile(r"(?<!\d)\d{13}(?!\d)")),
    ("CREDIT_CARD", re.compile(r"(?<!\d)(?:\d[ -]?){15,16}\d(?!\d)")),
]


def recent_blocks(limit: int = 40) -> list[dict[str, Any]]:
    """Recent TCMM archive blocks + block_class, for spot-checking how each
    turn was classified. Reads v_archive on the SAME Postgres as pii_audit.
    Fully-shaped [] on the Lance backend or any error."""
    if not _PG:
        return []
    import psycopg2
    out: list[dict[str, Any]] = []
    try:
        conn = psycopg2.connect(_AUDIT_DSN)
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT aid, namespace, user_id, block_class, recallable, "
                    "left(text, 200) AS preview, created_ts FROM v_archive "
                    "ORDER BY created_ts DESC NULLS LAST LIMIT %s",
                    [int(limit)],
                )
                for aid, ns, uid, bc, rec, prev, ts in cur.fetchall():
                    out.append({
                        "aid": aid,
                        "conversation_id": ns,
                        "user_id": uid,
                        "block_class": bc or "(unclassified)",
                        "recallable": bool(rec),
                        "preview": " ".join((prev or "").split())[:140],
                        "created_ts": float(ts) if ts is not None else None,
                    })
        finally:
            conn.close()
    except Exception as e:
        logger.warning("recent_blocks failed: %s", e)
    return out


def suspected_pii_misses(
    limit: int = 40,
    *,
    start_ts: Optional[float] = None,
    end_ts: Optional[float] = None,
    user_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Scan recent TO_LLM audit content for raw PII the redactor MISSED.

    Content is POST-redaction (real PII -> REF_* tokens), so any unambiguous
    PII shape still present as plain text is a redaction gap. One row per
    (turn, suspected token). REF_* tokens never match these patterns."""
    s, e, sub = _windowed_rows(start_ts, end_ts, None, user_id)
    if sub is None:
        return []
    dirs = sub.column("direction").to_pylist()
    contents = sub.column("content").to_pylist()
    convs = sub.column("conversation_id").to_pylist()
    users = sub.column("user_id").to_pylist()
    tss = sub.column("created_at").to_pylist()
    out: list[dict[str, Any]] = []
    for d, c, conv, uid, ts in zip(dirs, contents, convs, users, tss):
        if d != "TO_LLM" or not c:
            continue
        seen: set = set()
        for kind, pat in _MISS_PATTERNS:
            for m in pat.finditer(c):
                tok = m.group(0).strip()
                if not tok or tok in seen:
                    continue
                seen.add(tok)
                out.append({
                    "type": kind,
                    "snippet": tok[:80],
                    "conversation_id": conv,
                    "user_id": uid,
                    "created_ts": float(ts) if ts is not None else None,
                })
        if len(out) >= limit:
            break
    return out[:limit]


def nlp_errors(limit: int = 40) -> dict[str, Any]:
    """Recent NLP worker errors from nlp_error_log + a count-by-kind summary,
    for the 'see NLP worker errors' panel. Fully-shaped empty result on the
    Lance backend or any error."""
    out: dict[str, Any] = {"recent": [], "by_kind": {}, "total": 0}
    if not _PG:
        return out
    import psycopg2
    try:
        conn = psycopg2.connect(_AUDIT_DSN)
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT error_kind, count(*) FROM nlp_error_log GROUP BY 1 ORDER BY 2 DESC")
                out["by_kind"] = {(k or "?"): int(c) for k, c in cur.fetchall()}
                out["total"] = sum(out["by_kind"].values())
                cur.execute(
                    "SELECT extract(epoch FROM ts), aid, stage, error_kind, finish_reason, "
                    "left(detail,160), left(text_snippet,100) FROM nlp_error_log "
                    "ORDER BY ts DESC NULLS LAST LIMIT %s", [int(limit)])
                for ts, aid, stage, kind, fr, detail, snip in cur.fetchall():
                    out["recent"].append({
                        "ts": float(ts) if ts is not None else None,
                        "aid": aid, "stage": stage, "error_kind": kind,
                        "finish_reason": fr, "detail": detail or "",
                        "snippet": " ".join((snip or "").split())[:100],
                    })
        finally:
            conn.close()
    except Exception as e:
        logger.warning("nlp_errors failed: %s", e)
    return out


def messages_per_user(
    window_hours: Optional[int] = 24,
    top_n: int = 50,
    *,
    start_ts: Optional[float] = None,
    end_ts: Optional[float] = None,
    direction: Optional[str] = None,  # None | "TO_LLM" | "FROM_LLM"
) -> list[dict[str, Any]]:
    """Per-user *message* counts (separate from redactions).

    Counts messages by direction + cumulative byte volume + token totals
    + distinct conversation count, sorted by call volume. Same shape as
    ``per_user`` but trimmed for the messages-focused panel.
    """
    rows = per_user(
        window_hours=window_hours,
        top_n=top_n,
        start_ts=start_ts,
        end_ts=end_ts,
    )
    if direction == "TO_LLM":
        rows.sort(key=lambda r: r.get("to_llm", 0), reverse=True)
    elif direction == "FROM_LLM":
        rows.sort(key=lambda r: r.get("from_llm", 0), reverse=True)
    return rows


# ── Prompt-cache stats ────────────────────────────────────────────────────
#
# Anthropic prompt-caching token economics, Sonnet 4.7 base rates as of
# 2026-04. cache_read is 10% of the input rate; cache_write_5m is 1.25× the
# input rate. The dashboard math is self-contained here so price moves are
# a one-file edit. Higher-tier models (Opus) bill ~5× these rates but the
# RATIO (savings vs base input) is identical, so the dashboard's "% of
# input from cache" stays accurate cross-model. The dollar figure is
# Sonnet-equivalent — a useful relative indicator, conservative for Opus.
_RATE_INPUT_PER_M = 3.0          # $/M tokens — base input
_RATE_CACHE_READ_PER_M = 0.30    # $/M tokens — 10% of input
_RATE_CACHE_WRITE_PER_M = 3.75   # $/M tokens — 1.25× input (5-min TTL)


def cache_overview(
    window_hours: Optional[int] = 24,
    *,
    start_ts: Optional[float] = None,
    end_ts: Optional[float] = None,
    user_id: Optional[str] = None,
) -> dict[str, Any]:
    """Aggregated prompt-cache (KV-cache) stats over a window.

    The numbers come from FROM_LLM rows of pii_audit, which carry the
    ``cache_create``/``cache_read``/``tokens_input`` values Anthropic
    returned in its response. TO_LLM rows are skipped — those are the
    request-side audit and don't have usage filled in.

    Returns counts, token totals, hit rate, time-series suitable for a
    stacked chart, and a Sonnet-equivalent cost-savings estimate.
    """
    empty = {
        "start_ts": 0.0,
        "end_ts": 0.0,
        "window_hours": window_hours,
        "user_id": user_id,
        "from_llm_rows": 0,
        "cache_hits": 0,
        "cache_writes": 0,
        "no_cache": 0,
        "hit_rate": 0.0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "input_tokens": 0,
        "input_volume_from_cache_ratio": 0.0,
        "savings_usd": 0.0,
        "extra_cost_usd": 0.0,
        "net_savings_usd": 0.0,
        "time_series": [],
        "bucket_seconds": 3600,
    }
    s, e, sub = _windowed_rows(start_ts, end_ts, window_hours, user_id)
    if sub is None:
        return {
            **empty,
            "start_ts": s,
            "end_ts": e,
            "error": "pii_audit table not found",
        }

    dirs = sub.column("direction").to_pylist()
    extras = sub.column("extra").to_pylist()

    def _is_turn_summary(i: int) -> bool:
        try:
            return (json.loads(extras[i]) if extras[i] else {}).get("kind") == "turn_summary"
        except Exception:
            return False

    # [DEDUP_2026_06_01] FROM_LLM rows EXCLUDING the per-turn roll-up
    # (record_turn) so cache/token totals aren't double-counted against
    # the per-call agent_event rows that carry the same numbers.
    from_idx = [
        i for i, d in enumerate(dirs)
        if d == "FROM_LLM" and not _is_turn_summary(i)
    ]
    if not from_idx:
        return {**empty, "start_ts": s, "end_ts": e}

    cc = sub.column("cache_create").to_pylist()
    cr = sub.column("cache_read").to_pylist()
    ti = sub.column("tokens_input").to_pylist()
    ts_col = sub.column("created_at").to_pylist()

    span = max(e - s, 1)
    bucket_size = _pick_bucket_size(span)
    buckets: dict[int, dict[str, int]] = {}

    n = hits = writes = no_cache = 0
    sum_cr = sum_cc = sum_ti = 0
    for i in from_idx:
        n += 1
        v_cr = cr[i] or 0
        v_cc = cc[i] or 0
        v_ti = ti[i] or 0
        sum_cr += v_cr
        sum_cc += v_cc
        sum_ti += v_ti
        if v_cr > 0:
            hits += 1
        if v_cc > 0:
            writes += 1
        if v_cr == 0 and v_cc == 0:
            no_cache += 1

        b = int(ts_col[i] // bucket_size) * bucket_size
        bd = buckets.setdefault(
            b, {"cache_read": 0, "cache_write": 0, "input": 0, "n": 0}
        )
        bd["cache_read"] += v_cr
        bd["cache_write"] += v_cc
        bd["input"] += v_ti
        bd["n"] += 1

    time_series = [
        {"ts": int(b), **vals} for b, vals in sorted(buckets.items())
    ]

    hit_rate = (hits / n) if n else 0.0
    # Total "input paths" = bytes that had to reach Anthropic on this turn,
    # whether served from cache or charged at full rate. This is the
    # denominator most operators care about: "of the input I sent, what
    # fraction did the cache absorb?"
    total_input_paths = sum_cr + sum_ti
    vol_ratio = (sum_cr / total_input_paths) if total_input_paths else 0.0

    # cache_read saves (input - cache_read) per token vs paying base input
    # cache_write costs (cache_write - input) extra per token (the "tax")
    savings = sum_cr * (_RATE_INPUT_PER_M - _RATE_CACHE_READ_PER_M) / 1_000_000
    extra = sum_cc * (_RATE_CACHE_WRITE_PER_M - _RATE_INPUT_PER_M) / 1_000_000

    return {
        "start_ts": s,
        "end_ts": e,
        "window_hours": window_hours,
        "user_id": user_id,
        "from_llm_rows": n,
        "cache_hits": hits,
        "cache_writes": writes,
        "no_cache": no_cache,
        "hit_rate": round(hit_rate, 4),
        "cache_read_tokens": sum_cr,
        "cache_write_tokens": sum_cc,
        "input_tokens": sum_ti,
        "input_volume_from_cache_ratio": round(vol_ratio, 4),
        "savings_usd": round(savings, 4),
        "extra_cost_usd": round(extra, 4),
        "net_savings_usd": round(savings - extra, 4),
        "time_series": time_series,
        "bucket_seconds": bucket_size,
    }
