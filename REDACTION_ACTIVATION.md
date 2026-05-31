# PII Redaction — Render-Layer Activation (2026-05-30)

Everything below is **implemented + locally tested**. The redaction redesign is
**off by default** (`VEILGUARD_REDACT_IN_RENDER` unset → current boundary
behavior, byte-for-byte). Flipping it on moves redaction into TCMM's render
layer, where it's **redacted once per live-memory-block aid** (FragmentCache),
**immune to cache-marker churn**, and **7× faster on warm turns**.

This doc is the only thing left — the deploy-time wiring that can't run or be
validated locally (it lives in `tcmm-service`).

---

## Already done (local, tested green)

- `pii/redactor.py` — unified, fail-closed redactor; aid-keyed cache;
  `warm_batch()` (batched cold path); `warm()` (large-doc prewarm);
  `VEILGUARD_REDACT_IN_RENDER` boundary-skip; content memo.
- `pii/session_store.py` — `SessionId.canonical()` / `SessionId.parse()`
  (the wire contract so render-sid == rehydrate-sid).
- TCMM `core/renderers/base_renderer.py` — host-injectable hook:
  `set_pii_hook(fn)`, `set_redact_sid(sid)` / `reset_redact_sid(tok)`,
  `apply_pii_redaction()`. `RenderPolicy.redact` + `VeilguardPolicy.redact`
  route through it (no-op until a hook is registered).
- `llm/tcmm_client.py` — `render_structured(..., pii_sid=...)` carries the
  caller's `SessionId.canonical()`.
- `agent/base.py` — passes `pii_sid=sid.canonical()`; system blocks go through
  `redact_memory_blocks` (boundary-skip when flag on).
- `agent-proxy/app/main.py` — startup calls `redactor.warm()`.

Tests: `_e2e_render_redact.py` (full render→redact→rehydrate w/ sid contract),
`_test_pii_hook.py`, `_test_redact_in_render_skip.py`, `_bench_aid_redaction.py`,
`_live_perf_test.py`, `pii/tests/test_byte_stability.py` — all green.

---

## Deploy-time wiring (3 things, all in `tcmm-service`)

> `tcmm-service/server.py` + `api/models.py` are **VM source-of-truth** (local
> copies are stale — local has no `/render_structured` route). Apply on the VM
> copy and keep `pip` deps in sync.

### 1. tcmm-service imports `veilguard.pii` + shares the session store
- Put `pii/` on the tcmm-service `PYTHONPATH` (sibling mount, like the proxy's
  `/pii`) and install Presidio + the spaCy model into its env.
- Set `VEILGUARD_PII_DB_PATH` to the **same Lance path** the proxy/runtime use, so
  REF tokens minted at render are rehydratable at the boundary.

### 2. Register the hook at startup (`lifespan`, server.py:84)
```python
try:
    from core.renderers.base_renderer import set_pii_hook
    from pii import get_redactor
    _r = get_redactor()          # fail_closed=True by default
    _r.warm()                    # large-doc prewarm
    set_pii_hook(lambda text, sid: _r.redact_text(text, sid))
    logger.info("[pii] render-layer redaction hook registered")
except Exception as e:
    logger.warning("[pii] hook registration skipped (boundary redacts): %s", e)
```

### 3. Thread the sid per render request
`api/models.py` → `RenderRequest`:
```python
    pii_sid: str = Field("", description="caller SessionId.canonical()")
```
`/render` and `/render_structured` handlers — wrap the render call:
```python
from core.renderers.base_renderer import set_redact_sid, reset_redact_sid
from pii import SessionId
_tok = set_redact_sid(SessionId.parse(req.pii_sid)) if req.pii_sid else None
try:
    result = renderer.render_structured(req.task_query)   # or renderer.render(...)
finally:
    if _tok is not None:
        reset_redact_sid(_tok)
```
Optional cold-start: before the render,
`_r.warm_batch([b.text for b in tcmm.live_blocks], SessionId.parse(req.pii_sid))`.

### 4. Flip the flag (boundary)
`VEILGUARD_REDACT_IN_RENDER=1` on the **proxy + agent-runtime**.
ON → render redacts memory, boundary skips system re-scan + only redacts the
latest prompt + rehydrates. OFF → today's behavior (= rollback).

---

## Activation order (safe)
1. Deploy #1–#3 with the flag **OFF** (hook registered, boundary still redacts →
   no behavior change). Confirm `[pii] render-layer redaction hook registered`.
2. Flip `VEILGUARD_REDACT_IN_RENDER=1` under the **real tenant** (not
   `default`/`ptest`). Send a turn with known PII.
3. Verify in `pii_audit` + live UI: outbound prompt has REF tokens (no raw PII),
   response rehydrated, conversation works end-to-end.
4. Watch render/`aid_cache` logs + per-turn latency (warm ~flat) + rehydration.
5. **Rollback** = unset the flag (instant, no redeploy).

## Open product decision (not a blocker)
Sid contract guarantees render-sid == rehydrate-sid **per path**. Proxy keys
per-USER (`pii-{user}`), runtime per-CONVERSATION — they needn't agree with each
other. If you later want "same person → same token across a user's
conversations," standardize the proxy onto the canonical `SessionId`.
