"""
TCMM NLP out-of-process worker.

Long-lived subprocess that owns the AIStudioNLPAdapter (Gemini Flash +
SpaCy). Spawned by NLPAdapterStub in the parent tcmm-service process so
that a crash in the NLP code path (HTTP timeout, JSON parser regression,
SpaCy memory blow-up, circuit-breaker bug, etc.) cannot bring down
tcmm-service itself — the parent supervises and respawns the child
without touching recall / render / observe traffic.

Modeled on services/reranker-service/reranker_worker.py. Same
newline-delimited JSON protocol on stdin/stdout, same ``{"ok": true,
"ready": true}`` ready handshake, same stderr-for-logs convention.

Protocol — request:
  {"op": "ping"}
  {"op": "process_batch_gemma", "texts": [...], "roles": [...]?}
  {"op": "process_batch",       "texts": [...], "text_offsets": [...]?,
                                "topic_hints": [...]?,
                                "entity_hints": [...]?,
                                "roles": [...]?}
  {"op": "classify_episodic",   "text": "...", "prev_text": "...",
                                "next_text": "..."}
  {"op": "classify_episodic_recallable", "text": "...", "prev_text":
                                "...", "next_text": "..."}
  {"op": "extract_topics_batch", "texts": [...]}
  {"op": "canonicalize_topics", "all_topics": [...]}
  {"op": "gemma_generate",      "prompt": "..."}

Response (always one line of JSON):
  {"ok": true, "result": <call-specific>}
  {"ok": false, "error": "<msg>", "error_type": "<exc class>"}

The SpaCy ``nlp`` object on AIStudioNLPAdapter is NOT exposed through
this protocol — it's an in-memory object that cannot serialize across a
pipe. Parent-process callers that touch ``adapter.nlp`` (the dream
engine and the HR-fusion sentence-splitter) get their own local SpaCy
instance from NLPAdapterStub. Small memory duplication, full
transparency to existing call sites.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from typing import Any


# ─── stdout discipline ─────────────────────────────────────────────────
# Anything that prints to stdout from inside this child corrupts the
# JSON-RPC protocol — the parent stub reads one JSON line per request
# off our stdout and `bad json` errors trip the call into the fail-open
# branch. Several adapters we import print boot banners (FastEmbed's
# ONNX loader, GLiNER, future deps we don't control), so we capture
# the real stdout once at boot and reroute the public ``sys.stdout``
# to stderr. Only ``_write_response`` writes JSON, and it writes to
# the captured ``_PROTO_OUT`` — no third-party code can clobber it.
_PROTO_OUT = sys.stdout
sys.stdout = sys.stderr


def _log(msg: str) -> None:
    # stderr — keeps the JSON-on-stdout protocol clean.
    print(f"[nlp_worker pid={os.getpid()}] {msg}", file=sys.stderr, flush=True)


def _write_response(payload: dict) -> None:
    _PROTO_OUT.write(json.dumps(payload) + "\n")
    _PROTO_OUT.flush()


def _build_adapter():
    """Instantiate AIStudioNLPAdapter exactly the way tcmm-service does.

    Mirrors the env-var → constructor-arg pulling at
    server.py:_init_shared_resources so behaviour is identical whether
    the adapter is loaded in-process or in this child.
    """
    from adapters.ai_studio_nlp_adapter import AIStudioNLPAdapter

    nlp_model = os.environ.get("NLP_MODEL", "gemini-2.5-flash")
    api_key = (
        os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("VERTEX_API_KEY")
        or ""
    )
    if not api_key:
        raise RuntimeError(
            "NLP worker needs an API key — none of GOOGLE_API_KEY / "
            "GEMINI_API_KEY / VERTEX_API_KEY are set."
        )
    return AIStudioNLPAdapter(api_key=api_key, model=nlp_model)


def _dispatch(adapter, op: str, req: dict) -> Any:
    """Map the op string + request to the corresponding adapter call.

    Each branch validates only the fields it needs and lets KeyError
    surface as a structured error — typo'd ops can't silently no-op.
    """
    if op == "ping":
        return {"pid": os.getpid(), "model": adapter.model}

    if op == "process_batch_gemma":
        texts = req["texts"]
        roles = req.get("roles")
        return adapter.process_batch_gemma(texts, roles=roles)

    if op == "process_batch":
        return adapter.process_batch(
            req["texts"],
            text_offsets=req.get("text_offsets"),
            topic_hints=req.get("topic_hints"),
            entity_hints=req.get("entity_hints"),
            roles=req.get("roles"),
        )

    if op == "classify_episodic":
        return adapter.classify_episodic(
            req["text"],
            prev_text=req.get("prev_text"),
            next_text=req.get("next_text"),
        )

    if op == "classify_episodic_recallable":
        cls, recallable = adapter.classify_episodic_recallable(
            req["text"],
            prev_text=req.get("prev_text"),
            next_text=req.get("next_text"),
        )
        return {"class": cls, "recallable": bool(recallable)}

    if op == "extract_topics_batch":
        return adapter.extract_topics_batch(req["texts"])

    if op == "canonicalize_topics":
        return adapter.canonicalize_topics(req["all_topics"])

    if op == "gemma_generate":
        return adapter._gemma_generate(req["prompt"])

    raise ValueError(f"unknown op: {op!r}")


def main() -> int:
    _log("booting")

    t0 = time.perf_counter()
    try:
        adapter = _build_adapter()
    except Exception as e:
        # Boot failure: the parent's _spawn_worker() loop watches stdout
        # for the ready handshake. Emit a structured failure event so the
        # parent can log a precise reason instead of "child closed stdout".
        _log(f"adapter init FAILED: {type(e).__name__}: {e}")
        try:
            _write_response({
                "ok": False,
                "ready": False,
                "error": str(e),
                "error_type": type(e).__name__,
            })
        except Exception:
            pass
        return 2
    init_ms = (time.perf_counter() - t0) * 1000.0
    _log(f"adapter ready model={adapter.model} init_ms={init_ms:.0f}")

    # Ready handshake. Same shape as reranker_worker so the parent's
    # supervisor logic can be near-identical between the two.
    _write_response({"ok": True, "ready": True, "pid": os.getpid()})

    # Main request loop. One request → one response, in order, on stdin.
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:
            _write_response({
                "ok": False,
                "error": f"bad json: {e}",
                "error_type": "JSONDecodeError",
            })
            continue

        op = req.get("op")
        try:
            result = _dispatch(adapter, op, req)
            _write_response({"ok": True, "result": result})
        except Exception as e:
            # Don't crash on bad input or transient adapter errors —
            # report the failure, keep serving. The parent can decide
            # whether repeated errors should trigger a restart.
            _log(
                f"op={op!r} FAILED: {type(e).__name__}: {e}\n"
                f"{traceback.format_exc()}"
            )
            _write_response({
                "ok": False,
                "error": str(e),
                "error_type": type(e).__name__,
            })

    _log("stdin closed, exiting")
    return 0


if __name__ == "__main__":
    sys.exit(main())
