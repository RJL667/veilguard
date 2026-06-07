"""PII redaction + rehydration — byte-stable, cache-safe.

Ported from agent-proxy/app/redactor.py with these changes:
  - Session store is Lance-backed (veilguard.pii.session_store) so
    pii-proxy AND agent-runtime see the same mapping for the same conv.
  - SessionId is now a typed key (tenant_id, conv_id) — not just a string.
  - Added `redact_blocks` + `rehydrate_blocks` for the post-TCMM-render
    pipeline (block-level redaction preserves cache_control markers).

Hard invariants (tested in tests/test_byte_stability.py):
  A. Same input + same SessionId → same output bytes
  B. Existing REF tokens NEVER change across turns
  C. Block dicts mutate only `.text`; cache_control / type / other
     keys are preserved unchanged
  D. No whitespace normalization, no Unicode normalization — pure
     N-byte substring replacement
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Union

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_anonymizer import AnonymizerEngine

from .recognizers import (
    PassportRecognizer,
    SouthAfricanBankAccountRecognizer,
    SouthAfricanIDRecognizer,
    SouthAfricanPhoneRecognizer,
)
from .session_store import SessionId, get_store as _get_store

logger = logging.getLogger("veilguard.pii.redactor")


# ── Fail-closed contract ─────────────────────────────────────────────────


class RedactionUnavailable(RuntimeError):
    """Raised when PII redaction cannot complete for ANY reason.

    Veilguard fails CLOSED at the LLM boundary: when Presidio can't
    analyze (model unload, OOM, regex engine error, etc.) the request is
    REFUSED rather than forwarded with raw PII intact.  Both consumers
    catch this:
      - agent-proxy: returns HTTP 503 (`error.type=redaction_unavailable`)
      - agent-runtime: ends the turn with an `error` event

    NOT a failure (these return text unchanged): empty/short text, no PII
    detected, all matches were our own REF_* tokens.

    Defined here so `from pii import RedactionUnavailable` resolves on the
    single shared module — there is no longer a separate proxy-local
    redactor with its own copy.
    """
    pass


# JSON keys whose string VALUES carry user-authored content to scan.
USER_CONTENT_KEYS = {
    "text", "content", "query", "prompt", "input", "message", "value",
    "system",
}

# JSON keys that are metadata, never user content — passed through verbatim.
SKIP_KEYS = {
    "model", "type", "role", "stop_reason", "id", "object",
    "created", "usage", "index", "finish_reason", "stream",
    "max_tokens", "temperature", "top_p", "anthropic_version",
    "name", "source", "media_type", "cache_control", "tool_use_id",
    "signature", "thinking",  # Anthropic thinking blocks — signature dies if content changes
}

# Veilguard-internal block metadata keys.  Stripped before the body goes
# upstream (Anthropic 400s on unknown fields).  `_skip_pii` is the legacy
# "this block is PII-free" sentinel; `_vg_*` is the richer render-time
# contract (see PII_FAST_REDACTION_SPEC.md §3.2) that activates
# automatically once TCMM's renderer emits it.
_META_KEYS = {
    "_skip_pii", "_vg_id", "_vg_immutable", "_vg_pii",
    # [PII_AID_CACHE_2026_05_30] live-memory-block render tags:
    "_vg_aid",        # origin_archive_id — the redaction cache key
    "_vg_clean",      # True = immutable/static (preamble, tools) — never redact
    "_vg_stability",  # stability_class (byte_stable, …) — informational
}


# ── Config ──────────────────────────────────────────────────────────────


PII_ENTITIES = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "PERSON",
    "IBAN_CODE",
    "IP_ADDRESS",
    "SA_ID_NUMBER",
    "SA_PHONE_NUMBER",
    "SA_BANK_ACCOUNT",
    "PASSPORT_NUMBER",
    "DENY_PII",
]


def _load_allow_list_file() -> list[str]:
    """Load the STATIC allow_list FILE — the immutable fail-safe base so
    Presidio doesn't redact `Docker`, `LibreChat`, etc. as PERSON (spaCy
    en_core_web_lg false-positives those at ~0.85).

    Search order:
      1. ${PII_ALLOW_LIST_PATH} env override
      2. veilguard/pii/allow_list.txt (this dir)
      3. agent-proxy/app/allow_list.txt (compat with the old location)
    """
    candidates = [
        Path(__file__).parent / "allow_list.txt",
        Path(__file__).parent.parent / "agent-proxy" / "app" / "allow_list.txt",
    ]
    env_override = os.environ.get("PII_ALLOW_LIST_PATH", "")
    if env_override:
        candidates.insert(0, Path(env_override))
    for p in candidates:
        if p.is_file():
            with open(p, "r", encoding="utf-8") as f:
                terms = [
                    line.strip() for line in f
                    if line.strip() and not line.strip().startswith("#")
                ]
            logger.info(f"[pii] loaded {len(terms)} allow_list terms from file {p}")
            return terms
    logger.warning("[pii] no allow_list.txt found — brand names may false-positive")
    return []


def _load_allow_list_db() -> list[str]:
    """[ALLOWLIST_DB_OVERLAY_2026-06-07] Active terms from the `pii_allow_list`
    table — the LIVE, loop-maintained whitelist. Returns [] on ANY failure;
    the caller always has the file base, so DB problems degrade to file-only
    (never fail-open, never drop the base). Disable with PII_ALLOWLIST_DB=0.
    """
    if os.environ.get("PII_ALLOWLIST_DB", "1").lower() in ("0", "false", "no", "off"):
        return []
    try:
        from .session_store import _PG_DSN
        import psycopg2
        conn = psycopg2.connect(_PG_DSN, connect_timeout=3)
        try:
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute("SELECT term FROM pii_allow_list WHERE active = true")
            return [r[0] for r in cur.fetchall() if r and r[0]]
        finally:
            conn.close()
    except Exception as e:
        logger.warning(
            "[pii] allow_list DB overlay unavailable (%s); file-only this cycle",
            type(e).__name__,
        )
        return []


def _load_allow_list() -> list[str]:
    """The whitelist is the `pii_allow_list` DB table — the ONLY runtime source.
    No file is read during redaction. If the DB returns nothing (empty or
    unreachable) the list is EMPTY → the redactor OVER-redacts (fail-closed,
    never leaks) until the DB recovers; the background refresh keeps the last
    good list in memory across transient blips. `allow_list.txt` is retained
    purely as a one-time SEED source for the table — never loaded here.
    """
    db = _load_allow_list_db()
    if db:
        logger.info("[pii] allow_list = %d terms from DB (pii_allow_list)", len(db))
    else:
        logger.warning(
            "[pii] allow_list DB returned 0 terms — EMPTY allow_list "
            "(fail-closed; brands/code may over-redact until the DB recovers)"
        )
    return db


def _load_deny_list() -> list[str]:
    """[DENYLIST_2026-06-07] Active terms from the `pii_deny_list` table — the
    LLM-fed BLACKLIST of high-confidence MISSED PII. These are FORCE-redacted
    even when the default recognizers miss them. This is the SAFE direction:
    over-redacting is a quality bug, not a leak, so the LLM's flags auto-apply
    with NO confirm gate (the asymmetric opposite of the whitelist). Returns []
    on ANY failure (defaults still run). Disable with PII_DENYLIST_DB=0."""
    if os.environ.get("PII_DENYLIST_DB", "1").lower() in ("0", "false", "no", "off"):
        return []
    try:
        from .session_store import _PG_DSN
        import psycopg2
        conn = psycopg2.connect(_PG_DSN, connect_timeout=3)
        try:
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute("SELECT term FROM pii_deny_list WHERE active = true")
            terms = [r[0] for r in cur.fetchall() if r and r[0]]
        finally:
            conn.close()
        logger.info("[pii] deny_list = %d terms from DB (pii_deny_list)", len(terms))
        return terms
    except Exception as e:
        logger.warning("[pii] deny_list DB unavailable (%s); none this cycle", type(e).__name__)
        return []


def _build_deny_recognizer(terms):
    """Presidio recognizer that FORCE-flags blacklist terms (score 1.0) so they
    redact even when the default recognizers don't catch them. Passed per-call
    as an ad-hoc recognizer; None when empty (no-op)."""
    terms = [t.strip() for t in (terms or []) if t and t.strip()]
    if not terms:
        return None
    try:
        from presidio_analyzer import PatternRecognizer
        return PatternRecognizer(supported_entity="DENY_PII", deny_list=terms)
    except Exception as e:
        logger.warning("[pii] deny recognizer build failed (%s)", type(e).__name__)
        return None


# ── Redactor ────────────────────────────────────────────────────────────


class PIIRedactor:
    """Presidio-backed analyzer + Lance-backed session mapping.

    Singleton per process.  Holds the analyzer state (model weights) so
    the per-request overhead is just the analyze() call.
    """

    _instance: Optional["PIIRedactor"] = None

    def __init__(self, min_score: float = 0.7, fail_closed: bool = True):
        self.min_score = min_score
        # [FAIL_CLOSED_2026_05_29]  At the LLM boundary, a Presidio failure
        # must REFUSE the request, not ship raw PII.  Default True for both
        # the proxy and the runtime (both send to api.anthropic.com).  Set
        # False only for offline/test contexts that explicitly accept the
        # soft-degrade (return-raw) behavior.
        self._fail_closed = fail_closed

        registry = RecognizerRegistry()
        registry.load_predefined_recognizers()
        registry.add_recognizer(SouthAfricanIDRecognizer())
        registry.add_recognizer(SouthAfricanPhoneRecognizer())
        registry.add_recognizer(SouthAfricanBankAccountRecognizer())
        registry.add_recognizer(PassportRecognizer())

        self.analyzer = AnalyzerEngine(registry=registry)
        # [PII_PIPELINE_TRIM_2026_05_29]  THE redaction-speed fix.
        #
        # Presidio loads the FULL spaCy pipeline but only needs the
        # tokenizer (+ lemmatizer for its context enhancer, + NER for
        # PERSON).  Measured on a realistic 30k-token prompt:
        #   full pipeline ............ 7570 ms
        #   - parser ................. 4960 ms   (parser = 34%, NEVER used by Presidio)
        #   - NER .................... 1497 ms
        #   regex-only (all spaCy off) 463 ms    (~15ms/1000 tok — the documented floor)
        #
        # So: ALWAYS drop the dependency parser (free, prod-safe — Presidio
        # has no use for it).  And when PII_REGEX_ONLY is set, drop the
        # spaCy NER stack entirely → regex-only redaction (16x faster).
        # That sacrifices PERSON-name detection; the 8 pattern entities
        # (EMAIL/PHONE/CREDIT_CARD/IBAN/IP/SA_ID/SA_PHONE/SA_BANK) are
        # untouched since they're regex/checksum recognizers independent
        # of spaCy.
        self._regex_only = os.environ.get(
            "PII_REGEX_ONLY", ""
        ).lower() in ("1", "true", "yes", "on")
        try:
            _nlp = getattr(self.analyzer.nlp_engine, "nlp", {}) or {}
            _lang = _nlp.get("en")
            if _lang is not None:
                _drop = ["parser"]
                if self._regex_only:
                    _drop += ["ner", "tagger", "attribute_ruler",
                              "lemmatizer", "tok2vec"]
                for _p in _drop:
                    if _p in _lang.pipe_names:
                        _lang.disable_pipe(_p)
                logger.info(
                    "[pii] spaCy pipeline trimmed: disabled=%s active=%s "
                    "(regex_only=%s)",
                    _drop, _lang.pipe_names, self._regex_only,
                )
        except Exception as e:
            logger.warning("[pii] pipeline trim skipped (%s)", e)
        self.anonymizer = AnonymizerEngine()
        self.allow_list = _load_allow_list()
        self.deny_list = _load_deny_list()
        self.deny_recognizer = _build_deny_recognizer(self.deny_list)
        self.store = _get_store()

        # [PII_AID_CACHE_2026_05_30]  THE blazingly-fast layer, keyed on the
        # LIVE MEMORY BLOCK id (origin_archive_id / `aid`) — NOT on rendered
        # or cache-marked bytes.  A live memory block keeps its aid as it
        # migrates tiers (live → stable → frozen), so once redacted it stays
        # cached forever.  Cache markers (cache_control TTL, breakpoint
        # placement, re-bundling) churn turn-to-turn and MUST NOT be part of
        # the key — that was the old content-hash bug.  Key: (sid.root(), aid).
        # Value: redacted block text.  See PII_FAST_REDACTION_SPEC.md.
        self._aid_cache: "OrderedDict[tuple, str]" = OrderedDict()
        self._aid_cache_max = int(os.environ.get("PII_AID_CACHE_MAX", "8192"))
        self._aid_hits = 0
        self._aid_misses = 0
        self._uncached_redactions = 0   # blocks with no aid + not clean
        self._clean_skipped = 0
        # CLEAN-skip is provenance-anchored, NOT heuristic: we only skip
        # blocks that EXACTLY match a known system-authored static template
        # (magic prefix, Veilguard preamble).  Anything user-derived is
        # always scanned.  Toggle off for paranoid/audit runs.
        self._clean_skip = os.environ.get(
            "PII_CLEAN_SKIP", "1"
        ).lower() in ("1", "true", "yes", "on")
        self._clean_fingerprints = self._load_clean_fingerprints()
        # [REDACT_IN_RENDER_2026_05_30]  When TCMM's render-layer hook
        # (policy.redact) is active, memory blocks arrive ALREADY redacted.
        # The boundary then SKIPS re-scanning system/memory blocks (it would
        # be duplicate work, and tier-blob blocks have no aid to cache on) —
        # it only strips wire metadata and passes them through.  The latest
        # user PROMPT is still redacted (redact_messages, always).  Operator
        # enables this ONLY after wiring set_pii_hook server-side.
        self._redact_in_render = os.environ.get(
            "VEILGUARD_REDACT_IN_RENDER", "0"
        ).lower() in ("1", "true", "yes", "on")
        # [PII_WARM_BATCH_2026_05_30]  Content memo keyed on (sid, RAW block
        # text) → redacted text.  RAW block text is stable per aid (it's the
        # live-memory content, NOT the rendered/cache-marked bytes), so this
        # is safe.  warm_batch() fills it for many blocks in ONE Presidio
        # pass — collapsing the cold-render cost from N analyze() calls (N×
        # ~80ms pipeline overhead) to ~1.  redact_text() reads it first.
        self._content_memo: "OrderedDict[tuple, str]" = OrderedDict()
        self._content_memo_max = int(os.environ.get("PII_CONTENT_MEMO_MAX", "8192"))

        logger.info(
            f"[pii] redactor ready (Presidio + SA recognizers + "
            f"{len(self.allow_list)} allow_list terms; "
            f"fail_closed={self._fail_closed} clean_skip={self._clean_skip} "
            f"redact_in_render={self._redact_in_render} aid_cache_max={self._aid_cache_max})"
        )

        # [ALLOWLIST_DB_OVERLAY_2026-06-07] Background refresh of the DB overlay
        # so loop-promoted terms apply WITHOUT a restart. Strictly off the hot
        # path: a daemon thread reloads every PII_ALLOW_LIST_TTL_S and atomically
        # swaps self.allow_list (analyze() reads it per-call, so no rebuild).
        # All failures swallowed → the current list is kept; self.allow_list is
        # only ever replaced with a non-empty list (>= the file base).
        self._allow_ttl_s = float(os.environ.get("PII_ALLOW_LIST_TTL_S", "300"))
        _db_on = os.environ.get("PII_ALLOWLIST_DB", "1").lower() not in ("0", "false", "no", "off")
        if self._allow_ttl_s > 0 and _db_on:
            import threading, time as _time
            def _allow_refresh_loop():
                while True:
                    _time.sleep(self._allow_ttl_s)
                    try:
                        fresh = _load_allow_list()
                        if fresh:
                            self.allow_list = fresh
                        # deny_list: reload + rebuild the recognizer (may be empty)
                        self.deny_list = _load_deny_list()
                        self.deny_recognizer = _build_deny_recognizer(self.deny_list)
                    except Exception:
                        pass
            threading.Thread(
                target=_allow_refresh_loop, daemon=True, name="pii-allowlist-refresh",
            ).start()

    @staticmethod
    def _load_clean_fingerprints() -> tuple:
        """Exact-prefix fingerprints of system-authored, provably PII-free
        blocks.  Matched with startswith on the stripped text.  Kept tiny
        and anchored to literal constants — NEVER content heuristics — so a
        user message can't accidentally classify CLEAN (spec §5-E).
        """
        fps = [
            # The Anthropic OAuth magic prefix (base.py / anthropic_renderer).
            "You are a Claude agent, built on Anthropic's Claude Agent SDK.",
        ]
        try:
            # The Veilguard preamble template, when importable, is static.
            from agent.preamble import _VEILGUARD_PREAMBLE_TEMPLATE  # type: ignore
            head = _VEILGUARD_PREAMBLE_TEMPLATE.strip()[:120]
            if head:
                fps.append(head)
        except Exception:
            pass
        return tuple(fps)

    @classmethod
    def get(cls) -> "PIIRedactor":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── Session-key coercion ──────────────────────────────────────────

    @staticmethod
    def _coerce_sid(sid: Union[SessionId, str]) -> SessionId:
        """Accept either a typed SessionId (runtime) or a plain string
        (proxy's per-user `pii-{user}` key).  A bare string becomes the
        conv_id under the legacy tenant partition so both consumers share
        ONE store + ONE code path without changing the proxy's per-user
        token scoping.  Override the partition via PII_LEGACY_TENANT.
        """
        if isinstance(sid, SessionId):
            return sid
        legacy_tenant = os.environ.get("PII_LEGACY_TENANT", "_proxy")
        return SessionId(tenant_id=legacy_tenant, conv_id=str(sid or "pii-default"))

    @staticmethod
    def _aid_key(sid: SessionId, aid) -> tuple:
        """Cache key = (session root, live-memory-block aid).  Stable across
        turns and tier migration; independent of rendered/cache-marked bytes.
        """
        r = sid.root()
        return (r.tenant_id, r.conv_id, str(aid))

    def _is_clean(self, text: str) -> bool:
        """True iff `text` is a system-authored static template (never
        scan).  Provenance-anchored exact-prefix match only."""
        if not self._clean_skip:
            return False
        t = text.strip()
        return any(t.startswith(fp) for fp in self._clean_fingerprints)

    # ── Text-level ────────────────────────────────────────────────────

    def _analyze(self, text: str):
        """Single whole-text Presidio pass.

        Redaction caching happens one level up, keyed by live-memory-block
        aid (`redact_memory_blocks`) — a block is either an aid-cache HIT
        (never re-analyzed) or new (analyzed once).  This is just the raw
        analyze.

        Fail-closed: a genuine analyzer failure raises RedactionUnavailable
        (unless built with fail_closed=False) — never silently "no PII".
        """
        try:
            return self.analyzer.analyze(
                text=text,
                entities=PII_ENTITIES,
                language="en",
                score_threshold=self.min_score,
                allow_list=self.allow_list if self.allow_list else None,
                ad_hoc_recognizers=(
                    [self.deny_recognizer]
                    if getattr(self, "deny_recognizer", None) else None
                ),
            )
        except Exception as e:
            logger.error("[pii] analyze failed: %s", e)
            if self._fail_closed:
                raise RedactionUnavailable(
                    f"Presidio analyze failed: {type(e).__name__}: {e}"
                ) from e
            return []

    def redact_text(self, text: str, sid: Union[SessionId, str]) -> str:
        """Replace PII spans in `text` with REF tokens.

        Pure substring replacement — no whitespace normalization, no
        Unicode normalization.  Output bytes are deterministic given
        (text, sid).

        Fail-closed: a Presidio analyzer failure raises
        ``RedactionUnavailable`` (unless the redactor was built with
        ``fail_closed=False``).  "No PII found" is NOT a failure — it
        returns the text unchanged.
        """
        if not text or len(text.strip()) < 5:
            return text
        sid = self._coerce_sid(sid)
        mk = self._memo_key(sid, text)
        memoed = self._content_memo.get(mk)
        if memoed is not None:
            self._content_memo.move_to_end(mk)
            return memoed

        try:
            results = self._analyze(text)
            spans = [(r.entity_type, r.start, r.end) for r in results]
            redacted = self._apply_spans(text, spans, sid)
            # [PII_BATCH_WRITE_2026_05_29]  Persist all minted tokens in ONE
            # Lance write (add_mapping buffered them) before returning.
            try:
                self.store.flush()
            except Exception:
                pass
            self._memoize(mk, redacted)
            return redacted

        except RedactionUnavailable:
            raise
        except Exception as e:
            # [FAIL_CLOSED_2026_05_29]  Substitution-path failure also
            # refuses the request rather than shipping raw bytes.
            logger.error(f"[pii] redaction error: {e}")
            if self._fail_closed:
                raise RedactionUnavailable(
                    f"PII redaction failed: {type(e).__name__}: {e}"
                ) from e
            return text

    # ── Batched cold-path redaction (PII_WARM_BATCH_2026_05_30) ────────

    def _memo_key(self, sid: SessionId, text: str) -> tuple:
        r = sid.root()
        h = hashlib.blake2b(text.encode("utf-8", "replace"), digest_size=16).hexdigest()
        return (r.tenant_id, r.conv_id, h)

    def _memoize(self, key: tuple, value: str) -> None:
        self._content_memo[key] = value
        if len(self._content_memo) > self._content_memo_max:
            self._content_memo.popitem(last=False)

    def _apply_spans(self, text: str, spans, sid: SessionId) -> str:
        """Substitute REF tokens for PII spans (entity_type, start, end) in
        text-local coords.  Skips our own REF_ tokens; replaces end→start so
        offsets don't shift; mints/reuses deterministic tokens.  Does NOT
        flush (caller batches the Lance write)."""
        spans = [s for s in spans if not text[s[1]:s[2]].startswith("REF_")]
        if not spans:
            return text
        spans.sort(key=lambda s: s[1], reverse=True)
        out = text
        for (et, a, b) in spans:
            token = self.store.add_mapping(sid, et, text[a:b])
            out = out[:a] + token + out[b:]
        return out

    def warm_batch(self, texts, sid: Union[SessionId, str]) -> int:
        """Redact MANY live-memory block texts in ONE Presidio pass and
        memoize each (by raw text) so a later per-block redact_text() is a
        cache hit.  This is THE cold-start fix: instead of N analyze() calls
        (N × ~80ms fixed pipeline overhead), one analyze over the joined
        texts, spans split back per block by offset.  Call this before
        rendering with the live_blocks' raw texts.  Returns #blocks redacted.

        Fail-closed: a Presidio failure raises RedactionUnavailable.
        Determinism: produces byte-identical output to per-block redact_text
        (same _apply_spans + deterministic token store); a hard separator
        prevents cross-block span bleed.
        """
        sid = self._coerce_sid(sid)
        todo, keys, seen = [], [], set()
        for t in texts:
            if not isinstance(t, str) or len(t.strip()) < 5:
                continue
            k = self._memo_key(sid, t)
            if k in self._content_memo or k in seen:
                continue
            seen.add(k); todo.append(t); keys.append(k)
        if not todo:
            return 0
        SEP = "\n\x1f\x1f\n"        # unit separators — never PII, hard boundary
        ranges, pos = [], 0
        for t in todo:
            ranges.append((pos, pos + len(t)))
            pos += len(t) + len(SEP)
        results = self._analyze(SEP.join(todo))     # ONE analyze (fail-closed)
        per = [[] for _ in todo]
        for r in results:
            for i, (a, b) in enumerate(ranges):
                if a <= r.start < b:
                    per[i].append((r.entity_type, r.start - a, min(r.end - a, b - a)))
                    break
        for i, t in enumerate(todo):
            self._memoize(keys[i], self._apply_spans(t, per[i], sid))
        try:
            self.store.flush()
        except Exception:
            pass
        return len(todo)

    def warm(self) -> None:
        """Prewarm spaCy's LARGE-document path at startup.  The first NER
        pass over a big doc pays a one-time JIT/allocation tax (~2× a
        steady-state analyze) — paying it here at boot means the first real
        render doesn't.  Zero recall cost.  Host calls this once at startup
        (alongside the existing small Presidio prewarm).  Best-effort."""
        sample = ("Alice Johnson emailed alice.johnson@acme.co.za and called "
                  "+27 21 555 0100 about ID 8001015009087 and card "
                  "4111 1111 1111 1111 at the Cape Town branch on the 3rd. ") * 60
        try:
            self._analyze(sample)
        except Exception as e:
            logger.debug("[pii] large-doc warm skipped: %s", e)

    def rehydrate_text(self, text: str, sid: Union[SessionId, str]) -> str:
        # [PII_REHYDRATE_FASTPATH_2026_05_29]  No tokens → no work, no Lance.
        if not text or "REF_" not in text:
            return text
        return self.store.rehydrate(self._coerce_sid(sid), text)

    # ── Block-level (the new pipeline) ────────────────────────────────

    @staticmethod
    def _emit_block(blk: dict, new_text: Optional[str] = None) -> dict:
        """Copy a block, optionally replace `.text`, and STRIP Veilguard
        metadata keys (`_skip_pii`, `_vg_*`) so the block is wire-clean —
        Anthropic 400s on unknown fields.  All other keys (type,
        cache_control, …) pass through unchanged → cache-stability contract.
        """
        nb = {k: v for k, v in blk.items() if k not in _META_KEYS}
        if new_text is not None:
            nb["text"] = new_text
        return nb

    def _is_block_clean(self, blk: dict) -> bool:
        """True = immutable/static block that must NEVER be redacted:
        preamble, tool schemas, boundary markers.  Identified by the
        TCMM render tag `_vg_clean` / `_vg_pii=="clean"` / legacy
        `_skip_pii`, or the magic-prefix fingerprint fallback.
        """
        if blk.get("_vg_clean") is True or blk.get("_skip_pii") is True:
            return True
        if blk.get("_vg_pii") == "clean":
            return True
        text = blk.get("text", "")
        return isinstance(text, str) and self._is_clean(text)

    def redact_memory_blocks(
        self, blocks: list[dict], sid: Union[SessionId, str]
    ) -> list[dict]:
        """THE entry point for a TCMM-rendered system-block list.

        Redaction is keyed on the LIVE MEMORY BLOCK id (`_vg_aid` =
        origin_archive_id), NOT on rendered or cache-marked bytes:

          clean (preamble/tools/markers)  → emit verbatim, never scanned.
          has _vg_aid                     → aid-cache by (sid, aid):
                                              HIT  → reuse redacted text
                                                     (zero Presidio/Lance);
                                              MISS → redact once, cache by aid.
          no aid + not clean              → unknown live content: redact
                                            every turn (fail-safe), never
                                            cached on rendered bytes.

        A live memory block keeps its aid as it migrates tiers, so it is
        redacted exactly ONCE in its lifetime.  cache_control / marker churn
        cannot cause a miss — the aid is the only key.

        Strips `_vg_*` / `_skip_pii`.  Preserves cache_control + structure.
        Non-text blocks pass through unchanged.  Fail-closed propagates from
        redact_text on a real analyzer failure.
        """
        sid = self._coerce_sid(sid)
        # [REDACT_IN_RENDER_2026_05_30]  Memory already redacted at render →
        # strip wire metadata + pass through; do NOT re-scan.  (The latest
        # prompt is redacted separately via redact_messages.)
        if self._redact_in_render:
            return [self._emit_block(b) if isinstance(b, dict) else b for b in blocks]
        out: list[dict] = []
        for blk in blocks:
            if not isinstance(blk, dict) or blk.get("type") != "text":
                out.append(blk)
                continue
            text = blk.get("text", "")
            if not isinstance(text, str):
                out.append(self._emit_block(blk))
                continue
            if self._is_block_clean(blk):
                self._clean_skipped += 1
                out.append(self._emit_block(blk))
                continue
            aid = blk.get("_vg_aid")
            if aid is not None:
                key = self._aid_key(sid, aid)
                cached = self._aid_cache.get(key)
                if cached is not None:
                    self._aid_hits += 1
                    self._aid_cache.move_to_end(key)          # LRU touch
                    out.append(self._emit_block(blk, cached))
                    continue
                self._aid_misses += 1
                redacted = self.redact_text(text, sid)
                self._aid_cache[key] = redacted
                if len(self._aid_cache) > self._aid_cache_max:
                    self._aid_cache.popitem(last=False)       # evict LRU
                out.append(self._emit_block(blk, redacted))
                continue
            # No aid + not clean → can't cache safely on a stable key.
            # Redact every turn rather than risk keying on churning bytes.
            self._uncached_redactions += 1
            out.append(self._emit_block(blk, self.redact_text(text, sid)))
        self._log_block_stats()
        return out

    # Back-compat alias — old callers said redact_render_blocks.
    redact_render_blocks = redact_memory_blocks

    def _log_block_stats(self) -> None:
        total = self._aid_hits + self._aid_misses
        if total and total % 50 == 0:
            pct = self._aid_hits / total * 100
            logger.info(
                "[pii] aid_cache hits=%d misses=%d (%.0f%%)  clean_skipped=%d  "
                "uncached=%d  size=%d/%d",
                self._aid_hits, self._aid_misses, pct, self._clean_skipped,
                self._uncached_redactions, len(self._aid_cache), self._aid_cache_max,
            )

    def redact_blocks(self, blocks: list[dict], sid: Union[SessionId, str]) -> list[dict]:
        """Redact MESSAGE-content block lists (no aids — these are live
        conversation content, e.g. the latest user turn, redacted every
        turn).  Only `.text` is mutated; non-text blocks pass through.
        For SYSTEM/memory blocks use redact_memory_blocks (aid-keyed).
        """
        sid = self._coerce_sid(sid)
        out: list[dict] = []
        for blk in blocks:
            if not isinstance(blk, dict) or blk.get("type") != "text":
                out.append(blk)
                continue
            text = blk.get("text", "")
            new_text = self.redact_text(text, sid) if isinstance(text, str) else None
            out.append(self._emit_block(blk, new_text))
        return out

    def rehydrate_blocks(self, blocks: list[dict], sid: SessionId) -> list[dict]:
        """Reverse of redact_blocks.  Also rehydrates tool_use input args
        (a tool call may have been invoked with REF tokens that need
        swapping back before dispatch sees them).
        """
        out: list[dict] = []
        for blk in blocks:
            if not isinstance(blk, dict):
                out.append(blk)
                continue
            btype = blk.get("type")
            new_blk = dict(blk)
            if btype == "text" and isinstance(blk.get("text"), str):
                new_blk["text"] = self.rehydrate_text(blk["text"], sid)
            elif btype == "tool_use" and isinstance(blk.get("input"), dict):
                new_blk["input"] = self._rehydrate_dict(blk["input"], sid)
            out.append(new_blk)
        return out

    def _rehydrate_dict(self, d: dict, sid: SessionId) -> dict:
        """Walk a dict recursively, rehydrating any string values."""
        out = {}
        for k, v in d.items():
            if isinstance(v, str):
                out[k] = self.rehydrate_text(v, sid)
            elif isinstance(v, dict):
                out[k] = self._rehydrate_dict(v, sid)
            elif isinstance(v, list):
                out[k] = [
                    self._rehydrate_dict(x, sid) if isinstance(x, dict)
                    else (self.rehydrate_text(x, sid) if isinstance(x, str) else x)
                    for x in v
                ]
            else:
                out[k] = v
        return out

    # ── Message-level (Anthropic-shape conversation) ──────────────────

    def redact_messages(
        self, messages: list[dict], sid: Union[SessionId, str]
    ) -> list[dict]:
        """Redact a list of {role, content} messages.

        Content may be a plain string OR a list of blocks.  Other dict
        keys (e.g. `name`, `tool_use_id`) are passed through.

        [ASSISTANT_SHORTCIRCUIT_2026_05_29]  Content authored by the model
        in a prior turn (role=assistant / role=model) is NOT re-scanned:
          1. Re-redaction corrupts code the model emitted (`newline=""` →
             `REF_PERSON_N=""`).
          2. The model already saw genuine PII as REF_* tokens, so there's
             nothing left to redact — only false positives to introduce.
        Tool results live inside role=user (Anthropic) / role=tool (OpenAI)
        messages and remain IN scope for scanning.
        """
        sid = self._coerce_sid(sid)
        out: list[dict] = []
        for m in messages:
            if not isinstance(m, dict):
                out.append(m)
                continue
            new_m = dict(m)
            if m.get("role") in ("assistant", "model"):
                out.append(new_m)          # model-authored — never re-scan
                continue
            content = m.get("content")
            if isinstance(content, str):
                new_m["content"] = self.redact_text(content, sid)
            elif isinstance(content, list):
                new_m["content"] = self.redact_blocks(content, sid)
            out.append(new_m)
        return out

    # ── Whole-body (proxy compat entry point) ─────────────────────────

    def redact_json(
        self, obj, sid: Union[SessionId, str], depth: int = 0,
        in_user_content: bool = False, assistant_authored: bool = False,
    ):
        """Recursively redact PII from a provider request body.

        Compat shim for the proxy, which hands the WHOLE request dict.
        Behavior matches the legacy proxy redactor (USER_CONTENT_KEYS /
        SKIP_KEYS / `_skip_pii` / assistant short-circuit) with two
        upgrades:
          - the top-level Anthropic ``system`` block list is routed through
            ``redact_render_blocks`` (CLEAN-skip + FROZEN block cache), and
          - it runs on the SHARED fail-closed engine (line cache + memo +
            batch write), so the proxy gets the same speed + safety as the
            runtime.

        Fail-closed: ``RedactionUnavailable`` from ``redact_text``
        propagates (NOT caught here) — the proxy converts it to HTTP 503.
        """
        if depth == 0:
            sid = self._coerce_sid(sid)
        if depth > 20:
            return obj

        if isinstance(obj, str):
            if assistant_authored:
                return obj
            if in_user_content and len(obj) > 5:
                return self.redact_text(obj, sid)
            return obj
        elif isinstance(obj, list):
            return [
                self.redact_json(item, sid, depth + 1, in_user_content, assistant_authored)
                for item in obj
            ]
        elif isinstance(obj, dict):
            if obj.get("_skip_pii") is True:
                return {k: v for k, v in obj.items() if k not in _META_KEYS}
            child_assistant = assistant_authored
            role = obj.get("role")
            if isinstance(role, str) and role in ("assistant", "model"):
                child_assistant = True
            result = {}
            for key, value in obj.items():
                # Top-level Anthropic system block list → fast block path.
                if (
                    depth == 0 and key == "system" and isinstance(value, list)
                    and not child_assistant
                ):
                    result[key] = self.redact_render_blocks(value, sid)
                elif key in SKIP_KEYS:
                    result[key] = value
                elif key in USER_CONTENT_KEYS:
                    result[key] = self.redact_json(
                        value, sid, depth + 1,
                        in_user_content=True, assistant_authored=child_assistant,
                    )
                else:
                    result[key] = self.redact_json(
                        value, sid, depth + 1, in_user_content, child_assistant,
                    )
            return result
        return obj


# ── Module convenience ──────────────────────────────────────────────────


def get_redactor(min_score: float = 0.7, fail_closed: bool = True) -> PIIRedactor:
    # min_score / fail_closed honored only on first construction (singleton).
    if PIIRedactor._instance is None:
        PIIRedactor._instance = PIIRedactor(min_score=min_score, fail_closed=fail_closed)
    return PIIRedactor._instance


__all__ = [
    "PIIRedactor",
    "get_redactor",
    "RedactionUnavailable",
    "PII_ENTITIES",
]
