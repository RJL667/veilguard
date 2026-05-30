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
_META_KEYS = {"_skip_pii", "_vg_id", "_vg_immutable", "_vg_pii"}


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
]


def _load_allow_list() -> list[str]:
    """Load brand/tech allow_list so Presidio doesn't redact `Docker`,
    `LibreChat`, etc. as PERSON.  spaCy en_core_web_lg false-positives
    those at ~0.85 confidence.

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
            logger.info(f"[pii] loaded {len(terms)} allow_list terms from {p}")
            return terms
    logger.warning("[pii] no allow_list.txt found — brand names may false-positive")
    return []


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
        self.store = _get_store()

        # [PII_BLOCK_CACHE_2026_05_29]  THE blazingly-fast layer.
        # Key: (sid.root(), blake2b(text)).  Value: fully-redacted text.
        # Token assignment is deterministic + append-only per session, so
        # the redacted form of a block is a pure function of (text, sid):
        # once we've redacted these exact bytes for this session, replay
        # them with ZERO Presidio / line-split / token-mint / Lance I/O.
        # Frozen memory tiers + past `[Memory index=N]` fragments are
        # byte-stable across turns → permanent hits.  A miss just recomputes
        # (never wrong), so LRU eviction is free.  See PII_FAST_REDACTION_SPEC.md.
        self._block_cache: "OrderedDict[tuple, str]" = OrderedDict()
        self._block_cache_max = int(os.environ.get("PII_BLOCK_CACHE_MAX", "2048"))
        self._block_hits = 0
        self._block_misses = 0
        self._clean_skipped = 0
        # CLEAN-skip is provenance-anchored, NOT heuristic: we only skip
        # blocks that EXACTLY match a known system-authored static template
        # (magic prefix, Veilguard preamble).  Anything user-derived is
        # always scanned.  Toggle off for paranoid/audit runs.
        self._clean_skip = os.environ.get(
            "PII_CLEAN_SKIP", "1"
        ).lower() in ("1", "true", "yes", "on")
        self._clean_fingerprints = self._load_clean_fingerprints()

        logger.info(
            f"[pii] redactor ready (Presidio + SA recognizers + "
            f"{len(self.allow_list)} allow_list terms; "
            f"fail_closed={self._fail_closed} clean_skip={self._clean_skip} "
            f"block_cache_max={self._block_cache_max})"
        )

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
    def _block_key(sid: SessionId, text: str) -> tuple:
        h = hashlib.blake2b(text.encode("utf-8", "replace"), digest_size=16).hexdigest()
        r = sid.root()
        return (r.tenant_id, r.conv_id, h)

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

        [LINE_CACHE_REMOVED_2026_05_29]  The per-line analyzer cache was
        deleted: TCMM memory blocks are immutable, so a block is either a
        block-cache HIT (already redacted, never re-analyzed) or BRAND NEW
        (every line is new anyway → line caching bought nothing).  The
        block-output cache (`_block_cache`) is the only layer that earns
        its keep.  One analyze() per new block — simpler and correct.

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

        try:
            results = self._analyze(text)
            if not results:
                return text

            # Don't re-redact our own REF tokens.
            results = [
                r for r in results
                if not text[r.start:r.end].startswith("REF_")
            ]
            if not results:
                return text

            # Replace end-to-start so later positions aren't shifted.
            results.sort(key=lambda r: r.start, reverse=True)

            redacted = text
            for r in results:
                original = text[r.start:r.end]
                token = self.store.add_mapping(sid, r.entity_type, original)
                redacted = redacted[:r.start] + token + redacted[r.end:]
            # [PII_BATCH_WRITE_2026_05_29]  Persist all the tokens minted
            # for this text in ONE Lance write (add_mapping buffered them)
            # instead of one write per PII span.  Flush BEFORE returning
            # so the mappings are committed well before rehydration (which
            # happens in a later request after the LLM responds).
            try:
                self.store.flush()
            except Exception:
                pass
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

    def _classify(self, blk: dict) -> str:
        """CLEAN | FROZEN | VOLATILE for a text block.

        Tier-2 (TCMM-emitted `_vg_*`) is authoritative when present;
        Tier-1 falls back to provenance fingerprints + the block cache's
        own self-validating hash (handled in redact_render_blocks).
        """
        pii = blk.get("_vg_pii")
        if pii == "clean":
            return "CLEAN"
        text = blk.get("text", "")
        if isinstance(text, str) and self._is_clean(text):
            return "CLEAN"
        # FROZEN vs VOLATILE is decided by the cache (hash hit) in
        # redact_render_blocks; the explicit `_vg_immutable` only biases
        # logging.  Both go through the cache path safely.
        return "FROZEN" if blk.get("_vg_immutable") else "VOLATILE"

    def redact_render_blocks(
        self, blocks: list[dict], sid: Union[SessionId, str]
    ) -> list[dict]:
        """THE fast entry point for a TCMM-rendered system-block list.

        Per block:
          CLEAN     → emit verbatim (never scanned).  Provenance-anchored.
          else      → block-output cache lookup by (sid, hash(text)).
                      HIT  → replay redacted bytes (zero Presidio/Lance).
                      MISS → redact_text (line-cache + memo + batch write),
                             store, emit.
        Strips `_skip_pii`/`_vg_*`.  Preserves cache_control + structure.
        Non-text blocks pass through unchanged.

        Fail-closed propagates from redact_text on a real analyzer failure.
        """
        sid = self._coerce_sid(sid)
        out: list[dict] = []
        for blk in blocks:
            if not isinstance(blk, dict):
                out.append(blk)
                continue
            if blk.get("type") != "text":
                out.append(blk)
                continue
            text = blk.get("text", "")
            if not isinstance(text, str):
                out.append(self._emit_block(blk))
                continue
            # Legacy sentinel: caller guaranteed PII-free.
            if blk.get("_skip_pii") is True or self._classify(blk) == "CLEAN":
                self._clean_skipped += 1
                out.append(self._emit_block(blk))
                continue
            key = self._block_key(sid, text)
            cached = self._block_cache.get(key)
            if cached is not None:
                self._block_hits += 1
                self._block_cache.move_to_end(key)        # LRU touch
                out.append(self._emit_block(blk, cached))
                continue
            self._block_misses += 1
            redacted = self.redact_text(text, sid)
            self._block_cache[key] = redacted
            if len(self._block_cache) > self._block_cache_max:
                self._block_cache.popitem(last=False)     # evict LRU
            out.append(self._emit_block(blk, redacted))
        self._log_block_stats()
        return out

    def _log_block_stats(self) -> None:
        total = self._block_hits + self._block_misses
        if total and total % 50 == 0:
            pct = self._block_hits / total * 100
            logger.info(
                "[pii] block_cache hits=%d misses=%d (%.0f%%)  "
                "clean_skipped=%d  size=%d/%d",
                self._block_hits, self._block_misses, pct,
                self._clean_skipped, len(self._block_cache),
                self._block_cache_max,
            )

    def redact_blocks(self, blocks: list[dict], sid: Union[SessionId, str]) -> list[dict]:
        """Redact each text block in place-shape, preserving structure.

        Each block is a dict like:
          {"type": "text", "text": "...", "cache_control": {...}}

        ONLY the `.text` field is mutated.  Every other key — type,
        cache_control, etc. — is copied through unchanged (metadata keys
        stripped).  This is the cache-stability contract.

        Non-text blocks (images, tool_use, tool_result) are returned
        unchanged.  Delegates to redact_render_blocks so the block cache +
        clean-skip apply here too.
        """
        return self.redact_render_blocks(blocks, sid)

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
