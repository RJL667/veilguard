"""Phase 6.0 — Mechanical AC executors with 3-state result + evidence hashes.

Phase 7.5 added `llm_verify` (2026-05-28) under the **paired-mechanical
rule**: it's a *quality lift* over mechanical checks, never a sole gate.
The Director-side validator (`build_acceptance_critic_payload`)
enforces that any task with an `llm_verify` AC must also have at least
one mechanical required AC.  The iron rule survives — LLM is allowed
to fail tasks that mechanical checks let through, but is forbidden
to *clear* tasks the mechanical layer hasn't already cleared.

Per spec §6.0 and `tests/PHASE_6_ACCEPTANCE.md`:

  Eight kinds total — seven mechanical + one user-gate.  Plus
  `llm_verify` (Phase 7.5) which sits OUTSIDE the mechanical set and
  is gated by the pairing rule above.

  Every executor returns a `CheckResult` with:
    - status: 'pass' | 'fail' | 'error'  (3-state, AC-27)
        pass  = check ran successfully and the artifact meets the AC
        fail  = check ran successfully but the artifact does NOT meet the AC
        error = the check itself failed to execute (cmd not on PATH,
                file unreadable, regex compile error, timeout, ...)
        `fail` and `error` BOTH block the gate.  Distinction is for
        diagnostics — `fail` means "artifact wrong"; `error` means
        "we couldn't verify."
    - evidence: dict carrying hashes / counts / exit codes — verifiable
        on re-check by the critic to detect partial-write / corruption
        (AC-26: builder partial-writes, retry passes a DIFFERENT artifact
        under the same path → evidence mismatch catches it).
    - reason: ≤2 KB human-readable explanation; required on fail/error.

  Iron rule (Phase 6.0.2 + AC-4): mechanical kinds are the only
  acceptable required ACs.  `manual_user` is allowed only as advisory
  (`required=True` AC with `check_kind=manual_user` is rejected by the
  Director-side validator).

The executor side is intentionally side-effect-free except for
`test_passes` (which shells out under a sandbox restriction — AC-8).
Pure I/O lives in the runner that calls these; the executors decide
PASS/FAIL/ERROR + emit evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

# ── Public constants ────────────────────────────────────────────────────

# Eight kinds: seven mechanical + manual_user + llm_verify (Phase 7.5).
# Note that llm_verify is in CHECK_KINDS (executable) but NOT in
# MECHANICAL_CHECK_KINDS (cannot satisfy the Director's "≥1 required
# mechanical AC" rule).  That keeps the iron rule intact while letting
# llm_verify be used as a quality lift.
CHECK_KINDS: frozenset[str] = frozenset({
    "claim_count",
    "claim_predicate",
    "output_path_exists",
    "output_path_matches_regex",
    "output_path_jsonschema",
    "test_passes",
    "word_count_range",   # [AC_WORD_COUNT_2026-06-05] deterministic word-count gate
    "manual_user",
    "llm_verify",     # Phase 7.5 — quality lift only; pairing rule enforced
})

# Director's `create_task` validator requires ≥1 required AC from this set.
# (AC-4: 7 — manual_user and llm_verify are excluded; they cannot be a
# task's only required AC.)
MECHANICAL_CHECK_KINDS: frozenset[str] = frozenset({
    "claim_count",
    "claim_predicate",
    "output_path_exists",
    "output_path_matches_regex",
    "output_path_jsonschema",
    "test_passes",
    "word_count_range",   # [AC_WORD_COUNT_2026-06-05]
})

# AC-8 — `test_passes` sandbox bounds.  Set once at import time; the
# runtime resolves to an absolute path so subprocess can validate.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_TEST_TIMEOUT_S = 60


# ── CheckResult ─────────────────────────────────────────────────────────


CheckStatus = Literal["pass", "fail", "error"]


@dataclass
class CheckResult:
    """Output of one AC executor.

    Three-state per AC-27.  `evidence` is the canonical re-verification
    payload — critic re-runs the same check and compares evidence to
    detect builder partial-write / retry-different-content corruption
    (AC-26).
    """
    status: CheckStatus
    evidence: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status":   self.status,
            "evidence": self.evidence,
            "reason":   self.reason,
        }


# ── Executor registry ──────────────────────────────────────────────────

Executor = Callable[[dict[str, Any], dict[str, Any]], CheckResult]
# Signature: executor(check_args, context) -> CheckResult
# `context` carries optional task_id, workspace_root, claim_log, etc.


EXECUTOR_REGISTRY: dict[str, Executor] = {}


def _register(kind: str) -> Callable[[Executor], Executor]:
    def deco(fn: Executor) -> Executor:
        if kind not in CHECK_KINDS:
            raise ValueError(
                f"refusing to register executor for unknown kind {kind!r}; "
                f"add to CHECK_KINDS first"
            )
        EXECUTOR_REGISTRY[kind] = fn
        return fn
    return deco


# ── Shared helpers ─────────────────────────────────────────────────────


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_path(path_str: str, workspace_root: Path | None = None) -> Path:
    """Resolve a check-arg path; reject anything escaping the sandbox.

    AC-8 — `test_passes` cwd_must_be_under(REPO_ROOT).  Applied to all
    path-touching executors so a check_args.path = "../../../etc/passwd"
    can't read arbitrary files.
    """
    # [SAFE_PATH_STR_COERCE_2026_05_29]  Callers (e.g. review_decision's
    # AC-check wiring) legitimately pass workspace_root as a STRING from
    # the env (`VEILGUARD_WORKSPACE_ROOT`).  `str / str` raises
    # TypeError → the executor returned `error` → required AC never
    # `pass` → review_decision auto-downgraded → task bounced forever.
    # Coerce to Path so str and Path callers both work.
    base = Path(workspace_root) if workspace_root else _REPO_ROOT
    candidate = (base / path_str).resolve()
    try:
        candidate.relative_to(base.resolve())
    except ValueError:
        raise PermissionError(
            f"path {path_str!r} resolves outside sandbox root {base}"
        )
    return candidate


# ── Executors ──────────────────────────────────────────────────────────


@_register("output_path_exists")
def _exec_output_path_exists(args: dict, ctx: dict) -> CheckResult:
    """File at `path` exists and is ≥ `min_bytes` (default 1).

    AC-7 anti-false-positive: 0-byte file with `min_bytes=1` → fail,
    not pass.
    """
    try:
        path_str = args.get("path")
        if not path_str:
            return CheckResult("error", reason="check_args.path required")
        min_bytes = int(args.get("min_bytes", 1))
        path = _safe_path(path_str, ctx.get("workspace_root"))
        if not path.exists():
            return CheckResult(
                "fail",
                evidence={"path": str(path_str), "exists": False},
                reason=f"file does not exist: {path_str!r}",
            )
        size = path.stat().st_size
        if size < min_bytes:
            return CheckResult(
                "fail",
                evidence={
                    "path": str(path_str), "exists": True,
                    "size_bytes": size, "min_bytes": min_bytes,
                },
                reason=(
                    f"file exists but {size} bytes < min_bytes={min_bytes} "
                    "(likely stub / partial write)"
                ),
            )
        return CheckResult(
            "pass",
            evidence={
                "path": str(path_str),
                "size_bytes": size,
                "path_sha256": _sha256_file(path),  # AC-26 evidence hash
            },
        )
    except PermissionError as e:
        return CheckResult("error", reason=str(e))
    except Exception as e:
        return CheckResult("error", reason=f"unexpected: {type(e).__name__}: {e}")


@_register("output_path_matches_regex")
def _exec_output_path_matches_regex(args: dict, ctx: dict) -> CheckResult:
    """File exists AND its contents match `pattern` (re.search semantics).

    AC-7 — empty file with regex `.*` returns `fail`, not `pass`.  We
    enforce content-length > 0 in addition to the regex match so a
    trivially-matching pattern can't pass on a stub.
    """
    try:
        path_str = args.get("path")
        pattern = args.get("pattern")
        if not path_str or pattern is None:
            return CheckResult(
                "error",
                evidence={"ac_malformed": True},
                reason="check_args.path and check_args.pattern required",
            )
        flags_str = args.get("flags", "")
        flags = 0
        if "i" in flags_str.lower(): flags |= re.IGNORECASE
        if "m" in flags_str.lower(): flags |= re.MULTILINE
        if "s" in flags_str.lower(): flags |= re.DOTALL
        try:
            rx = re.compile(pattern, flags)
        except re.error as e:
            return CheckResult(
                "error",
                reason=f"pattern compile failed: {e}",
            )
        path = _safe_path(path_str, ctx.get("workspace_root"))
        if not path.exists():
            return CheckResult(
                "fail",
                evidence={"path": str(path_str), "exists": False},
                reason=f"file does not exist: {path_str!r}",
            )
        body = path.read_text(encoding="utf-8", errors="replace")
        # AC-7: explicit empty-file rejection regardless of regex.
        if not body.strip():
            return CheckResult(
                "fail",
                evidence={
                    "path": str(path_str),
                    "size_bytes": path.stat().st_size,
                },
                reason="file is empty / whitespace-only (anti-false-positive)",
            )
        m = rx.search(body)
        if not m:
            return CheckResult(
                "fail",
                evidence={
                    "path": str(path_str),
                    "pattern": pattern,
                    "body_sha256": _sha256_bytes(body.encode("utf-8")),
                },
                reason=f"pattern {pattern!r} not found in file",
            )
        return CheckResult(
            "pass",
            evidence={
                "path": str(path_str),
                "pattern": pattern,
                "match_span": [m.start(), m.end()],
                "body_sha256": _sha256_bytes(body.encode("utf-8")),
            },
        )
    except PermissionError as e:
        return CheckResult("error", reason=str(e))
    except Exception as e:
        return CheckResult("error", reason=f"unexpected: {type(e).__name__}: {e}")


@_register("word_count_range")
def _exec_word_count_range(args: dict, ctx: dict) -> CheckResult:
    """[AC_WORD_COUNT_2026-06-05] File exists and its word count is within
    [min_words, max_words] (inclusive).

    Word count = whitespace-delimited tokens in the file text — deterministic,
    so it replaces the LLM critic's non-deterministic counting (live-observed
    the same critic counting 118 then 165 words for one unchanged file).
    At least one of min/max is required; accepts several arg spellings the
    model tends to emit (min_words/min/minimum, max_words/max/maximum).
    """
    try:
        path_str = args.get("path")
        if not path_str:
            return CheckResult(
                "error",
                evidence={"ac_malformed": True},
                reason="check_args.path required",
            )

        def _as_int(*keys):
            for k in keys:
                v = args.get(k)
                if v is not None:
                    try:
                        return int(v)
                    except (TypeError, ValueError):
                        pass
            return None

        min_w = _as_int("min_words", "min", "min_count", "minimum")
        max_w = _as_int("max_words", "max", "max_count", "maximum")
        if min_w is None and max_w is None:
            return CheckResult(
                "error",
                evidence={"ac_malformed": True},
                reason="check_args needs min_words and/or max_words",
            )
        path = _safe_path(path_str, ctx.get("workspace_root"))
        if not path.exists():
            return CheckResult(
                "fail",
                evidence={"path": str(path_str), "exists": False},
                reason=f"file does not exist: {path_str!r}",
            )
        body = path.read_text(encoding="utf-8", errors="replace")
        if not body.strip():
            return CheckResult(
                "fail",
                evidence={"path": str(path_str), "size_bytes": path.stat().st_size},
                reason="file is empty / whitespace-only (anti-false-positive)",
            )
        words = len(body.split())
        ev = {
            "path": str(path_str),
            "word_count": words,
            "min_words": min_w,
            "max_words": max_w,
            "body_sha256": _sha256_bytes(body.encode("utf-8")),
        }
        if min_w is not None and words < min_w:
            return CheckResult("fail", evidence=ev, reason=f"word count {words} below min {min_w}")
        if max_w is not None and words > max_w:
            return CheckResult("fail", evidence=ev, reason=f"word count {words} above max {max_w}")
        return CheckResult("pass", evidence=ev)
    except PermissionError as e:
        return CheckResult("error", reason=str(e))
    except Exception as e:
        return CheckResult("error", reason=f"unexpected: {type(e).__name__}: {e}")


@_register("output_path_jsonschema")
def _exec_output_path_jsonschema(args: dict, ctx: dict) -> CheckResult:
    """File parses as JSON and validates against `schema` (JSONSchema).

    Parse failure → `error` (not `fail` — distinct per AC-27 because
    the artifact isn't analyzable, not just wrong).
    """
    try:
        path_str = args.get("path")
        schema = args.get("schema")
        if not path_str or schema is None:
            return CheckResult(
                "error",
                reason="check_args.path and check_args.schema required",
            )
        path = _safe_path(path_str, ctx.get("workspace_root"))
        if not path.exists():
            return CheckResult(
                "fail",
                evidence={"path": str(path_str), "exists": False},
                reason=f"file does not exist: {path_str!r}",
            )
        body = path.read_text(encoding="utf-8", errors="replace")
        try:
            data = json.loads(body)
        except json.JSONDecodeError as e:
            return CheckResult(
                "error",
                evidence={"path": str(path_str), "parse_error": str(e)},
                reason=f"JSON parse failed: {e}",
            )
        # Lazy import — jsonschema is optional dep.
        try:
            import jsonschema  # type: ignore
        except ImportError:
            return CheckResult(
                "error",
                reason="jsonschema not installed; "
                "add to agent-runtime/requirements.txt",
            )
        try:
            jsonschema.validate(data, schema)
        except jsonschema.ValidationError as e:
            return CheckResult(
                "fail",
                evidence={
                    "path": str(path_str),
                    "validation_error": e.message,
                    "validation_path": list(e.absolute_path),
                },
                reason=f"jsonschema validation failed: {e.message}",
            )
        return CheckResult(
            "pass",
            evidence={
                "path": str(path_str),
                "body_sha256": _sha256_bytes(body.encode("utf-8")),
            },
        )
    except PermissionError as e:
        return CheckResult("error", reason=str(e))
    except Exception as e:
        return CheckResult("error", reason=f"unexpected: {type(e).__name__}: {e}")


@_register("test_passes")
def _exec_test_passes(args: dict, ctx: dict) -> CheckResult:
    """Run `cmd` under `cwd` and assert `expect_exit` (default 0).

    AC-8 — sandbox: cwd MUST resolve under _REPO_ROOT.  Timeout default
    60s; cap at 600s to bound runaway tasks.  Timeout → `error` (not
    `fail`).  Exit 127 (cmd not found) → `error`.
    """
    try:
        cmd = args.get("cmd") or args.get("command")
        if not cmd:
            return CheckResult(
                "error",
                reason="check_args.cmd (or .command) required",
            )
        cwd_str = args.get("cwd", ".")
        timeout_s = min(int(args.get("timeout_s", _DEFAULT_TEST_TIMEOUT_S)), 600)
        expect_exit = int(args.get("expect_exit", 0))
        cwd = _safe_path(cwd_str, ctx.get("workspace_root"))
        if not cwd.is_dir():
            return CheckResult(
                "error",
                reason=f"cwd {cwd_str!r} is not a directory",
            )
        try:
            proc = subprocess.run(
                cmd if isinstance(cmd, list) else cmd,
                shell=isinstance(cmd, str),
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            return CheckResult(
                "error",
                evidence={"cmd": str(cmd), "timeout_s": timeout_s},
                reason=f"command timed out after {timeout_s}s",
            )
        except FileNotFoundError as e:
            return CheckResult(
                "error",
                evidence={"cmd": str(cmd)},
                reason=f"command not found: {e}",
            )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        stdout_hash = _sha256_bytes(stdout.encode("utf-8"))
        # "Command not found" detection — needs to handle both POSIX
        # (exit 127, "command not found" in stderr) and Windows
        # (cmd.exe returns exit 1 with "is not recognized as an internal
        # or external command" in stderr).  Treat both as `error` (the
        # check couldn't execute), not `fail` (which would imply the
        # tests ran and disagreed).
        stderr_low = stderr.lower()
        is_not_found = (
            proc.returncode == 127
            or "is not recognized as an internal or external" in stderr_low
            or "command not found" in stderr_low
            or "no such file or directory" in stderr_low and "exec" in stderr_low
        )
        if is_not_found:
            return CheckResult(
                "error",
                evidence={
                    "cmd": str(cmd), "exit_code": proc.returncode,
                    "stderr": stderr[-2000:],
                },
                reason=(
                    f"command not found (exit {proc.returncode}); "
                    f"stderr tail: {stderr[-150:]!r}"
                ),
            )
        if proc.returncode != expect_exit:
            return CheckResult(
                "fail",
                evidence={
                    "cmd":         str(cmd),
                    "exit_code":   proc.returncode,
                    "expect_exit": expect_exit,
                    "stdout_hash": stdout_hash,
                    "stderr":      stderr[-2000:],
                },
                reason=(
                    f"exit {proc.returncode} (expected {expect_exit}); "
                    f"stderr tail: {stderr[-200:]!r}"
                ),
            )
        return CheckResult(
            "pass",
            evidence={
                "cmd":         str(cmd),
                "exit_code":   proc.returncode,
                "stdout_hash": stdout_hash,
            },
        )
    except PermissionError as e:
        return CheckResult("error", reason=str(e))
    except Exception as e:
        return CheckResult("error", reason=f"unexpected: {type(e).__name__}: {e}")


@_register("claim_count")
def _exec_claim_count(args: dict, ctx: dict) -> CheckResult:
    """Count claims in `ctx['claims']` matching `predicate`, compare to `n`.

    `predicate` is a JSON-path-ish dict like `{"category": "finding"}`;
    we do shallow key-equality matching (deliberately limited — richer
    predicates go through `claim_predicate`).

    AC-7 — `op='>='` with `n=1` on empty claim list → fail, not pass.
    """
    try:
        predicate = args.get("predicate") or {}
        op = args.get("op", ">=")
        n = int(args.get("n", 1))
        claims = ctx.get("claims") or []
        if not isinstance(claims, list):
            return CheckResult("error", reason="ctx.claims must be a list")
        if op not in ("==", ">=", "<=", "<", ">"):
            return CheckResult("error", reason=f"unknown op {op!r}")

        def _match(claim: dict) -> bool:
            if not isinstance(claim, dict): return False
            if isinstance(predicate, str):
                # Treat string predicate as substring search over body.
                return predicate in str(claim.get("body", ""))
            if isinstance(predicate, dict):
                for k, v in predicate.items():
                    if claim.get(k) != v:
                        return False
                return True
            return False

        matched = sum(1 for c in claims if _match(c))
        ops = {
            "==": matched == n, ">=": matched >= n,
            "<=": matched <= n, "<":  matched <  n,
            ">":  matched >  n,
        }
        if ops[op]:
            return CheckResult(
                "pass",
                evidence={
                    "matched": matched, "expected": f"{op} {n}",
                    "predicate_match_count": matched,
                },
            )
        return CheckResult(
            "fail",
            evidence={
                "matched": matched, "expected": f"{op} {n}",
                "predicate_match_count": matched,
            },
            reason=f"matched={matched}; expected {op} {n}",
        )
    except Exception as e:
        return CheckResult("error", reason=f"unexpected: {type(e).__name__}: {e}")


@_register("claim_predicate")
def _exec_claim_predicate(args: dict, ctx: dict) -> CheckResult:
    """≥1 claim in `ctx['claims']` must match the predicate.

    Predicate accepts the same shapes as `claim_count`'s `predicate` —
    shallow dict-equality or substring on body.  `must_exist=True`
    (default) means at least one match; `must_exist=False` means none.
    """
    try:
        predicate = args.get("predicate")
        must_exist = bool(args.get("must_exist", True))
        if predicate is None:
            return CheckResult("error", reason="check_args.predicate required")
        claims = ctx.get("claims") or []
        if not isinstance(claims, list):
            return CheckResult("error", reason="ctx.claims must be a list")

        def _match(claim: dict) -> bool:
            if not isinstance(claim, dict): return False
            if isinstance(predicate, str):
                return predicate in str(claim.get("body", ""))
            if isinstance(predicate, dict):
                for k, v in predicate.items():
                    if claim.get(k) != v:
                        return False
                return True
            return False

        matched = sum(1 for c in claims if _match(c))
        ok = (matched > 0) if must_exist else (matched == 0)
        if ok:
            return CheckResult(
                "pass",
                evidence={"matched": matched, "must_exist": must_exist},
            )
        return CheckResult(
            "fail",
            evidence={"matched": matched, "must_exist": must_exist},
            reason=(
                f"must_exist={must_exist} but matched={matched}"
            ),
        )
    except Exception as e:
        return CheckResult("error", reason=f"unexpected: {type(e).__name__}: {e}")


@_register("llm_verify")
def _exec_llm_verify(args: dict, ctx: dict) -> CheckResult:
    """Phase 7.5 — LLM-judged acceptance criterion (quality lift only).

    The Director-side validator pairs this with ≥1 mechanical required
    AC on the same task (`build_acceptance_critic_payload` rejects an
    `llm_verify`-only task).  Within the gate, this check is treated
    identically to a mechanical one: `pass` clears, `fail` blocks,
    `error` blocks with diagnostic.

    `args`:
        rubric        — required, str.  Plain-language statement the
                        LLM must adjudicate against the deliverable.
                        Example: "The memo is grounded in the cited
                        source and does not paraphrase claims absent
                        from it."
        target_path   — optional, str.  Workspace-relative path to the
                        deliverable being judged.  When set, the
                        executor reads (size-capped) and passes the
                        content to the LLM as `<artifact>...</artifact>`.
        target_text   — optional, str.  Direct content to judge.  Either
                        target_path OR target_text must be set; if both,
                        target_path wins.
        model         — optional, str.  Override the default judge model
                        (`claude-haiku-4-5`).  Per `architecture_token_
                        accounting.md` Haiku is the cost-correct judge
                        for short rubrics.

    `ctx` may carry:
        workspace_root — Path for sandboxed read of target_path.
        llm_judge      — callable(prompt: str, model: str) -> dict with
                         keys {verdict: 'pass'|'fail', confidence:
                         float[0,1], reason: str}.  Tests inject a stub.
                         Production wires this in `run_check` callers
                         (Critic dispatch in inbox_poller).

    Evidence carries the judge's verdict, confidence, rubric hash, and
    the artifact's sha256 — so re-running the check on the same
    deliverable detects builder partial-write / retry-different-content
    corruption (AC-26 evidence-mismatch principle).

    Cost guard: artifact content is capped at 50 KB before the prompt
    is built — beyond that the executor returns `error` rather than
    blow out a budget.  Set `args.allow_large=True` to bypass for
    deliberate large-doc reviews.
    """
    rubric = (args.get("rubric") or "").strip()
    if not rubric:
        return CheckResult(
            "error", reason="llm_verify requires non-empty `rubric`"
        )
    if len(rubric) > 4000:
        return CheckResult(
            "error",
            reason=f"rubric too long ({len(rubric)} chars > 4000); "
            f"split into multiple narrower llm_verify ACs",
        )

    # Resolve artifact content.
    target_path = args.get("target_path")
    target_text = args.get("target_text")
    allow_large = bool(args.get("allow_large"))
    artifact_text: str
    artifact_source: str
    if target_path:
        try:
            ws_root = ctx.get("workspace_root")
            ws_path = Path(ws_root) if ws_root else None
            p = _safe_path(target_path, ws_path)
        except PermissionError as e:
            return CheckResult(
                "error",
                reason=f"target_path sandbox violation: {e}",
                evidence={"target_path": target_path},
            )
        if not p.exists():
            return CheckResult(
                "fail",
                reason=f"target_path does not exist: {target_path}",
                evidence={"target_path": target_path},
            )
        try:
            artifact_bytes = p.read_bytes()
        except OSError as e:
            return CheckResult(
                "error", reason=f"target_path read failed: {e}",
                evidence={"target_path": target_path},
            )
        if len(artifact_bytes) > 50_000 and not allow_large:
            return CheckResult(
                "error",
                reason=(
                    f"target_path size {len(artifact_bytes)} bytes > 50KB "
                    f"cap; pass allow_large=True to override"
                ),
                evidence={"target_path": target_path,
                          "size_bytes": len(artifact_bytes)},
            )
        try:
            artifact_text = artifact_bytes.decode("utf-8", errors="replace")
        except Exception as e:
            return CheckResult(
                "error", reason=f"target_path decode failed: {e}",
            )
        artifact_source = f"file:{target_path}"
        artifact_sha    = _sha256_bytes(artifact_bytes)
    elif target_text is not None:
        if not isinstance(target_text, str):
            return CheckResult(
                "error",
                reason=f"target_text must be string, got {type(target_text).__name__}",
            )
        if len(target_text) > 50_000 and not allow_large:
            return CheckResult(
                "error",
                reason=f"target_text length {len(target_text)} > 50KB cap; "
                f"pass allow_large=True to override",
            )
        artifact_text   = target_text
        artifact_source = "inline"
        artifact_sha    = _sha256_bytes(target_text.encode("utf-8"))
    else:
        return CheckResult(
            "error",
            reason="llm_verify requires target_path OR target_text",
        )

    # Acquire judge.
    judge = ctx.get("llm_judge")
    if judge is None:
        return CheckResult(
            "error",
            reason=(
                "no llm_judge in context — llm_verify cannot run.  "
                "Caller (Critic dispatch) must inject a callable "
                "ctx['llm_judge'](prompt, model) -> {verdict, confidence, reason}."
            ),
            evidence={"rubric_sha": _sha256_bytes(rubric.encode("utf-8"))[:16]},
        )

    model = args.get("model") or "claude-haiku-4-5"

    prompt = _build_llm_verify_prompt(rubric=rubric, artifact=artifact_text)
    try:
        verdict_dict = judge(prompt, model)
    except Exception as e:
        return CheckResult(
            "error",
            reason=f"judge call raised: {type(e).__name__}: {e}",
            evidence={"model": model, "artifact_source": artifact_source},
        )

    if not isinstance(verdict_dict, dict):
        return CheckResult(
            "error",
            reason=f"judge returned {type(verdict_dict).__name__}, expected dict",
        )
    verdict = (verdict_dict.get("verdict") or "").lower().strip()
    if verdict not in ("pass", "fail"):
        return CheckResult(
            "error",
            reason=f"judge verdict must be 'pass' or 'fail', got {verdict!r}",
            evidence={"raw_verdict": verdict_dict},
        )
    try:
        confidence = float(verdict_dict.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    reason_text = (verdict_dict.get("reason") or "")[:1500]

    rubric_sha   = _sha256_bytes(rubric.encode("utf-8"))[:16]
    evidence = {
        "model":           model,
        "verdict":         verdict,
        "confidence":      round(confidence, 3),
        "rubric_sha":      rubric_sha,
        "artifact_sha":    artifact_sha[:16],
        "artifact_source": artifact_source,
    }

    if verdict == "pass":
        return CheckResult("pass", evidence=evidence, reason=reason_text)
    return CheckResult("fail", evidence=evidence, reason=reason_text)


def _build_llm_verify_prompt(*, rubric: str, artifact: str) -> str:
    """Build the structured judge prompt.

    Format constrains the LLM to a parseable JSON envelope so the
    caller's adapter (the `llm_judge` callable) can return a clean
    dict.  The rubric is presented in isolation so the LLM never
    mixes rubric text with artifact text.
    """
    return (
        "You are evaluating an artifact against a single acceptance "
        "criterion.  Return strict JSON with keys `verdict` "
        "('pass'|'fail'), `confidence` (0.0-1.0), and `reason` "
        "(≤2 sentences).  Be conservative: when in doubt, return "
        "'fail' with a concrete reason.\n\n"
        "<rubric>\n"
        f"{rubric}\n"
        "</rubric>\n\n"
        "<artifact>\n"
        f"{artifact}\n"
        "</artifact>\n\n"
        "Output JSON only."
    )


@_register("manual_user")
def _exec_manual_user(args: dict, ctx: dict) -> CheckResult:
    """Gates to user.  Critic emits `undecidable` mapped to `error`.

    The check itself never auto-passes.  Director must escalate to a
    real human via the approval gate; if unanswered, this AC blocks
    `state='done'` indefinitely (which is the intended belt-and-braces
    behavior — see AC-25).
    """
    question = args.get("question") or args.get("prompt") or "(no question)"
    return CheckResult(
        "error",  # never auto-pass; ledger gate treats `error` as block
        evidence={"question": question, "awaiting_user": True},
        reason=(
            f"manual_user check awaiting operator answer: {question!r}"
        ),
    )


# ── Public dispatch ────────────────────────────────────────────────────


def run_check(
    check_kind: str,
    check_args: dict[str, Any] | str,
    context: dict[str, Any] | None = None,
) -> CheckResult:
    """Dispatch one AC check and return its CheckResult.

    `check_args` may be a dict OR a JSON-encoded string (Lance stores
    them as strings).  Unknown `check_kind` → `error` (not exception).
    """
    if isinstance(check_args, str):
        try:
            check_args = json.loads(check_args) if check_args else {}
        except json.JSONDecodeError as e:
            return CheckResult(
                "error",
                evidence={"ac_malformed": True},
                reason=f"check_args JSON parse failed: {e}",
            )
    if not isinstance(check_args, dict):
        return CheckResult(
            "error",
            evidence={"ac_malformed": True},
            reason=f"check_args must be dict or JSON string, got {type(check_args).__name__}",
        )
    ctx = dict(context or {})  # copy so we can inject without aliasing
    executor = EXECUTOR_REGISTRY.get(check_kind)
    if executor is None:
        return CheckResult(
            "error",
            evidence={"ac_malformed": True},  # broken AC definition, not artifact
            reason=f"unknown check_kind {check_kind!r}; "
            f"known kinds: {sorted(EXECUTOR_REGISTRY.keys())}",
        )
    # [PHASE_7_5_LLM_JUDGE_WIRE_2026_05_28]  When llm_verify is
    # dispatched without an explicit `ctx['llm_judge']`, fall back to
    # the production default that wraps the Anthropic backend.  Tests
    # always inject their own judge, so this fallback only activates
    # in real Critic dispatch paths.  Module-level import is fine —
    # llm_judge.py lazy-imports the backend.
    if check_kind == "llm_verify" and "llm_judge" not in ctx:
        try:
            from .llm_judge import default_llm_judge
            ctx["llm_judge"] = default_llm_judge
        except Exception as e:
            return CheckResult(
                "error",
                reason=(
                    f"llm_verify dispatched without ctx['llm_judge'] and "
                    f"default judge unavailable: {type(e).__name__}: {e}"
                ),
            )
    return executor(check_args, ctx)
