"""DirectorAgent — the reactive orchestrator + proactive pre-evaluator.

Director's job (from MULTI_AGENT_PLATFORM.md §4.1):
  - Reactive stream: takes user messages, routes to ICs via create_task,
    answers solo from recall when possible, synthesizes when ICs return
  - Proactive stream: pre-evaluates dream-cycle proposals via a Haiku
    rank-pass before they hit the user's decision-ledger queue

Persona contract (agents/director.md):
  Model: reactive=claude-sonnet-4-5, rank_pass=claude-haiku-4-5
  Tools: create_task / assign_task / consult / final_synthesis,
         rank_proposals / convert_proposal / shelve_proposal /
         surface_org_memory_candidate, read_constitution / recall
  Role:  director  (only one per tenant; recall scope includes
         conv + team/knowledge, NEVER private agent memory)

This class is intentionally thin today.  As Phase 3 (proactive stream)
lands, the rank-pass entry point will live here as a separate method
that uses persona.model_for("rank_pass") instead of the default
reactive model.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .base import Agent

logger = logging.getLogger("veilguard.agent.director")


# Phase 6.11 — per-method latency telemetry.  Populated by `route`,
# `synthesize`, `propose` on every invocation.  Phase 8 split trigger
# reads this to decide when to break Director into Router+Synthesis
# personas: "p95 latency > 2× slowest specialist" / "Director prompt >
# 8k structured tokens".
DIRECTOR_METHOD_LATENCY_MS: dict[str, list[float]] = {
    "route":      [],
    "synthesize": [],
    "propose":    [],
}


def _record_latency(method: str, elapsed_ms: float) -> None:
    """Keep the last 1000 samples per method for p95 calc.  Lossy under
    sustained load is acceptable — this is telemetry, not audit."""
    buf = DIRECTOR_METHOD_LATENCY_MS.setdefault(method, [])
    buf.append(elapsed_ms)
    if len(buf) > 1000:
        del buf[: len(buf) - 1000]


def director_p95_ms(method: str) -> float | None:
    """Return p95 latency in ms for the given method, or None if no data."""
    buf = DIRECTOR_METHOD_LATENCY_MS.get(method) or []
    if not buf:
        return None
    s = sorted(buf)
    idx = int(len(s) * 0.95)
    idx = min(idx, len(s) - 1)
    return s[idx]


class DirectorAgent(Agent):
    """Tier-0 orchestrator.

    Phase 6.11 — exposes three distinct methods that map to three
    cognitive modes.  Today all three call into the same underlying
    persona / model.  Phase 8 split (Q21) replaces `route` with a
    Haiku-class Router persona and keeps `synthesize` / `propose` on
    Sonnet+ — the interface is unchanged so the split is mechanical.

    Trigger conditions for the actual Phase 8 split:
      - Director prompt > 8k structured tokens per turn (p95), OR
      - Director p95 latency > 2× slowest specialist (read from
        DIRECTOR_METHOD_LATENCY_MS), OR
      - Synthesis-correctness errors in eval (sub-agent findings lost
        in Director synthesis).
    """

    # Phase 3 hook: when the proactive pre-eval method lands, it'll
    # construct an adapter with model=persona.model_for("rank_pass")
    # (Haiku) instead of the default reactive (Sonnet).  For now the
    # base Agent.model() returns the reactive variant, which is what
    # we want for Director's user-facing turns.

    def _emit_anchor(
        self,
        *,
        decision_point: str,
        signal: dict[str, Any] | None = None,
        task_id: str | None = None,
        extras: dict[str, Any] | None = None,
    ) -> None:
        """Phase 8.0 — record a SnapshotAnchor at this decision point.

        Best-effort: anchors are observability, not a correctness gate.
        On any failure (no tenant context, replay package unavailable,
        ledger write error) we swallow the exception so the Director
        decision itself is never blocked.
        """
        try:
            from app.middleware import tenant  # type: ignore
            from app.replay import compute_anchor, record_anchor  # type: ignore
        except ImportError:
            return
        ctx = tenant.current()
        if ctx is None:
            return
        try:
            anchor = compute_anchor(
                decision_point=decision_point,
                agent_id=ctx.agent_id or "director",
                tenant_id=ctx.tenant_id,
                user_id=ctx.user_id,
                # Substrate inputs — Phase 8.0 ships with conservative
                # references; the replay harness will fill these out
                # with concrete substrate snapshots when it lands.
                task_graph=signal or {},
                approvals=[],
                tool_outputs={},
                constitution_version=0,
                tcmm_snapshot_ref=str(ctx.conversation_id or ""),
                alignment_weights_ver=0,
                dream_graph_signals_ref="",
                extras=extras or {},
            )
            record_anchor(anchor=anchor, task_id=task_id)
        except Exception:
            # Phase 8.0 contract — never block the decision.
            return

    async def route(self, signal: dict[str, Any]) -> dict[str, Any]:
        """Pure routing: which persona owns this work?  Sub-1k-token call.

        Phase 6.11 — same persona / model as `synthesize` and `propose`;
        the interface separation lets Phase 8 swap in a Haiku Router
        without touching call sites.  Emits per-method telemetry.

        Phase 8.0 — also emits a SnapshotAnchor so the replay harness
        can later detect "same substrate state at this decision point".

        `signal` carries enough context to make a routing decision:
        {brief, signal_type, originating_agent_id, recent_topics, ...}.
        Returns {target_persona, rationale_one_line}.
        """
        t0 = time.monotonic()
        try:
            self._emit_anchor(
                decision_point="director.route",
                signal=signal,
                extras={"signal_keys": sorted(signal.keys()) if signal else []},
            )
            return {
                "target_persona": "researcher",  # placeholder
                "rationale_one_line": (
                    "Phase 6.11 stub — actual routing happens in runtime.py"
                ),
            }
        finally:
            _record_latency("route", (time.monotonic() - t0) * 1000.0)

    async def synthesize(
        self,
        task_id: str,
        child_outputs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Final synthesis: collect, summarise, decide.  ≤8k-token call.

        Phase 8 split keeps this method on Sonnet/Opus.  See class
        docstring for trigger conditions.

        Phase 8.0 — emits a SnapshotAnchor scoped to the task so the
        replay harness can reproduce this synthesis call against a
        frozen snapshot.
        """
        t0 = time.monotonic()
        try:
            self._emit_anchor(
                decision_point="director.synthesize",
                signal={"task_id": task_id,
                        "child_outputs_count": len(child_outputs)},
                task_id=task_id,
                extras={"child_outputs_count": len(child_outputs)},
            )
            return {
                "task_id": task_id,
                "deliverable_summary": (
                    "Phase 6.11 stub — actual synthesis happens through "
                    "the standard Agent.run_turn loop today"
                ),
                "child_outputs_count": len(child_outputs),
            }
        finally:
            _record_latency("synthesize", (time.monotonic() - t0) * 1000.0)

    async def propose(
        self,
        ranked_candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Proactive-stream surface: select N from ranked queue.

        Phase 8 split keeps this method on Sonnet.

        Phase 8.0 — emits a SnapshotAnchor capturing the ranked
        candidate set so the replay harness can detect "same proactive
        queue → expect same selection".
        """
        t0 = time.monotonic()
        try:
            # Anchor over the candidate ids only — full payloads inflate
            # the hash without changing the proactive-decision identity.
            cand_ids = [c.get("id") for c in ranked_candidates]
            self._emit_anchor(
                decision_point="director.propose",
                signal={"candidate_ids": cand_ids,
                        "count": len(ranked_candidates)},
                extras={"candidate_ids": cand_ids},
            )
            # Phase 6.11 stub: pass through top-N, no LLM call yet.
            return list(ranked_candidates[:3])
        finally:
            _record_latency("propose", (time.monotonic() - t0) * 1000.0)


__all__ = [
    "DirectorAgent",
    "DIRECTOR_METHOD_LATENCY_MS",
    "director_p95_ms",
]
