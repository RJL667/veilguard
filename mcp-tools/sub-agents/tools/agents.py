"""Agent spawning tools — spawn_agent, parallel_agents, coordinate, pipeline, review_loop, spawn_agentic."""

import asyncio
import json
import logging
import uuid

from config import ROLES, DEFAULT_BACKEND, DEFAULT_MODEL, PLAN_MODELS, SCRATCHPAD_DIR
from core import state
from core.llm import call_llm, resolve_backend, compact_results
from core.agentic import agentic_loop, AGENT_TOOLS, EFFORT_LEVELS
from core.request_ctx import spawn_scope as _spawn_scope, capture_context as _capture_ctx
from utils.helpers import load_skill, maybe_offload
from utils.afk import touch_activity
from utils.tcmm import tcmm_recall
from storage.session import save_session, log_transcript

logger = logging.getLogger("veilguard.agents")

COORDINATOR_SYSTEM = """\
You are a mission coordinator at Phishield. Given a complex goal, you must:
1. Break it into 2-5 discrete subtasks
2. Assign each to the best specialist role
3. Specify what each worker should produce

Available specialist roles: analyst, threat, writer, coder, researcher, critic

You MUST respond with ONLY a JSON array. No markdown, no explanation.
Each object: {"task": "specific instruction", "role": "role_name"}
"""

RECONCILER_SYSTEM = """\
You are a senior analyst at Phishield. Reconcile all worker findings into a single coherent output.
Resolve contradictions, highlight key findings, produce a clear structured final report.
"""


def register(mcp):
    @mcp.tool()
    async def spawn_agent(task: str, role: str = "", system_prompt: str = "", model: str = "",
                          backend: str = "", memory: bool = False, skill: str = "") -> str:
        """Spawn a sub-agent for a focused task. Supports roles, skills, memory, and multi-backend."""
        touch_activity()
        if system_prompt:
            sys_prompt, role_name = system_prompt, "Custom Agent"
        elif role in ROLES:
            sys_prompt, role_name = ROLES[role]["system"], ROLES[role]["name"]
        else:
            return f"Error: Unknown role '{role}'. Available: {', '.join(ROLES.keys())}"

        if skill:
            skill_content = load_skill(skill)
            if skill_content:
                sys_prompt += f"\n\n--- SKILL: {skill} ---\n{skill_content}\n--- END SKILL ---"

        cfg = resolve_backend(backend)
        use_model = model if model else cfg["default_model"]
        backend_label = (backend or DEFAULT_BACKEND).lower()

        task_with_memory = task
        if memory:
            mem_ctx = await tcmm_recall(task)
            if mem_ctx:
                task_with_memory = f"{mem_ctx}\n\n--- CURRENT TASK ---\n{task}"

        # Spawn scope: child cid keeps the sub-agent's LLM iteration
        # out of the parent's namespace; lineage_parent_conv travels
        # in metadata so TCMM can wire the edge on ingest.
        async with _spawn_scope("agent"):
            result = await call_llm(sys_prompt, task_with_memory, model=use_model, backend=backend_label, role=role)
        result = maybe_offload(result, f"agent-{uuid.uuid4().hex[:6]}")
        return f"# Sub-Agent: {role_name} ({use_model} via {backend_label})\n\n{result}"

    @mcp.tool()
    async def parallel_agents(tasks: str) -> str:
        """Spawn multiple sub-agents in parallel and collect results."""
        try:
            task_list = json.loads(tasks)
        except json.JSONDecodeError:
            return "Error: 'tasks' must be a valid JSON array"

        async def run_one(idx, item):
            r = item.get("role", "")
            sp = item.get("system_prompt", "")
            m = item.get("model", DEFAULT_MODEL)
            t = item.get("task", "")
            if not t:
                return f"## Agent {idx+1}: Error — no task"
            prompt = sp or ROLES.get(r, {}).get("system", "You are a helpful assistant.")
            rn = ROLES.get(r, {}).get("name", r or "General")
            # Each parallel worker gets its OWN child cid so their
            # archive blocks land in distinct namespaces — otherwise
            # gather() would serialize them through one shared scope
            # and you'd lose the per-worker lineage.
            async with _spawn_scope(f"par{idx}"):
                res = await call_llm(prompt, t, model=m, role=r)
            return f"## Agent {idx+1}: {rn}\n\n{res}"

        results = await asyncio.gather(*[run_one(i, t) for i, t in enumerate(task_list)], return_exceptions=True)
        output = [f"# Parallel Agents — {len(task_list)} tasks\n"]
        for i, r in enumerate(results):
            output.append(str(r) if not isinstance(r, Exception) else f"## Agent {i+1}: Error\n{r}")
        return "\n---\n".join(output)

    @mcp.tool()
    async def coordinate(goal: str, max_workers: int = 5, backend: str = "", model: str = "",
                         plan_model: str = "", verify: bool = False, memory: bool = False) -> str:
        """Coordinate a complex task with ULTRAPLAN: Plan (strong) → Execute (fast, parallel) → Reconcile (strong) → Verify (optional)."""
        touch_activity()
        cfg = resolve_backend(backend)
        worker_model = model if model else cfg["default_model"]
        backend_label = (backend or DEFAULT_BACKEND).lower()
        planner_model = plan_model if plan_model else PLAN_MODELS.get(backend_label, cfg["default_model"])
        max_workers = min(max(max_workers, 2), 8)

        goal_with_memory = goal
        if memory:
            mem_ctx = await tcmm_recall(goal)
            if mem_ctx:
                goal_with_memory = f"{mem_ctx}\n\n--- CURRENT GOAL ---\n{goal}"

        output = [f"# Coordinator Mission (ULTRAPLAN)\n**Goal:** {goal}\n"]

        # One scope for the whole coordinate() episode: planner +
        # workers + reconciler + verify all share a single child cid
        # so their archive blocks land in one cohesive sub-namespace
        # with a common lineage parent. (If you want per-worker
        # namespaces instead, wrap run_worker individually like
        # parallel_agents does.)
        async with _spawn_scope("coord"):
            return await _coordinate_inner(
                goal, goal_with_memory, planner_model, worker_model,
                backend_label, max_workers, verify, output,
            )

    async def _coordinate_inner(goal, goal_with_memory, planner_model,
                                 worker_model, backend_label, max_workers,
                                 verify, output):
        # Phase 1: Plan
        plan_response = await call_llm(COORDINATOR_SYSTEM,
            f"Goal: {goal_with_memory}\n\nCreate a work plan with up to {max_workers} specialist tasks.",
            model=planner_model, backend=backend_label, temperature=0.3, role="coordinator")

        try:
            clean = plan_response.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
                if clean.endswith("```"): clean = clean[:-3]
            task_plan = json.loads(clean.strip())
            if not isinstance(task_plan, list): task_plan = [task_plan]
            task_plan = task_plan[:max_workers]
        except (json.JSONDecodeError, ValueError):
            result = await call_llm("You are a senior cybersecurity analyst at Phishield.",
                goal, model=worker_model, backend=backend_label)
            output.append(f"## Fallback Result\n\n{result}")
            return "\n".join(output)

        for i, t in enumerate(task_plan):
            output.append(f"- **Worker {i+1}** [{t.get('role','?')}]: {t.get('task','?')}")

        # Phase 2: Execute
        async def run_worker(idx, spec):
            r = spec.get("role", "researcher")
            sp = ROLES.get(r, {}).get("system", "You are a helpful assistant.")
            return f"### Worker {idx+1}: {ROLES.get(r, {}).get('name', r)}\n\n{await call_llm(sp, spec.get('task',''), model=worker_model, backend=backend_label, role=r)}"

        worker_results = await asyncio.gather(*[run_worker(i, t) for i, t in enumerate(task_plan)], return_exceptions=True)
        worker_texts = [str(r) if not isinstance(r, Exception) else f"### Worker: ERROR\n{r}" for r in worker_results]

        # Phase 3: Reconcile
        all_worker_output = await compact_results(worker_texts, backend=backend_label, model=worker_model)
        reconciled = await call_llm(RECONCILER_SYSTEM,
            f"Original goal: {goal}\n\nWorker results:\n\n{all_worker_output}\n\nSynthesize into a unified report.",
            model=planner_model, backend=backend_label, temperature=0.3, role="coordinator")
        output.append(f"\n## Final Report\n\n{reconciled}")

        # Phase 4: Verify (optional)
        if verify:
            verification = await call_llm(ROLES["verify"]["system"],
                f"Original goal: {goal}\n\nReport to verify:\n{reconciled}",
                model=planner_model, backend=backend_label, temperature=0.2, role="verify")
            output.append(f"\n## Verification\n\n{verification}")

        return maybe_offload("\n".join(output), f"coord-{uuid.uuid4().hex[:6]}")

    @mcp.tool()
    async def pipeline(steps: str, backend: str = "", model: str = "", memory: bool = False) -> str:
        """Run a sequential agent pipeline. Each step's output feeds into the next."""
        touch_activity()
        try:
            step_list = json.loads(steps)
        except json.JSONDecodeError:
            return "Error: 'steps' must be a valid JSON array"

        cfg = resolve_backend(backend)
        use_model = model if model else cfg["default_model"]
        backend_label = (backend or DEFAULT_BACKEND).lower()
        pipe_id = uuid.uuid4().hex[:6]
        output = [f"# Pipeline `{pipe_id}` — {len(step_list)} steps\n"]
        prev_result = ""

        # One scope for the whole pipeline run — each step is a
        # stage in the same episode, so one shared child cid +
        # one lineage edge back to the parent makes sense.
        async with _spawn_scope(f"pipe-{pipe_id}"):
            for idx, step in enumerate(step_list):
                role = step.get("role", "researcher")
                task_text = step.get("task", "")
                sys_prompt = ROLES.get(role, {}).get("system", "You are a helpful assistant.")
                full_task = f"Previous step output:\n{prev_result}\n\n---\n\nCurrent task:\n{task_text}" if prev_result else task_text
                if memory and idx == 0:
                    mem_ctx = await tcmm_recall(task_text)
                    if mem_ctx: full_task = f"{mem_ctx}\n\n--- CURRENT TASK ---\n{full_task}"

                SCRATCHPAD_DIR.mkdir(parents=True, exist_ok=True)
                (SCRATCHPAD_DIR / f"pipeline-{pipe_id}-progress.txt").write_text(
                    f"Pipeline {pipe_id}: Step {idx+1}/{len(step_list)} running...", encoding="utf-8")

                prev_result = await call_llm(sys_prompt, full_task, model=use_model, backend=backend_label, role=role)
                output.append(f"## Step {idx+1}: {ROLES.get(role,{}).get('name',role)}\n\n{prev_result}\n")

        return maybe_offload("\n---\n".join(output), f"pipe-{pipe_id}")

    @mcp.tool()
    async def review_loop(task: str, role: str = "coder", critic_prompt: str = "", max_rounds: int = 3, model: str = "") -> str:
        """Agent with automatic quality review loop: work → critique → revise."""
        touch_activity()
        use_model = model if model else DEFAULT_MODEL
        worker_prompt = ROLES.get(role, {}).get("system", "You are a helpful assistant.")
        critic_sys = critic_prompt if critic_prompt else ROLES["critic"]["system"]
        output_log = [f"# Review Loop ({max_rounds} max rounds)\n"]
        current_task = task

        # Whole review loop = one episode (worker + critic iterating
        # on the same task). Shared child cid keeps them together.
        async with _spawn_scope("reviewloop"):
            for round_num in range(1, max_rounds + 1):
                worker_result = await call_llm(worker_prompt, current_task, model=use_model, role=role)
                output_log.append(f"## Round {round_num}: Worker\n{worker_result}")
                review_input = f"Original task:\n{task}\n\nWorker output:\n{worker_result}\n\nReview this."
                critic_result = await call_llm(critic_sys, review_input, model=use_model, temperature=0.3, role="critic")
                output_log.append(f"## Round {round_num}: Critic\n{critic_result}")
                if critic_result.strip().upper().startswith("PASS"):
                    output_log.append(f"\n**APPROVED in round {round_num}**")
                    break
                if round_num < max_rounds:
                    current_task = f"Original task:\n{task}\n\nYour previous attempt:\n{worker_result}\n\nFeedback:\n{critic_result}\n\nPlease revise."

        return "\n\n".join(output_log)

    @mcp.tool()
    async def spawn_agentic(task: str, role: str = "researcher", max_turns: int = 10, backend: str = "",
                            model: str = "", memory: bool = False, effort: str = "medium") -> str:
        """Spawn an agentic agent that uses tools iteratively: think → tool → result → think → ... → answer.

        Available tools: read_file, write_file, edit_file, search_files, grep, run_command,
        web_search, web_fetch, write_scratchpad, read_scratchpad, tcmm_recall, todo_write.

        Effort: low (3 turns), medium (7), high (15), max (20).
        """
        touch_activity()
        sys_prompt = ROLES.get(role, {}).get("system", "You are a helpful assistant.")
        sys_prompt += "\n\nYou have access to tools. Use them to gather information before answering. Don't guess — look things up."

        eff = EFFORT_LEVELS.get(effort, EFFORT_LEVELS["medium"])
        effective_turns = eff["max_turns"] if effort != "medium" else min(max_turns, 20)

        task_with_memory = task
        if memory:
            mem_ctx = await tcmm_recall(task)
            if mem_ctx: task_with_memory = f"{mem_ctx}\n\n--- CURRENT TASK ---\n{task}"

        cfg = resolve_backend(backend)
        if eff["model_hint"] == "pro" and not model:
            use_model = PLAN_MODELS.get((backend or DEFAULT_BACKEND).lower(), cfg["default_model"])
        else:
            use_model = model if model else cfg["default_model"]
        backend_label = (backend or DEFAULT_BACKEND).lower()

        agent_tools = AGENT_TOOLS
        if role == "explore":
            agent_tools = [t for t in AGENT_TOOLS if t["name"] not in ("write_file", "edit_file", "run_command")]

        # The whole agentic loop (every turn's LLM call + tool result)
        # runs under one child cid so all blocks from this sub-agent
        # cluster in one sub-namespace and share a lineage edge back
        # to the parent's tool_use aid.
        async with _spawn_scope("agentic"):
            result = await agentic_loop(sys_prompt, task_with_memory, max_turns=effective_turns,
                                         backend=backend_label, model=use_model, tools=agent_tools)
        agent_id = f"agentic-{uuid.uuid4().hex[:6]}"
        log_transcript(agent_id, role, result)
        save_session()
        return maybe_offload(f"# Agentic Agent: {ROLES.get(role,{}).get('name',role)} ({use_model}, effort={effort})\n\n{result}", agent_id)
