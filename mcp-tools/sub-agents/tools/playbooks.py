"""Playbook tools — pre-built coordinator plans for common scenarios."""

import json
from config import PLAYBOOKS_DIR
from utils.helpers import load_playbook
from utils.afk import touch_activity


def register(mcp):
    @mcp.tool()
    async def list_playbooks() -> str:
        """List available incident playbooks."""
        if not PLAYBOOKS_DIR.exists(): return "No playbooks directory found."
        files = sorted(PLAYBOOKS_DIR.glob("*.json"))
        if not files: return "No playbooks found."
        lines = ["# Available Incident Playbooks\n"]
        for f in files:
            try:
                pb = json.loads(f.read_text(encoding="utf-8"))
                roles = [t.get("role", "?") for t in pb.get("tasks", [])]
                lines.append(f"- **{f.stem}**: {pb.get('name', f.stem)} — {pb.get('description', '')}")
                lines.append(f"  Workers: {', '.join(roles)}")
            except Exception:
                lines.append(f"- **{f.stem}**: (invalid JSON)")
        return "\n".join(lines)

    @mcp.tool()
    async def run_playbook(name: str, context: str = "", backend: str = "", model: str = "", plan_model: str = "") -> str:
        """Run a pre-built incident playbook. Loads a plan, injects context, executes the coordinator."""
        touch_activity()
        pb = load_playbook(name)
        if not pb:
            available = [f.stem for f in PLAYBOOKS_DIR.glob("*.json")] if PLAYBOOKS_DIR.exists() else []
            return f"Error: Playbook '{name}' not found. Available: {', '.join(available) or '(none)'}"
        tasks = pb.get("tasks", [])
        if not tasks: return f"Error: Playbook '{name}' has no tasks."
        if context:
            for t in tasks: t["task"] = t["task"].replace("{context}", context)

        # Import and call coordinate
        from tools.agents import register as _  # Ensure agents module is loaded
        # We need to call coordinate directly — import the function from the mcp tools
        # Since tools are registered on the mcp instance, we call it through the module
        from core.llm import call_llm, resolve_backend, compact_results
        from config import ROLES, DEFAULT_BACKEND, PLAN_MODELS
        import asyncio

        goal = f"[Playbook: {pb.get('name', name)}] {context}" if context else f"[Playbook: {pb.get('name', name)}]"
        cfg = resolve_backend(backend)
        worker_model = model if model else cfg["default_model"]
        backend_label = (backend or DEFAULT_BACKEND).lower()
        planner_model = plan_model if plan_model else PLAN_MODELS.get(backend_label, cfg["default_model"])

        output = [f"# Playbook: {pb.get('name', name)}\n"]
        if context: output.append(f"**Context:** {context}\n")

        # Execute workers in parallel
        async def run_worker(idx, spec):
            r = spec.get("role", "researcher")
            sp = ROLES.get(r, {}).get("system", "You are a helpful assistant.")
            return f"### Worker {idx+1}: {ROLES.get(r,{}).get('name',r)}\n\n{await call_llm(sp, spec.get('task',''), model=worker_model, backend=backend_label, role=r)}"

        results = await asyncio.gather(*[run_worker(i, t) for i, t in enumerate(tasks)], return_exceptions=True)
        worker_texts = [str(r) if not isinstance(r, Exception) else f"### ERROR\n{r}" for r in results]

        # Reconcile
        all_output = await compact_results(worker_texts, backend=backend_label, model=worker_model)
        reconciled = await call_llm(
            "You are a senior analyst. Reconcile all findings into a unified report.",
            f"Goal: {goal}\n\nWorker results:\n\n{all_output}", model=planner_model, backend=backend_label, role="coordinator"
        )
        output.append(f"\n## Report\n\n{reconciled}")
        return "\n".join(output)
