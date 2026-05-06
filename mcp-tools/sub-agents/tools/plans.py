"""Plan-mode tools — propose, review, and track multi-step plans.

The "plan mode" pattern: the agent drafts a structured multi-step plan,
presents it to the user for approval, then marks each step as it executes.
This mirrors Claude Code's EnterPlanMode/ExitPlanMode but adapts to
LibreChat's request/response chat — there's no special "plan" UI gate,
so approval is just the user's next chat turn ("approve" / "modify" /
"cancel").

WHEN TO USE plan_create:
- Task has 3+ distinct steps with order or dependencies
- The user might want to redirect or adjust the approach BEFORE work
  starts (e.g. "actually skip step 2, just do 1 and 3")
- You want a checkpoint the model can refer back to mid-execution to
  stay on track

WHEN NOT TO USE:
- Trivial single-step tasks
- Tasks where the user has already given explicit instructions
- Pure information requests ("what does this code do?")

Plans persist to disk so they survive across MCP requests and the model
can resume execution after an interruption.
"""

import json
import re
import time
from pathlib import Path

from config import PROJECT_ROOT

PLANS_DIR = Path(PROJECT_ROOT) / "tcmm-data" / "plans"

_VALID_STATUSES = ("pending", "in_progress", "done", "skipped")


def _safe_id(s: str) -> str:
    return re.sub(r"[^\w\-]", "_", s.lower())[:40] or "plan"


def _load(plan_id: str):
    path = PLANS_DIR / f"{plan_id}.json"
    if not path.exists():
        return None, None
    try:
        return path, json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return path, None


def _save(path: Path, plan: dict):
    path.write_text(json.dumps(plan, indent=2), encoding="utf-8")


def register(mcp):
    @mcp.tool()
    async def plan_create(title: str, steps: str) -> str:
        """Create a multi-step plan and present it to the user for approval.

        Returns a formatted plan plus a prompt asking the user to approve.
        The user replies in chat with `approve`, `modify <changes>`, or
        `cancel`. After approval, call plan_update_step as you execute
        each step so the user can see live progress.

        Args:
            title: Short title (e.g. "Refactor auth module").
            steps: Newline-separated list of steps. Each step on its own
                   line. Example:
                       "Read existing auth.py
                        Identify dead code paths
                        Remove unused imports and update tests"
        """
        title = (title or "").strip()
        if not title:
            return "Error: title is required"

        step_list = [s.strip() for s in (steps or "").split("\n") if s.strip()]
        if not step_list:
            return "Error: steps must contain at least one non-empty line"
        if len(step_list) > 50:
            return f"Error: too many steps ({len(step_list)}); cap is 50"

        PLANS_DIR.mkdir(parents=True, exist_ok=True)
        plan_id = f"{int(time.time())}_{_safe_id(title)}"
        plan = {
            "id": plan_id,
            "title": title,
            "created_at": int(time.time()),
            "status": "awaiting_approval",
            "steps": [
                {"index": i, "text": s, "status": "pending", "notes": ""}
                for i, s in enumerate(step_list)
            ],
        }
        _save(PLANS_DIR / f"{plan_id}.json", plan)

        out = [f"# Plan: {title}", f"_id: `{plan_id}`_", ""]
        for i, s in enumerate(step_list, 1):
            out.append(f"  {i}. {s}")
        out.append("")
        out.append("**Please review and reply with one of:**")
        out.append("  - `approve` — proceed as drafted")
        out.append("  - `modify: <changes>` — request adjustments before I start")
        out.append("  - `cancel` — abandon")
        return "\n".join(out)

    @mcp.tool()
    async def plan_get(plan_id: str) -> str:
        """Retrieve a plan with current step statuses.

        Use this mid-execution to refresh your memory of the plan or
        confirm where you are in the sequence.
        """
        path, plan = _load(plan_id)
        if path is None:
            return f"Error: plan '{plan_id}' not found"
        if plan is None:
            return f"Error: plan '{plan_id}' file is corrupt"

        markers = {"pending": "[ ]", "in_progress": "[~]",
                   "done": "[x]", "skipped": "[-]"}
        out = [f"# Plan: {plan['title']}",
               f"_id: `{plan['id']}` | status: **{plan['status']}**_",
               ""]
        for s in plan["steps"]:
            marker = markers.get(s["status"], "[?]")
            out.append(f"  {s['index'] + 1}. {marker} {s['text']}")
            if s.get("notes"):
                out.append(f"     _{s['notes']}_")
        return "\n".join(out)

    @mcp.tool()
    async def plan_update_step(plan_id: str, step_index: int, status: str,
                                notes: str = "") -> str:
        """Update a step's status as you execute the plan.

        Call this immediately after starting (status='in_progress') and
        finishing (status='done') each step so the user can follow
        progress. Use 'skipped' for steps you decided not to run, with a
        note explaining why.

        Args:
            plan_id: Plan to update.
            step_index: 0-indexed step position (matches the index in plan_get).
            status: 'pending', 'in_progress', 'done', or 'skipped'.
            notes: Optional one-line note (what was done, blocker, etc.).
        """
        path, plan = _load(plan_id)
        if path is None:
            return f"Error: plan '{plan_id}' not found"
        if plan is None:
            return f"Error: plan '{plan_id}' file is corrupt"

        if step_index < 0 or step_index >= len(plan["steps"]):
            return (f"Error: step_index {step_index} out of range "
                    f"(plan has {len(plan['steps'])} steps)")
        if status not in _VALID_STATUSES:
            return (f"Error: status must be one of "
                    f"{'/'.join(_VALID_STATUSES)}; got '{status}'")

        plan["steps"][step_index]["status"] = status
        if notes:
            plan["steps"][step_index]["notes"] = notes[:280]

        # Auto-bump plan status. The model never sets plan-level status
        # directly — we infer it from step states so progress is always
        # consistent with what plan_get shows.
        if all(s["status"] in ("done", "skipped") for s in plan["steps"]):
            plan["status"] = "complete"
        elif plan["status"] == "awaiting_approval" and any(
            s["status"] != "pending" for s in plan["steps"]
        ):
            plan["status"] = "in_progress"

        _save(path, plan)
        return f"Plan `{plan_id}` step {step_index + 1} → {status}"

    @mcp.tool()
    async def plan_list() -> str:
        """List recent plans. Auto-prunes plans older than 7 days."""
        if not PLANS_DIR.exists():
            return "No plans yet."

        now = time.time()
        rows = []
        # Sort by mtime desc so the freshest plan shows first.
        for path in sorted(PLANS_DIR.glob("*.json"),
                           key=lambda p: p.stat().st_mtime, reverse=True):
            age_days = (now - path.stat().st_mtime) / 86400
            if age_days > 7:
                try:
                    path.unlink()
                except OSError:
                    pass
                continue
            try:
                p = json.loads(path.read_text(encoding="utf-8"))
                done = sum(1 for s in p["steps"] if s["status"] == "done")
                total = len(p["steps"])
                rows.append(
                    f"- `{p['id']}` — {p['title']} "
                    f"[**{p['status']}**] {done}/{total} done"
                )
            except (json.JSONDecodeError, KeyError):
                continue

        return "# Plans\n" + "\n".join(rows) if rows else "No plans yet."
