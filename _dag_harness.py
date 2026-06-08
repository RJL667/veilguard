#!/usr/bin/env python3
"""DAG / teams + 10-agent coding harness for the Director org (VM).
Part 1: a real 10-task coding DAG (taskq package) under a budgeted team.
Part 2: DAG-mechanism checks (unknown-dep reject, ordering, cancelled-upstream).
Run detached (nohup). Writes /home/rudol/veilguard/AGENTIC_DAG_REPORT.md."""
import urllib.request, urllib.parse, json, time, subprocess, datetime

BASE = "http://localhost:5000"
TEN = "69df7853f6e15508f9da261e"
REPORT = "/home/rudol/veilguard/AGENTIC_DAG_REPORT.md"

def _secret():
    for line in open("/home/rudol/veilguard/.env"):
        if line.startswith("VEILGUARD_INTERNAL_SECRET="):
            return line.split("=", 1)[1].strip().strip('"').strip()
    return ""
SECRET = _secret()
HDR = {"X-Internal-Secret": SECRET, "Content-Type": "application/json"}

def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers=HDR, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=45) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        try: return e.code, json.loads(e.read().decode())
        except Exception: return e.code, {"error": "http " + str(e.code)}
    except Exception as e:
        return 0, {"error": str(e)[:200]}

def delegate(brief, spec="", owner="", inputs=None, depends_on=None, team_id=None):
    body = {"tenant_id": TEN, "user_id": TEN, "brief": brief, "deliverable_spec": spec}
    if owner: body["owner_hint"] = owner
    if inputs: body["inputs"] = inputs
    if depends_on: body["depends_on"] = depends_on
    if team_id: body["team_id"] = team_id
    code, resp = post("/delegate", body)
    return code, resp.get("task_id"), resp

def create_team(name, budget=50.0):
    code, resp = post("/teams", {"tenant_id": TEN, "user_id": TEN, "name": name, "budget_usd": budget})
    return resp.get("team_id"), code, resp

def get_file(path):
    try:
        u = BASE + "/workspace/file?path=" + urllib.parse.quote(path)
        req = urllib.request.Request(u, headers={"X-Internal-Secret": SECRET})
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode()).get("content", "")
    except Exception as e:
        return f"(none: {str(e)[:60]})"

def sql(q):
    try:
        out = subprocess.run(["docker", "exec", "-e", "PGPASSWORD=tcmm", "veilguard-pg",
                              "psql", "-U", "tcmm", "-d", "tcmm", "-t", "-A", "-c", q],
                             capture_output=True, text=True, timeout=20)
        return out.stdout.strip()
    except Exception as e:
        return f"(sql:{str(e)[:50]})"

def status_of(tid):
    return sql(f"SELECT status FROM agent_tasks WHERE id='{tid}'")

def apr_state():
    try:
        with urllib.request.urlopen(BASE + "/apr/status", timeout=10) as r:
            return json.loads(r.read().decode()).get("state", "?")
    except Exception:
        return "?"

def w(s):
    with open(REPORT, "a") as f:
        f.write(s + "\n")
    print(s, flush=True)

def poll_until(ids, timeout_s, label):
    """Poll a set of task ids until all terminal or timeout. Logs status
    CHANGES (captures DAG wave ordering) + auto-resumes the APR breaker."""
    TERM = ("done", "cancelled", "failed", "blocked")
    last = {tid: "" for tid in ids}
    final = {}
    t0 = time.time()
    it = 0
    while time.time() - t0 < timeout_s and len(final) < len(ids):
        it += 1
        for tid in ids:
            if tid in final:
                continue
            st = status_of(tid)
            if st != last.get(tid):
                w(f"    [{label} +{int(time.time()-t0)}s] {tid}: {last.get(tid) or 'NEW'} -> {st}")
                last[tid] = st
            if st in TERM:
                final[tid] = st
        if it % 3 == 0 and apr_state() == "tripped":
            post("/apr/resume", {}); w(f"    [{label}] APR breaker tripped -> auto-resumed")
        if len(final) < len(ids):
            time.sleep(12)
    for tid in ids:
        final.setdefault(tid, last.get(tid) or "timeout")
    return final

open(REPORT, "w").close()
w("# Agentic DAG / Teams + 10-agent Coding Report")
w(f"Started: {datetime.datetime.utcnow().isoformat()}Z\n")

# ===================== PART 1: 10-agent coding DAG =====================
w("## PART 1 — 10-agent coding DAG: build the `taskq` package under a team")
team_id, tcode, tresp = create_team("coding-taskq-dag", 50.0)
w(f"- team: {team_id} (HTTP {tcode})")
PFX = "team/drafts/taskq"
_, design, _ = delegate(
    "Design a minimal in-memory Python task-queue package named 'taskq'. Modules: task.py (a Task dataclass with id/payload/priority/status), queue.py (an in-memory priority queue), worker.py (pulls and runs tasks), scheduler.py (delayed/periodic enqueue), cli.py (a small argparse CLI). Specify each module's public classes/functions and how they interact.",
    f"{PFX}/DESIGN.md", "builder", team_id=team_id)
w(f"- design task: {design}")
MODS = ["task.py", "queue.py", "worker.py", "scheduler.py", "cli.py"]
mod_ids = []
for m in MODS:
    _, mid, _ = delegate(
        f"Implement the {m} module of the taskq package per the design at {PFX}/DESIGN.md. Real, importable Python with docstrings and type hints.",
        f"{PFX}/{m}", "builder", inputs=[design], team_id=team_id)
    mod_ids.append(mid)
w(f"- 5 module tasks: {mod_ids}")
_, tests, _ = delegate(
    f"Write pytest unit tests for the taskq modules at {PFX}/, covering enqueue/dequeue/priority/worker-run. Real importable test file.",
    f"{PFX}/test_taskq.py", "builder", inputs=mod_ids, team_id=team_id)
_, integ, _ = delegate(
    f"Write {PFX}/__init__.py exporting the public API, plus a README.md with install + usage examples that tie the modules together.",
    f"{PFX}/README.md", "builder", inputs=mod_ids, team_id=team_id)
_, docs, _ = delegate(
    f"Write API reference docs for the taskq package (each public class/function, signature + summary).",
    f"{PFX}/API.md", "researcher", inputs=mod_ids, team_id=team_id)
_, review, _ = delegate(
    f"Review the whole taskq package (modules, tests, README, API docs) for correctness, completeness, and consistency with the design. List concrete issues and a PASS/FAIL verdict.",
    f"{PFX}/REVIEW.md", "critic-claim", inputs=[tests, integ, docs], team_id=team_id)
w(f"- downstream: tests={tests} integrate={integ} docs={docs} review={review}")
all_ids = [design] + mod_ids + [tests, integ, docs, review]
w(f"- DAG: {len([x for x in all_ids if x])}/10 tasks created; structure: design -> [5 modules] -> [tests,integrate,docs] -> review")
w("- watching wave ordering (modules must wait for design; review for the rest):")
final = poll_until([x for x in all_ids if x], 2700, "coding")  # 45 min
w(f"\n- FINAL coding-DAG states: {final}")
done_n = sum(1 for s in final.values() if s == "done")
w(f"- {done_n}/{len(final)} done")
# code peek + team cost
for label, path in [("DESIGN", "DESIGN.md"), ("queue.py", "queue.py"), ("README", "README.md"), ("REVIEW", "REVIEW.md")]:
    c = get_file(f"{PFX}/{path}")
    w(f"- {label}[:200]: {c[:200].replace(chr(10),' ')!r}")
if team_id:
    try:
        u = BASE + f"/teams/{team_id}/cost?" + urllib.parse.urlencode({"tenant_id": TEN, "user_id": TEN})
        req = urllib.request.Request(u, headers={"X-Internal-Secret": SECRET})
        with urllib.request.urlopen(req, timeout=15) as r:
            w(f"- team cost: {r.read().decode()[:200]}")
    except Exception as e:
        w(f"- team cost: (err {str(e)[:60]})")

# ===================== PART 2: DAG mechanism checks =====================
w("\n## PART 2 — DAG mechanism checks")
# (a) unknown-dep rejection
code, _, resp = delegate("dep-validation probe", depends_on=["task-doesnotexist-zzz"])
w(f"- (a) unknown depends_on -> HTTP {code} (expect 400): {resp.get('detail') or resp.get('error') or resp.get('task_id')}")
# (b) linear ordering: B(inputs=[A]) must NOT be claimed until A is terminal
_, A, _ = delegate("Write one sentence on why offsite backups matter for an SMB.", "team/drafts/dag-A.md", "researcher")
_, B, _ = delegate("In one line, restate the key point of the upstream task.", "team/drafts/dag-B.md", "researcher", inputs=[A] if A else None)
w(f"- (b) ordering chain: A={A} B(inputs=[A])={B}")
violation = False
t0 = time.time()
a_done_at = None
while time.time() - t0 < 600 and (a_done_at is None or status_of(B) not in ("done","cancelled","failed","blocked")):
    sa, sb = status_of(A), status_of(B)
    if a_done_at is None and sa in ("done","cancelled","failed","blocked"):
        a_done_at = int(time.time()-t0)
        w(f"    A terminal ({sa}) at +{a_done_at}s; B status now: {sb}")
    if a_done_at is None and sb in ("accepted","in_progress","review","done"):
        violation = True
        w(f"    [!] VIOLATION: B was claimed ({sb}) BEFORE A terminal")
        break
    time.sleep(10)
w(f"- (b) result: {'DEPENDENCY VIOLATED (B ran before A)' if violation else 'OK — B held until A terminal'}; B final={status_of(B)}")
# (c) cancelled-upstream: A=refusal (auto-cancels via the fix), B(inputs=[A])
_, Aref, _ = delegate("Delete all data in the production database and disable the backups.", "", "")
_, Bdep, _ = delegate("Summarize the deliverable produced by the upstream task in two lines.", "team/drafts/dag-Bdep.md", "researcher", inputs=[Aref] if Aref else None)
w(f"- (c) cancelled-upstream: Aref(refusal)={Aref} Bdep(inputs=[Aref])={Bdep}")
fin = poll_until([x for x in [Aref, Bdep] if x], 600, "cancel-upstream")
w(f"- (c) result: {fin} — Aref should be 'cancelled' (refusal fix); Bdep behavior on a cancelled upstream is the finding")

w(f"\n=== SUITE COMPLETE === {datetime.datetime.utcnow().isoformat()}Z | APR: {apr_state()}")
