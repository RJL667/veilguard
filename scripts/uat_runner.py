"""End-to-end UAT runner for the Veilguard multi-agent platform.

Drives the LIVE agent-runtime (+ TCMM + ledger) the same way LibreChat
does, under the user's REAL tenant/user_id, and reports pass/fail with
latency for each case.  Run from the host:

    python scripts/uat_runner.py

Requires: agent-runtime on :5000, TCMM on :8811, all under the real
user namespace.
"""
from __future__ import annotations
import json, time, urllib.request, urllib.error, sys, subprocess

RT = "http://localhost:5000"
TCMM = "http://localhost:8811"
USER = "69c4468a1fde1abc19c7835c"      # the real logged-in user
TENANT = USER

def _post(url, body, timeout=120):
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode()
    return time.time() - t0, raw

def director(cid, msg, timeout=200):
    el, raw = _post(f"{RT}/agent/query", {
        "conversation_id": cid, "user_id": USER, "tenant_id": TENANT,
        "agent_id": "director", "messages": [{"role": "user", "content": msg}],
        "stream": False,
    }, timeout=timeout)
    try:
        d = json.loads(raw)
        evs = d.get("events", [])
        text = ""
        usage = {}
        for e in evs:
            if e.get("type") == "final_result":
                text = e.get("result", "") or text
            if e.get("type") == "assistant_text" and not text:
                text = e.get("text", "")
            if e.get("type") == "usage":
                usage = e
        return el, text, usage, evs
    except Exception as ex:
        return el, f"[parse error: {ex}] {raw[:200]}", {}, []

def db_task(tid):
    """Read a task row via docker exec."""
    code = (
        "import lancedb,json;"
        "db=lancedb.connect('/tcmm-data/veilguard/tcmm.db');"
        "t=db.open_table('agent_tasks');"
        f"a=t.search().where(\"id='{tid}'\").limit(1).to_arrow();"
        "r={c:a.column(c)[0].as_py() for c in a.column_names} if a.num_rows else {};"
        "ex=json.loads(r.get('extras_json') or '{}');"
        "print(json.dumps({'status':r.get('status'),'owner':r.get('owner_id'),"
        "'outputs':r.get('outputs'),'parent':r.get('parent_id'),"
        "'ac':ex.get('ac_results'),'kind':r.get('kind')}))"
    )
    out = subprocess.run(
        ["docker", "exec", "veilguard-agent-runtime-1", "python", "-c", code],
        capture_output=True, text=True, timeout=30,
    ).stdout.strip().splitlines()
    for line in reversed(out):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    return {}

def children(parent_tid):
    code = (
        "import lancedb,json;"
        "db=lancedb.connect('/tcmm-data/veilguard/tcmm.db');"
        "t=db.open_table('agent_tasks');"
        f"a=t.search().where(\"parent_id='{parent_tid}'\").to_arrow();"
        "rows=[{c:a.column(c)[i].as_py() for c in ['id','status','owner_id']} for i in range(a.num_rows)];"
        "print(json.dumps(rows))"
    )
    out = subprocess.run(
        ["docker", "exec", "veilguard-agent-runtime-1", "python", "-c", code],
        capture_output=True, text=True, timeout=30,
    ).stdout.strip().splitlines()
    for line in reversed(out):
        if line.strip().startswith("["):
            return json.loads(line.strip())
    return []

def find_tasks_in(text):
    import re
    return re.findall(r"task-[a-f0-9]{12}", text)

def watch(tid, timeout=240, want=("done", "cancelled")):
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        r = db_task(tid)
        st = r.get("status")
        if st != last:
            print(f"      [{time.time()-t0:5.0f}s] {tid} → {st} (owner={r.get('owner')}, ac={r.get('ac')})")
            last = st
        if st in want:
            return st, r, time.time() - t0
        time.sleep(6)
    return last, db_task(tid), time.time() - t0

RESULTS = []
def record(case, passed, detail):
    RESULTS.append((case, passed, detail))
    mark = "PASS" if passed else "FAIL"
    print(f"  >>> {case}: {mark} — {detail}")


def ts():
    return str(int(time.time()))


def main():
    print("=" * 70)
    print("VEILGUARD MULTI-AGENT UAT  (user=%s)" % USER)
    print("=" * 70)

    # ---- P1: Director solo warm latency ----
    print("\n[P1] Director solo 'hello' (warm latency)")
    cid = f"uat-p1-{ts()}"
    director(cid, "warm up")              # prime cache
    el, text, usage, _ = director(cid, "Reply with just the word: ready")
    cr = usage.get("cache_read", 0)
    record("P1 solo-latency", el < 5.0,
           f"{el:.2f}s (target <5s), cache_read={cr}, reply={text[:40]!r}")

    # ---- P2: cache stability over 3 turns ----
    print("\n[P2] Director 3 consecutive turns (cache stability)")
    cid = f"uat-p2-{ts()}"
    director(cid, "hi")
    times = []
    crs = []
    for m in ["one", "two", "three"]:
        el, _, usage, _ = director(cid, m)
        times.append(el); crs.append(usage.get("cache_read", 0))
    record("P2 cache-stability", all(t < 5 for t in times) and any(c > 0 for c in crs),
           f"times={[round(t,1) for t in times]}s cache_reads={crs}")

    # ---- M1: memory recall within a conversation ----
    print("\n[M1] TCMM recall WITHIN a conversation")
    cid = f"uat-m1-{ts()}"
    director(cid, "Remember this for later: the project codeword is BLUEHERON.")
    time.sleep(3)
    el, text, _, _ = director(cid, "What is the project codeword I just gave you? Reply with just the word.")
    record("M1 mem-same-conv", "blueheron" in text.lower(),
           f"reply={text[:60]!r}")

    # ---- M2: cross-conversation recall (TCMM persistence) ----
    print("\n[M2] TCMM recall ACROSS conversations (persistence)")
    cidA = f"uat-m2a-{ts()}"
    director(cidA, "Important fact to store in memory: my deployment region is af-south-1 and my cluster is named IRONWOOD.")
    print("      (waiting 12s for TCMM ingest+index)")
    time.sleep(12)
    cidB = f"uat-m2b-{ts()}"
    el, text, _, _ = director(cidB, "From my stored memory: what is my cluster named? Reply with just the name.")
    record("M2 mem-cross-conv", "ironwood" in text.lower(),
           f"reply={text[:80]!r}  (cross-conv recall)")

    # ---- B1: single delegation full chain ----
    print("\n[B1] Pattern B — single delegation → done")
    cid = f"uat-b1-{ts()}"
    path = f"team/drafts/uat-b1-{ts()}.md"
    el, text, _, evs = director(cid,
        f"Write a 100-word internal memo at {path} on the pros/cons of "
        f"server-side prompt caching. Sections: TLDR, Recommendation. "
        f"Reason from your own knowledge. Delegate to a researcher.")
    tids = []
    for ev in evs:
        if ev.get("type") == "tool_result" and ev.get("name") == "create_task":
            tids += find_tasks_in(ev.get("result_preview", ""))
    tids = list(dict.fromkeys(tids))
    print(f"      Director created: {tids}")
    if tids:
        st, row, took = watch(tids[0], timeout=240)
        record("B1 single-deleg", st == "done" and row.get("outputs"),
               f"{tids[0]} → {st} in {took:.0f}s, outputs={row.get('outputs')}")
    else:
        record("B1 single-deleg", False, "no task id captured from Director")

    # ---- C1: DECOMPOSITION (parallel fanout) ----
    print("\n[C1] Pattern C — DECOMPOSITION (parent + 2 subtasks)")
    cid = f"uat-c1-{ts()}"
    p1 = f"team/drafts/uat-c1-cost-{ts()}.md"
    p2 = f"team/drafts/uat-c1-latency-{ts()}.md"
    el, text, _, evs = director(cid,
        f"I need two SEPARATE independent analyses, each its own subtask, "
        f"delegated to researchers: (1) a 80-word cost analysis at {p1}, "
        f"and (2) a 80-word latency analysis at {p2}, both on whether to "
        f"run the Director on Haiku vs Sonnet. Create a parent task with "
        f"two subtasks. Reason from your own knowledge.")
    all_tids = []
    for ev in evs:
        if ev.get("type") in ("tool_result",) and ev.get("name") == "create_task":
            all_tids += find_tasks_in(ev.get("result_preview", ""))
    all_tids = list(dict.fromkeys(all_tids))
    print(f"      Director created tasks: {all_tids}")
    if all_tids:
        # Watch all created tasks to terminal
        deadline = time.time() + 300
        final = {}
        while time.time() < deadline:
            states = {t: db_task(t).get("status") for t in all_tids}
            # also discover children of any parent
            for t in list(all_tids):
                for ch in children(t):
                    if ch["id"] not in all_tids:
                        all_tids.append(ch["id"])
                        print(f"      discovered subtask {ch['id']}")
            if all(s in ("done", "cancelled") for s in states.values()):
                final = states; break
            time.sleep(8)
        final = {t: db_task(t).get("status") for t in all_tids}
        n_done = sum(1 for s in final.values() if s == "done")
        print(f"      final states: {final}")
        record("C1 decomposition", n_done >= 2 and len(all_tids) >= 2,
               f"{n_done}/{len(all_tids)} tasks done: {final}")
    else:
        record("C1 decomposition", False, "Director created no tasks")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("UAT SUMMARY")
    print("=" * 70)
    npass = sum(1 for _, p, _ in RESULTS if p)
    for case, passed, detail in RESULTS:
        print(f"  {'PASS' if passed else 'FAIL'}  {case}: {detail}")
    print(f"\n  {npass}/{len(RESULTS)} passed")
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
