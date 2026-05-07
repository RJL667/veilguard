"""Verify A (chain), B (tool lineage), D (classifier no-drop) end-to-end."""
import httpx, time, json

CONV = "bb400c87-0759-4448-ad33-c615ab35cedf"
UID = "69e719f60ace24208868fbbe"


def pre(msg):
    return httpx.post(
        "http://localhost:8811/pre_request",
        json={"user_message": msg, "conversation_id": CONV, "user_id": UID, "origin": "user"},
        timeout=60,
    )


def ingest_tool(items):
    return httpx.post(
        "http://localhost:8811/ingest_turn",
        json={"items": items, "conversation_id": CONV, "user_id": UID},
        timeout=60,
    )


def post_resp(answer):
    return httpx.post(
        "http://localhost:8811/post_response",
        json={"raw_output": answer, "conversation_id": CONV, "user_id": UID, "origin": "assistant_text"},
        timeout=60,
    )


# Trigger one full turn
print("=== firing one real turn with a tool round-trip ===")
t0 = time.time()
r = pre("what did we fix today in TCMM?")
print(f"pre_request: wall={(time.time()-t0)*1000:.0f}ms status={r.status_code}")
time.sleep(3)

# Simulate a tool round-trip
r = ingest_tool([
    {
        "text": "pwd && ls -la /home/rudol",
        "origin": "tool_use",
        "tool_name": "run_command",
        "tool_use_id": "toolu_VERIFY_01",
        "param_hash": "abc123",
        "params": {"command": "pwd && ls -la /home/rudol"},
    },
    {
        "text": "/home/rudol\\ntotal 124\\ndrwxr-xr-x 15 rudol rudol 4096 Apr 23 10:49 veilguard",
        "origin": "tool_result",
        "tool_name": "run_command",
        "tool_use_id": "toolu_VERIFY_01",
        "param_hash": "abc123",
        "result": "/home/rudol\\n...",
    },
])
print(f"ingest_turn (tool_use+result): status={r.status_code} added={r.json().get('added')}")
time.sleep(2)

r = post_resp(
    "Today we shipped four fixes: deferred ingest, per-session locks, Lance auto-compact, and tool-lineage stamping. "
    '{"knowledge_class": "novel", "used": {}}'
)
print(f"post_response: status={r.status_code}")
time.sleep(6)  # let bg ingests finish

# Now check the archive for the new rows
print()
print("=== inspecting new rows ===")
import lancedb
df = lancedb.connect("/home/rudol/veilguard/tcmm-data/veilguard/tcmm.db").open_table("archive").to_pandas()
recent = df[df["created_ts"] >= t0].sort_values("aid")
print(f"new rows: {len(recent)}")

for _, r in recent.iterrows():
    src = str(r["source"])[:40]
    lin = r["lineage"] if r["lineage"] is not None else {}
    try:
        lin_dict = dict(lin) if hasattr(lin, "keys") else {}
    except Exception:
        lin_dict = {}
    root = lin_dict.get("root", "-")
    _p = lin_dict.get("parents")
    parents = list(_p) if _p is not None else []
    temporal = r["temporal"] if r["temporal"] is not None else {}
    try:
        t_dict = dict(temporal) if hasattr(temporal, "keys") else {}
    except Exception:
        t_dict = {}
    prev_aid = t_dict.get("prev_aid", "-")
    next_aid = t_dict.get("next_aid", "-")
    print(
        f"aid={int(r['aid']):4d}  src={src:42s}  "
        f"prio={str(r['priority_class']):8s}  "
        f"prev={str(prev_aid):>5}  next={str(next_aid):>5}  "
        f"root={str(root):>5}  parents={parents}"
    )

print()
print("=== CHAIN INTEGRITY CHECK ===")
breaks = 0
ss = recent.reset_index(drop=True)
for i in range(len(ss) - 1):
    cur = ss.iloc[i]
    nxt = ss.iloc[i + 1]
    try:
        ct = dict(cur["temporal"]) if cur["temporal"] is not None else {}
        nt = dict(nxt["temporal"]) if nxt["temporal"] is not None else {}
    except Exception:
        continue
    if ct.get("next_aid") != nxt["aid"]:
        breaks += 1
        print(f"  BROKEN: aid={int(cur['aid'])} next_aid={ct.get('next_aid')} expected {int(nxt['aid'])}")
print(f"chain breaks in new rows: {breaks}/{max(len(ss)-1, 0)}")

print()
print("=== TOOL LINEAGE CHECK ===")
tool_rows = recent[recent["source"].astype(str).str.contains("tool_", na=False)]
print(f"tool rows: {len(tool_rows)}")
for _, r in tool_rows.iterrows():
    try:
        lin = dict(r["lineage"]) if r["lineage"] is not None else {}
    except Exception:
        lin = {}
    _p = lin.get("parents")
    parents = list(_p) if _p is not None else []
    print(f"  aid={int(r['aid'])}: lineage.parents={parents}")
