#!/usr/bin/env bash
# Run a controlled test that proves TCMM works end-to-end against
# Google AI Studio (not Vertex AI). Steps:
#
#   1. Stop the live TCMM systemd service (Vertex-mode)
#   2. Start TCMM manually with GOOGLE_CLOUD_PROJECT unset and
#      GEMINI_API_KEY=$GOOGLE_API_KEY so the dual-mode adapter at
#      vertex_nlp_adapter.py:168-188 takes the AI Studio branch
#   3. Wait for /health
#   4. POST /ingest_turn with two test blocks rich with named entities
#   5. Wait ~30s for async NLP workers to extract typed dicts
#   6. Query archive.lance and dump the new rows' topic_dicts /
#      entity_dicts so we can eyeball whether real PERSON/ORG/LOC
#      types came through
#   7. Kill the manual TCMM, restart systemd (back to Vertex)
#   8. Confirm TCMM is healthy on Vertex
#
# Idempotent re-runnable: each run uses a unique test namespace based
# on the start timestamp.

set -euo pipefail

RUDOLPH_UID="69df7853f6e15508f9da261e"
TEST_NS="aistudio-test-$(date +%s)"
DB_PATH="/home/rudol/veilguard/tcmm-data/veilguard/tcmm.db/archive.lance"

echo "=== test namespace: $TEST_NS ==="
echo "=== rudolph user_id: $RUDOLPH_UID ==="

# Load env (need GOOGLE_API_KEY)
set -a; source /home/rudol/veilguard/.env; set +a

if [ -z "${GOOGLE_API_KEY:-}" ]; then
    echo "FATAL: GOOGLE_API_KEY not set in /home/rudol/veilguard/.env"
    exit 1
fi
echo "  GOOGLE_API_KEY loaded (len=${#GOOGLE_API_KEY})"

# ── Step 1+2: stop systemd + start manual w/ AI Studio env ──────────
echo
echo "=== stopping systemd TCMM (Vertex mode) ==="
sudo systemctl stop veilguard-tcmm
sleep 2

echo
echo "=== starting manual TCMM with AI Studio env ==="
cd /home/rudol/veilguard
# Adapter selects AI Studio when project_id is empty AND api_key is set.
# It checks os.environ for VERTEX_API_KEY then GEMINI_API_KEY (NOT
# GOOGLE_API_KEY — see vertex_nlp_adapter.py:186), so we alias.
unset GOOGLE_CLOUD_PROJECT
export GEMINI_API_KEY="$GOOGLE_API_KEY"
export NLP_BACKEND=vertex  # the adapter class name; the AI-Studio-vs-Vertex switch is internal

# Foreground in background — capture output to a log file so we can
# tail it for diagnostics later. Log goes to /tmp so it doesn't
# pollute the prod log.
LOG=/tmp/tcmm-aistudio-test.log
PYTHONPATH=/home/rudol/veilguard/TCMM/TCMM \
TCMM_ROOT=/home/rudol/veilguard/TCMM/TCMM \
TCMM_DATA_DIR=/home/rudol/veilguard/tcmm-data \
nohup python3 mcp-tools/tcmm-service/server.py >"$LOG" 2>&1 &
MANUAL_PID=$!
echo "  manual TCMM PID: $MANUAL_PID  log: $LOG"

# ── Step 3: wait for /health ────────────────────────────────────────
echo
echo "=== waiting for /health to return 200 (max 60s) ==="
for i in $(seq 1 30); do
    sleep 2
    code=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8811/health --max-time 3 || echo 000)
    if [ "$code" = "200" ]; then
        echo "  ready after $((i*2))s"
        break
    fi
    if [ "$i" = "30" ]; then
        echo "  TIMEOUT — TCMM never came up. Tail of log:"
        tail -30 "$LOG"
        kill -TERM "$MANUAL_PID" 2>/dev/null || true
        sudo systemctl start veilguard-tcmm
        exit 1
    fi
done

echo
echo "=== confirm adapter is in AI Studio mode (look for 'api_key (AI Studio)' in log) ==="
grep -E "api_key.*AI Studio|oauth/ADC.*Vertex" "$LOG" | head -3 || echo "(no auth-mode log line)"

# ── Step 4: POST /ingest_turn with test blocks ──────────────────────
echo
echo "=== POSTing 2 test blocks to /ingest_turn ==="
PAYLOAD=$(cat <<JSON
{
  "conversation_id": "$TEST_NS",
  "user_id": "$RUDOLPH_UID",
  "items": [
    {
      "text": "AI Studio ingest test block 1: Sarel Strydom from Phishield reviewed the Veilguard installer 0.2.4 release on Tuesday from the London office. He reported 3 critical bugs to Rudolph Lamprecht.",
      "origin": "user"
    },
    {
      "text": "AI Studio ingest test block 2: PJ Schroeder confirmed the LanceDB compaction job completed in 1.4 seconds, freeing 1.7 GB. The TCMM service uptime was 6 days at restart. Inze Strydom is the new employee onboarded April 28.",
      "origin": "assistant"
    }
  ]
}
JSON
)
RESP=$(curl -s -X POST http://localhost:8811/ingest_turn \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD" --max-time 30)
echo "  response: $RESP"

# ── Step 5: wait for async NLP workers ──────────────────────────────
# Workers run in background threads/asyncio. NLP extraction takes
# 2-5s per block on AI Studio gemini-2.5-flash. 30s should be ample
# for 2 blocks.
echo
echo "=== waiting 30s for NLP workers to populate topic_dicts / entity_dicts ==="
sleep 30

# ── Step 6: query archive for the new blocks ────────────────────────
echo
echo "=== querying archive for blocks under namespace '$TEST_NS' ==="
python3 <<PYEOF
import lance, json
ds = lance.dataset("$DB_PATH")
arr = ds.to_table(
    columns=["aid","user_id","namespace","text","topics","entities",
             "topic_dicts","entity_dicts","claims","created_ts"]
).to_pylist()
new = [r for r in arr if r["namespace"] == "$TEST_NS"]
print(f"  found {len(new)} test row(s) under namespace '$TEST_NS'")
for r in new:
    print()
    print(f"  --- aid={r['aid']} ---")
    print(f"  text         : {(r.get('text') or '')[:80]}...")
    print(f"  topics       : {r.get('topics')}")
    print(f"  entities     : {r.get('entities')}")
    print(f"  topic_dicts  :")
    for td in (r.get('topic_dicts') or []):
        print(f"    {td}")
    print(f"  entity_dicts :")
    for ed in (r.get('entity_dicts') or []):
        print(f"    {ed}")
    print(f"  claims       : {r.get('claims')}")
PYEOF

# ── Step 7: kill manual TCMM, restart systemd ───────────────────────
echo
echo "=== stopping manual TCMM ==="
kill -TERM "$MANUAL_PID" 2>/dev/null || true
# wait up to 10s for graceful exit
for i in $(seq 1 5); do
    if ! kill -0 "$MANUAL_PID" 2>/dev/null; then break; fi
    sleep 2
done
if kill -0 "$MANUAL_PID" 2>/dev/null; then
    echo "  graceful stop didn't take; SIGKILL"
    kill -KILL "$MANUAL_PID" 2>/dev/null || true
fi

echo
echo "=== restarting systemd TCMM (back to Vertex) ==="
sudo systemctl start veilguard-tcmm
sleep 6

# ── Step 8: verify Vertex healthy ───────────────────────────────────
echo
echo "=== verify TCMM healthy on Vertex ==="
systemctl is-active veilguard-tcmm
curl -s -o /dev/null -w "  /health %{http_code}  time=%{time_total}s\n" \
    http://localhost:8811/health --max-time 5

echo
echo "=== test complete. log preserved at $LOG ==="
