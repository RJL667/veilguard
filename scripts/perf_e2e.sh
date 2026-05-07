#!/bin/bash
# E2E perf v3 — robust against silent failures.
# Drops `set -e` (the previous version died silently inside record_call
# from an empty grep | tail | awk pipeline). Drops the systemctl restart
# (each fresh conv_id triggers a cold TCMM session bootstrap inside the
# already-running service, which is the same "cold" we want to measure).
ANTHROPIC_KEY=$(grep -E "^ANTHROPIC_API_KEY=" /home/rudol/veilguard/.env | cut -d= -f2- | tr -d '"')
RUDOLPH_UID=69df7853f6e15508f9da261e
URL=http://localhost:4000/anthropic/v1/messages
RUN_ID="run-$(date +%s)-$(uuidgen | head -c 8)"

record_call() {
  local conv_id="$1"
  local msg="$2"
  local label="$3"
  local pass="$4"

  local payload
  payload=$(printf '{"model":"claude-haiku-4-5","max_tokens":50,"messages":[{"role":"user","content":%s}]}' \
    "$(jq -nR --arg s "$msg" '$s')")

  local result
  result=$(curl -s -o /tmp/_perf_resp.json \
    -w "%{time_total}|%{time_starttransfer}|%{http_code}|%{size_download}" \
    -X POST "$URL" \
    -H "x-api-key: $ANTHROPIC_KEY" \
    -H "anthropic-version: 2023-06-01" \
    -H "Content-Type: application/json" \
    -H "x-conversation-id: $conv_id" \
    -H "x-user-id: $RUDOLPH_UID" \
    -d "$payload" 2>/dev/null)

  local e2e=$(echo "$result" | cut -d'|' -f1)
  local ttfb=$(echo "$result" | cut -d'|' -f2)
  local status=$(echo "$result" | cut -d'|' -f3)
  local bytes_recv=$(echo "$result" | cut -d'|' -f4)
  local e2e_ms=$(awk -v t="${e2e:-0}" 'BEGIN{print int(t*1000)}')
  local ttfb_ms=$(awk -v t="${ttfb:-0}" 'BEGIN{print int(t*1000)}')

  sleep 1
  local pre_log
  pre_log=$(sudo grep "\\[PRE\\] session=$conv_id" /var/log/veilguard-tcmm.log 2>/dev/null | tail -1)
  local pre_ms=0
  if [ -n "$pre_log" ]; then
    local took=$(echo "$pre_log" | grep -oP 'took=\K[0-9.]+')
    pre_ms=$(awk -v t="${took:-0}" 'BEGIN{print int(t*1000)}')
  fi
  local recalled=0
  recalled=$(echo "$pre_log" | grep -oP 'recalled=\K\d+')
  : "${recalled:=0}"
  local prompt_chars=0
  prompt_chars=$(echo "$pre_log" | grep -oP 'prompt_words=\K\d+')
  : "${prompt_chars:=0}"

  local cache_line
  cache_line=$(sudo docker logs --since 30s veilguard-pii-proxy-1 2>&1 \
    | grep "\\[CACHE\\] conv=" | grep -F "${conv_id:0:8}" | tail -1)
  local tin=0 tout=0 cc=0 cr=0
  tin=$(echo "$cache_line" | grep -oP 'input=\K\d+'); : "${tin:=0}"
  tout=$(echo "$cache_line" | grep -oP 'output=\K\d+'); : "${tout:=0}"
  cc=$(echo "$cache_line" | grep -oP 'create=\K\d+'); : "${cc:=0}"
  cr=$(echo "$cache_line" | grep -oP 'read=\K\d+'); : "${cr:=0}"

  printf "  %-22s e2e=%5sms  pre=%5sms  recalled=%s  tin=%s  tout=%s  cc=%s  cr=%s  http=%s\n" \
    "$label" "$e2e_ms" "$pre_ms" "$recalled" "$tin" "$tout" "$cc" "$cr" "$status"

  PYTHONPATH=/home/rudol/.local/lib/python3.10/site-packages \
    python3 /tmp/perf_audit.py record \
      --run_id "$RUN_ID" --pass_label "$pass" --call_label "$label" \
      --conv "$conv_id" --user "$RUDOLPH_UID" --model "claude-haiku-4-5" \
      --e2e_ms "$e2e_ms" --ttfb_ms "$ttfb_ms" --status "$status" \
      --bytes_recv "$bytes_recv" --pre_ms "$pre_ms" --recalled "$recalled" \
      --prompt_chars "$prompt_chars" --tin "$tin" --tout "$tout" \
      --cc "$cc" --cr "$cr" 2>&1 | tail -1
}

echo "RUN_ID=$RUN_ID"
echo
echo "############# COLD PASS — 3 fresh conv_ids #############"
TS=$(date +%s)
record_call "perf-cold-${TS}-1" "Hello, can you say hi back?" "cold-1" "cold"
record_call "perf-cold-${TS}-2" "What is the capital of France?" "cold-2" "cold"
record_call "perf-cold-${TS}-3" "Tell me a one-line joke." "cold-3" "cold"

echo
echo "############# WARM PASS — 1 conv_id × 5 #############"
WARM_CONV="perf-warm-${TS}"
for i in 1 2 3 4 5; do
  record_call "$WARM_CONV" "Test $i — keep brief, 10 words max." "warm-$i" "warm"
  sleep 1
done

echo
echo "############# Aggregate (this run only) #############"
PYTHONPATH=/home/rudol/.local/lib/python3.10/site-packages \
  python3 /tmp/perf_audit.py summary --run_id "$RUN_ID"
