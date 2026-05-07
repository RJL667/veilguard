#!/bin/bash
cd ~/veilguard
export GOOGLE_CLOUD_PROJECT=rugged-sunbeam-492106-j1
export VERTEX_REGION=us-central1
export NLP_BACKEND=vertex
export EMBED_BACKEND=vertex
export TCMM_ROOT=~/veilguard/TCMM
export TCMM_DATA_DIR=~/veilguard/tcmm-data
export PYTHONPATH=~/veilguard/TCMM
export TCMM_STORAGE=lance
export TCMM_VECTOR=lance
export TCMM_SPARSE=lance
export TCMM_LANCE_DB=veilguard

# Source .env for API keys
set -a
source .env 2>/dev/null
set +a

echo '[1/4] Starting host-exec on :8808...'
nohup python3 mcp-tools/host-exec/server.py --sse --port 8808 > /tmp/host-exec.log 2>&1 &

echo '[2/4] Starting sub-agents on :8809...'
nohup python3 mcp-tools/sub-agents/server.py --sse --port 8809 > /tmp/sub-agents.log 2>&1 &

echo '[3/4] Starting forge on :8810...'
nohup python3 mcp-tools/forge/server.py --sse --port 8810 > /tmp/forge.log 2>&1 &

sleep 2

echo '[4/4] Starting TCMM on :8811...'
cd ~/veilguard
nohup python3 mcp-tools/tcmm-service/server.py > /tmp/tcmm.log 2>&1 &

echo 'All host services starting...'
sleep 5

# Check
for port in 8808 8809 8810 8811; do
  if ss -tlnp | grep -q ": "; then
    echo "  : UP"
  else
    echo "  : DOWN"
  fi
done
