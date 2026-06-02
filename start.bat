@echo off
title Veilguard - Starting All Services
echo ============================================
echo   VEILGUARD - Service Launcher
echo ============================================
echo.

cd /d "%~dp0"

echo [1/5] Starting host-exec on port 8808...
start "Veilguard-HostExec" /min python services\host-exec\server.py --sse --port 8808
timeout /t 2 /nobreak >nul

echo [2/5] Starting sub-agents on port 8809...
start "Veilguard-SubAgents" /min python services\sub-agents\server.py --sse --port 8809
timeout /t 2 /nobreak >nul

echo [3/6] Starting forge on port 8810...
start "Veilguard-Forge" /min python services\forge\server.py --sse --port 8810
timeout /t 2 /nobreak >nul

echo [4/6] Starting veilguard-mcp (search_memory/traverse_memory) on port 8812...
rem [MCP_8812_FIX_2026-06-01] This server was MISSING from start.bat — it was
rem chronically DOWN, so LibreChat hit a connection TIMEOUT on the configured
rem `veilguard` MCP server (librechat.yaml :8812) during tool discovery on
rem EVERY chat turn. That (plus forge being down) was the fixed ~8s that made
rem warm UI turns ~10s even though the proxy->agent-runtime path is ~2s.
start "Veilguard-MCP" /min python services\veilguard-mcp\server.py
timeout /t 2 /nobreak >nul

echo [5/6] Starting TCMM on port 8811...
rem [USER_SCOPE_RECALL_2026_05_29] user-scoped cross-conversation recall
rem [RECALL_PERF_2026_05_29] TCMM_EAGER_BULK_WARM=1 pre-loads the user's
rem archive into cache (one ~200ms query) so user-scoped recall's per-aid
rem lookups are O(1) instead of a 320x Lance checkout-storm (3.6s->0.13s).
set VEILGUARD_RECALL_SCOPE=user
set VEILGUARD_RENDER_RECALL_SCOPE=user
set TCMM_EAGER_BULK_WARM=1
rem [TCMM_ROOT_FIX_2026-06-01] .env line 53 has a stale container path
rem TCMM_ROOT=/opt/tcmm; on the Windows host that breaks `from adapters...`
rem (ModuleNotFoundError) because sys.path.insert gets a nonexistent dir.
rem Set the real canonical tree here so it wins over .env (load_dotenv
rem uses override=False, so a process-env value takes precedence).
rem Also pin NLP_BACKEND=ai_studio (host has no GPU; local NLP needs one).
start "Veilguard-TCMM" /min cmd /c "set TCMM_ROOT=C:\Users\rudol\.gemini\antigravity\tcmm\TCMM&& set NLP_BACKEND=ai_studio&& set VEILGUARD_RECALL_SCOPE=user&& set VEILGUARD_RENDER_RECALL_SCOPE=user&& set TCMM_EAGER_BULK_WARM=1&& python services\tcmm-service\server.py --port 8811"
timeout /t 2 /nobreak >nul

echo [6/6] Starting Docker services (LibreChat + PII proxy)...
docker compose up -d

echo.
echo ============================================
echo   All services started!
echo ============================================
echo.
echo   host-exec:     http://localhost:8808/sse
echo   sub-agents:    http://localhost:8809/sse
echo   forge:         http://localhost:8810/sse
echo   veilguard-mcp: http://localhost:8812/sse
echo   TCMM:          http://localhost:8811/health
echo   PII proxy:   http://localhost:4000/health
echo   LibreChat:   http://localhost:3080
echo.
echo   Use system_health() tool in LibreChat to check status.
echo.
pause
