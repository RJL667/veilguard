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

echo [3/5] Starting forge on port 8810...
start "Veilguard-Forge" /min python services\forge\server.py --sse --port 8810
timeout /t 2 /nobreak >nul

echo [4/5] Starting TCMM on port 8811...
rem [USER_SCOPE_RECALL_2026_05_29] user-scoped cross-conversation recall
rem [RECALL_PERF_2026_05_29] TCMM_EAGER_BULK_WARM=1 pre-loads the user's
rem archive into cache (one ~200ms query) so user-scoped recall's per-aid
rem lookups are O(1) instead of a 320x Lance checkout-storm (3.6s->0.13s).
set VEILGUARD_RECALL_SCOPE=user
set VEILGUARD_RENDER_RECALL_SCOPE=user
set TCMM_EAGER_BULK_WARM=1
start "Veilguard-TCMM" /min cmd /c "set VEILGUARD_RECALL_SCOPE=user&& set VEILGUARD_RENDER_RECALL_SCOPE=user&& set TCMM_EAGER_BULK_WARM=1&& python services\tcmm-service\server.py --port 8811"
timeout /t 2 /nobreak >nul

echo [5/5] Starting Docker services (LibreChat + PII proxy)...
docker compose up -d

echo.
echo ============================================
echo   All services started!
echo ============================================
echo.
echo   host-exec:   http://localhost:8808/sse
echo   sub-agents:  http://localhost:8809/sse
echo   forge:       http://localhost:8810/sse
echo   TCMM:        http://localhost:8811/health
echo   PII proxy:   http://localhost:4000/health
echo   LibreChat:   http://localhost:3080
echo.
echo   Use system_health() tool in LibreChat to check status.
echo.
pause
