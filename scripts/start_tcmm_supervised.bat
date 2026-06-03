@echo off
REM TCMM supervisor — keep tcmm-service on :8811 alive across crashes.
REM
REM This launches scripts/tcmm_supervisor.py in a *new* window so it
REM survives the parent shell closing.  Close the spawned window to
REM stop it; the supervisor catches Ctrl-C and terminates TCMM cleanly.
REM
REM See scripts/tcmm_supervisor.py for the watchdog logic.

setlocal
set "REPO=%~dp0.."
pushd "%REPO%" >nul

REM If supervisor is already running, do nothing (the script's own
REM port-check prevents double-spawn, but skipping the window-spawn
REM avoids stale pop-ups).
tasklist /FI "WINDOWTITLE eq TCMM-Supervisor" 2>nul | findstr /I "python.exe" >nul
if not errorlevel 1 (
    echo TCMM supervisor already running.
    popd >nul
    exit /b 0
)

REM --- Postgres backend pin (2026-06-03). The supervisor inherits this env;
REM WITHOUT it, server.py defaults TCMM_STORAGE to "lance" and the engine
REM silently runs on the wrong store. TCMM_DATA_DIR is pinned so the working
REM tier no longer depends on which dir server.py lives in. VEILGUARD_TCMM_SERVER_PY
REM points at the local-first dev-tree master (the single source of truth) so
REM there's no second copy to drift. ---
set "TCMM_STORAGE=postgres"
set "TCMM_VECTOR=postgres"
set "TCMM_SPARSE=postgres"
set "TCMM_DATABASE_URL=postgresql://tcmm:tcmm@localhost:5433/tcmm"
set "TCMM_ROOT=C:\Users\rudol\.gemini\antigravity\tcmm\TCMM"
set "TCMM_DATA_DIR=C:\Users\rudol\.veilguard\tcmm-data"
set "TCMM_PG_POOL_MAX=16"
set "TCMM_SKIP_NLP_GPU=1"
set "EMBED_BACKEND=local"
set "VEILGUARD_TCMM_SERVER_PY=%REPO%\services\tcmm-service\server.py"
set "VEILGUARD_TCMM_LOG_DIR=C:\Users\rudol\.veilguard\logs"

start "TCMM-Supervisor" cmd /c "python scripts\tcmm_supervisor.py & pause"

popd >nul
echo TCMM supervisor launched in a new window.
echo Close that window to stop the supervisor.
endlocal
