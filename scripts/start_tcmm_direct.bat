@echo off
REM [SUPERVISOR_RETIRED_2026_06_04] Direct TCMM launcher — NO supervisor/watchdog.
REM The crash-restart supervisor (scripts\tcmm_supervisor.py) was a LanceDB
REM memory-contention band-aid. On Postgres that contention is gone, so the
REM supervisor only (a) spawned a pop-up "Press any key" window on every restart
REM and (b) its dead_after watchdog KILLED the server during legitimately slow
REM turns (15-30s generations). This runs server.py directly on :8811, minimized,
REM logging to .veilguard\logs\tcmm-direct.log. Close the minimized window to stop.
setlocal
set "REPO=%~dp0.."
REM [UTF8_2026-06-04] Force UTF-8 stdio. The dream engine prints Unicode (→, box
REM glyphs, emoji); under Windows cp1252 stdout (esp. when logs are redirected to
REM a file) those throw UnicodeEncodeError and ABORT the dream cycle mid-run
REM (right after the arc phase — before identity bootstrap / meta clustering).
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "TCMM_STORAGE=postgres"
set "TCMM_VECTOR=postgres"
set "TCMM_SPARSE=postgres"
set "TCMM_DATABASE_URL=postgresql://tcmm:tcmm@localhost:5433/tcmm"
set "TCMM_ROOT=C:\Users\rudol\.gemini\antigravity\tcmm\TCMM"
set "TCMM_DATA_DIR=C:\Users\rudol\.veilguard\tcmm-data"
set "TCMM_PG_POOL_MAX=16"
set "TCMM_SKIP_NLP_GPU=1"
set "EMBED_BACKEND=local"
REM [DREAM_BACKEND_2026-06-04] Dream synthesizer = AI Studio Gemini (cloud, via
REM GOOGLE_API_KEY in .env). Unset => the adapter falls back to a local llama.cpp
REM GGUF that (a) crashes the process when synthesizing on this CPU/GPU-locked
REM box and (b) leaves synth.backend falsy so the fact-sheet LLM distillation
REM pass is skipped -> raw fragmentary fact sheets. "vertex" fixes both.
set "DREAM_BACKEND=vertex"
set "VEILGUARD_TCMM_SERVER_PY=%REPO%\services\tcmm-service\server.py"
set "VEILGUARD_TCMM_LOG_DIR=C:\Users\rudol\.veilguard\logs"
cd /d "%REPO%\services\tcmm-service"
echo Starting TCMM (direct, no supervisor) on :8811 ...
start "tcmm" /MIN cmd /c "python server.py > C:\Users\rudol\.veilguard\logs\tcmm-direct.log 2>&1"
echo Launched. Logs: C:\Users\rudol\.veilguard\logs\tcmm-direct.log
endlocal
