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

start "TCMM-Supervisor" cmd /c "python scripts\tcmm_supervisor.py & pause"

popd >nul
echo TCMM supervisor launched in a new window.
echo Close that window to stop the supervisor.
endlocal
