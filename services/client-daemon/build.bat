@echo off
REM Build Veilguard Client Installer
REM Prerequisites: pip install pyinstaller
REM               Inno Setup installed (iscc in PATH)

echo === Building Veilguard Client ===

echo [1/3] Installing dependencies...
REM Core deps + Phase C/D Windows UI (tray icon + toasts + viewer).
REM pystray + Pillow render the tray icon; winotify pops the approval
REM toast.  Pinned via requirements-windows.txt so the same versions
REM end up in the PyInstaller bundle every build.
pip install pyinstaller
pip install -r requirements-windows.txt

echo [2/3] Building executable with PyInstaller...
REM Use `python -m PyInstaller` rather than bare `pyinstaller` — the
REM bare command relies on Python's Scripts/ being on PATH which isn't
REM the case for `pip install --user` setups.  Earlier builds silently
REM failed when the bare command wasn't found; Inno Setup then errored
REM on "no dist/" and the .bat made it look like the whole pipeline
REM was at fault.
python -m PyInstaller veilguard.spec --noconfirm
if errorlevel 1 (
    echo *** PyInstaller failed.  Aborting before Inno Setup.
    exit /b 1
)

echo [3/3] Building installer with Inno Setup...
REM Check both common install paths.  Inno's interactive installer
REM offers per-user (LOCALAPPDATA) vs per-machine (Program Files);
REM whichever the user picked is where iscc.exe lives.  We'd rather
REM auto-detect than force one path.
set ISCC=
if exist "C:\Program Files (x86)\Inno Setup 6\iscc.exe" (
    set "ISCC=C:\Program Files (x86)\Inno Setup 6\iscc.exe"
) else if exist "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" (
    set "ISCC=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
)
if defined ISCC (
    "%ISCC%" installer.iss
    echo.
    echo === Done! Installer at: installer_output\VeilguardSetup-X.Y.Z.exe ===
) else (
    echo Inno Setup not found in either Program Files or LOCALAPPDATA.
    echo Standalone exe at: dist\VeilguardClient\VeilguardClient.exe
    echo Install Inno Setup 6 to build the .exe installer.
)

REM `pause` removed — keeping it would hang non-interactive runs
REM (build.bat invoked from automation, CI, etc.).  When run from
REM a developer's terminal, the prompt returns naturally after the
REM build completes.
