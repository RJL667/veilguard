@echo off
REM Build Veilguard Client Installer
REM Prerequisites: pip install pyinstaller
REM               Inno Setup installed (iscc in PATH)

echo === Building Veilguard Client ===

echo [1/3] Installing dependencies...
pip install websockets httpx pyyaml pyinstaller

echo [2/3] Building executable with PyInstaller...
pyinstaller veilguard.spec --noconfirm

echo [3/3] Building installer with Inno Setup...
if exist "C:\Program Files (x86)\Inno Setup 6\iscc.exe" (
    "C:\Program Files (x86)\Inno Setup 6\iscc.exe" installer.iss
    echo.
    echo === Done! Installer at: installer_output\VeilguardSetup.exe ===
) else (
    echo Inno Setup not found. Standalone exe at: dist\VeilguardClient\VeilguardClient.exe
    echo Install Inno Setup 6 to build the .exe installer.
)

pause
