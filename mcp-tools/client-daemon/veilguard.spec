# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for Veilguard Client Daemon
#
# Build: pyinstaller veilguard.spec
# Output: dist/VeilguardClient/VeilguardClient.exe

a = Analysis(
    ['veilguard_client.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'websockets',
        'websockets.legacy',
        'websockets.legacy.client',
        'httpx',
        'httpx._transports',
        'httpx._transports.default',
        'yaml',
        'setup_server',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'unittest', 'test'],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='VeilguardClient',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,       # Show console window for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    icon=None,           # TODO: Add veilguard.ico
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='VeilguardClient',
)
