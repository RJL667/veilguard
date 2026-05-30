# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for Veilguard Client Daemon
#
# Build: pyinstaller veilguard.spec
# Output: dist/VeilguardClient/VeilguardClient.exe
#
# IMPORTANT: clear build/ + dist/ before rebuilding when this spec
# changes — PyInstaller caches Analysis output and can serve a stale
# bundle if you forget.  build.bat doesn't do this for you (we don't
# want to nuke the cache on every successful incremental build).

from PyInstaller.utils.hooks import collect_all

# ── Collect everything from installed pip packages ───────────────────
# collect_all() returns (submodules, datas, binaries) for a package —
# more aggressive than just collect_submodules + collect_data_files
# because it also picks up native .pyd / .dll files and recursive
# package data.  pystray's Windows backend (pystray._win32) silently
# fails to load without this — it does a runtime `from . import _win32`
# that PyInstaller's static tracer can't follow on its own.

# PyInstaller's collect_all() returns (datas, binaries, hiddenimports)
# — NOT (submodules, datas, binaries).  Got this wrong once already
# and the spec exploded because hiddenimports got the data-tuples list.
_pystray_data,  _pystray_bins,  _pystray_subs  = collect_all('pystray')
_PIL_data,      _PIL_bins,      _PIL_subs      = collect_all('PIL')
_winotify_data, _winotify_bins, _winotify_subs = collect_all('winotify')

a = Analysis(
    ['veilguard_client.py'],
    pathex=[],
    binaries=_pystray_bins + _PIL_bins + _winotify_bins,
    # Bundle the local `daemon/` subpackage as a tree of source files.
    # Listing it as a hiddenimport alone doesn't work — PyInstaller
    # needs the .py files actually copied into the bundle so the
    # frozen import can find them at runtime.  Tuple shape is
    # (source_path, archive_path).
    datas=[
        ('daemon', 'daemon'),
    ] + _pystray_data + _PIL_data + _winotify_data,
    hiddenimports=[
        'websockets',
        'websockets.legacy',
        'websockets.legacy.client',
        'httpx',
        'httpx._transports',
        'httpx._transports.default',
        'yaml',
        'setup_server',
        # Phase C/D daemon submodules — listed both here AND copied via
        # `datas` above so the frozen importer has both metadata and
        # source.  Without datas, PyInstaller's PYZ might compile the
        # module but lose its package structure.
        'daemon',
        'daemon.tasks',
        'daemon.toast',
        'daemon.tray',
        'daemon.deeplink',
        'daemon.single_instance',
        # tkinter for the task viewer popup.  ttk subpackage explicit
        # because some PyInstaller versions miss it under tkinter alone.
        'tkinter',
        'tkinter.ttk',
        # multiprocessing.connection — needed by daemon.single_instance
        # for the Listener/Client handoff.  Stdlib but easy to miss.
        'multiprocessing.connection',
    ] + _pystray_subs + _PIL_subs + _winotify_subs,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # tkinter is required by daemon/tray.py's task-viewer popup window.
    # Removed from excludes — adds ~3 MB to the installer but eliminates
    # a runtime ImportError when the user clicks the tray.
    excludes=['unittest', 'test'],
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
    # console=False so users don't see the Python console window
    # alongside the tray icon.  All logs route to a rotating file at
    # %LOCALAPPDATA%\Veilguard\daemon.log (configured in main()), which
    # the tray viewer's "Logs" tab tails in real time.
    console=False,
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
