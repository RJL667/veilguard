; Veilguard Client Daemon — Inno Setup Installer
; Build: iscc installer.iss  (after pyinstaller veilguard.spec)
;
; MyAppVersion MUST match __version__ in veilguard_client.py.
; The auto-updater compares these values — if the manifest advertises
; 0.2.0 and the running client reports 0.1.x, the client downloads
; this installer and runs it silently with /VERYSILENT.
;
; Bumping a release is a one-line change here: edit MyAppVersion below.
; Every other version reference in this file (AppVersion, output filename,
; VersionInfo, UninstallDisplayName) derives from it.
;
; CloseApplications=yes + RestartApplications=yes:
;   Inno Setup detects VeilguardClient.exe is running, closes it,
;   installs the update, then relaunches it. Essential for auto-update
;   because otherwise the installer can't overwrite a running .exe.
;
; AppId is the stable upgrade-identity GUID. Never change this once
; published — it's how Inno recognises that 0.2.4 is "the same product
; as" 0.2.3 and should upgrade in place rather than installing in
; parallel. Without it, Inno falls back to AppName matching and the
; behaviour gets weird across versions (this caused Sarel's 0.2.4-
; installer-shows-0.2.1 confusion: Inno's upgrade prompt referred to
; the existing install's version, not the version about to be installed).

#define MyAppVersion "0.2.4"

[Setup]
AppId={{6A3BC56C-68CD-4F71-8466-0883395AF3EB}}
AppName=Veilguard Client
AppVersion={#MyAppVersion}
AppPublisher=Phishield
AppPublisherURL=https://phishield.ai
DefaultDirName={localappdata}\Veilguard
DisableProgramGroupPage=yes
OutputDir=installer_output
; Version-stamped output filename. Pre-0.2.5 every release shipped as
; the static ``VeilguardSetup.exe`` — when a user downloaded a fresh
; copy the browser appended (1), (2), (3)... to avoid clobbering the
; prior file in Downloads. With 19 nearly-identical filenames it was
; trivial to double-click the wrong one (this is exactly what bit
; Sarel: he ran an old 0.2.3 file thinking it was the freshly-fetched
; 0.2.4). Stamping the version into the filename means each release
; is a unique name in Downloads — no collisions, no (N) suffixes.
OutputBaseFilename=VeilguardSetup-{#MyAppVersion}
Compression=lzma2
SolidCompression=yes
PrivilegesRequired=lowest
WizardStyle=modern
CloseApplications=yes
RestartApplications=yes
; --- Version metadata embedded into the .exe itself ---
; Windows Explorer's right-click → Properties → Details tab reads
; these. Pre-0.2.4 the setup.exe shipped with empty FileVersion, so
; users couldn't tell which version they had until they ran it. Now
; PowerShell's (Get-Item ...).VersionInfo.FileVersion returns the
; right value, and Sarel can see "0.2.4" before double-clicking.
VersionInfoVersion={#MyAppVersion}.0
VersionInfoProductVersion={#MyAppVersion}.0
VersionInfoProductName=Veilguard Client
VersionInfoCompany=Phishield
VersionInfoDescription=Veilguard Client Daemon Setup
VersionInfoCopyright=Copyright (C) 2026 Phishield
; --- Cleaner Add/Remove Programs entry ---
UninstallDisplayName=Veilguard Client {#MyAppVersion}
; --- Upgrade UX ---
; UsePreviousAppDir=yes is the Inno default but explicit is better:
; on upgrade we install into the directory the user picked previously
; (almost always {localappdata}\Veilguard for unprivileged installs).
UsePreviousAppDir=yes
; DirExistsWarning=no suppresses the "the directory you selected
; already exists, would you like to install there anyway?" prompt
; on every upgrade. The dir always exists — we just installed there.
DirExistsWarning=no
; Uncomment when you have an icon:
; SetupIconFile=veilguard.ico

[Files]
Source: "dist\VeilguardClient\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs

[Icons]
Name: "{userdesktop}\Veilguard Client"; Filename: "{app}\VeilguardClient.exe"
Name: "{userstartup}\Veilguard Client"; Filename: "{app}\VeilguardClient.exe"; Comment: "Start Veilguard on login"

[Run]
; Launch after install — opens the setup page on first install, or
; just restarts the daemon on an auto-update (RestartApplications handles that).
Filename: "{app}\VeilguardClient.exe"; Description: "Launch Veilguard Client"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Was previously "{userappdata}\..\.veilguard" — the .. didn't resolve
; the way it looks because Inno's {userappdata} = ...\AppData\Roaming
; not just ...\AppData. So {userappdata}\.. landed in AppData and the
; final path was ...\AppData\.veilguard, which never exists, and the
; real ~/.veilguard config dir survived every uninstall (we hit this
; on Rudolph's machine). {%USERPROFILE} resolves to C:\Users\<him>
; cleanly via Inno's {sd}\Users\... or %USERPROFILE% env-var expansion.
;
; NOTE: this is intentionally commented out for now — preserving
; ~/.veilguard across uninstall/reinstall is actually the right user-
; level UX (their token survives an upgrade). Enable only if we want
; "real" uninstall to wipe credentials.
;Type: filesandordirs; Name: "{%USERPROFILE}\.veilguard"

[Code]
// No custom code needed — the exe handles first-run setup
