; Veilguard Client Daemon — Inno Setup Installer
; Build: iscc installer.iss  (after pyinstaller veilguard.spec)
;
; AppVersion MUST match __version__ in veilguard_client.py.
; The auto-updater compares these values — if the manifest advertises
; 0.2.0 and the running client reports 0.1.x, the client downloads
; this installer and runs it silently with /VERYSILENT.
;
; CloseApplications=yes + RestartApplications=yes:
;   Inno Setup detects VeilguardClient.exe is running, closes it,
;   installs the update, then relaunches it. Essential for auto-update
;   because otherwise the installer can't overwrite a running .exe.

[Setup]
AppName=Veilguard Client
AppVersion=0.2.1
AppPublisher=Phishield
AppPublisherURL=https://phishield.ai
DefaultDirName={localappdata}\Veilguard
DisableProgramGroupPage=yes
OutputDir=installer_output
OutputBaseFilename=VeilguardSetup
Compression=lzma2
SolidCompression=yes
PrivilegesRequired=lowest
WizardStyle=modern
CloseApplications=yes
RestartApplications=yes
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
Type: filesandordirs; Name: "{userappdata}\..\.veilguard"

[Code]
// No custom code needed — the exe handles first-run setup
