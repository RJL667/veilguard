; Veilguard Client Daemon — Inno Setup Installer
; Build: iscc installer.iss  (after pyinstaller veilguard.spec)

[Setup]
AppName=Veilguard Client
AppVersion=0.1.0
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
; Uncomment when you have an icon:
; SetupIconFile=veilguard.ico

[Files]
Source: "dist\VeilguardClient\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs

[Icons]
Name: "{userdesktop}\Veilguard Client"; Filename: "{app}\VeilguardClient.exe"
Name: "{userstartup}\Veilguard Client"; Filename: "{app}\VeilguardClient.exe"; Comment: "Start Veilguard on login"

[Run]
; Launch after install — opens the setup page
Filename: "{app}\VeilguardClient.exe"; Description: "Launch Veilguard Client"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{userappdata}\..\.veilguard"

[Code]
// No custom code needed — the exe handles first-run setup
