#!/usr/bin/env python3
"""
publish_release.py — Ship a new Veilguard client build to production.

Flow:
  1. Read __version__ from veilguard_client.py
  2. Verify installer_output/VeilguardSetup.exe was built from that version
     by checking the modification time is newer than veilguard_client.py
  3. Compute SHA-256 of the installer
  4. Upload the installer + a version.json manifest to the VM downloads dir
     via gcloud compute scp
  5. Tell user what clients will see

Usage:
    python publish_release.py                      # Uses defaults
    python publish_release.py --skip-upload        # Just generate manifest locally

Required first: run build.bat to produce installer_output/VeilguardSetup.exe.

The daemon polls GET /api/client/latest every 30min. When it sees a higher
version than its built-in __version__, it downloads the installer and runs
it silently — the user never has to reinstall manually again.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLIENT_PY = HERE / "veilguard_client.py"
INSTALLER_DIR = HERE / "installer_output"
MANIFEST_LOCAL = INSTALLER_DIR / "version.json"

# Remote targets (GCE VM)
VM_NAME = "veilguard-prod"
VM_ZONE = "us-central1-a"
REMOTE_DIR = "/home/rudol/veilguard/downloads"


def _resolve_gcloud() -> str:
    """Find the gcloud executable, accounting for the Windows .cmd wrapper.

    On Windows the gcloud SDK installs ``gcloud.cmd`` (a batch wrapper)
    rather than a native ``gcloud.exe``. ``subprocess.run(["gcloud", ...])``
    on Windows calls CreateProcess directly, which does NOT consult
    PATHEXT — so it fails with ``FileNotFoundError`` even though
    ``gcloud`` resolves fine from a shell. ``shutil.which`` does honor
    PATHEXT, so it returns the full path to ``gcloud.cmd`` here.
    """
    path = shutil.which("gcloud")
    if not path:
        print(
            "ERROR: 'gcloud' not on PATH. Install the Google Cloud SDK or add\n"
            "  C:\\Users\\<you>\\AppData\\Local\\Google\\Cloud SDK\\google-cloud-sdk\\bin\n"
            "to PATH.",
            file=sys.stderr,
        )
        sys.exit(1)
    return path


def read_version() -> str:
    """Parse __version__ = "X.Y.Z" from veilguard_client.py."""
    text = CLIENT_PY.read_text(encoding="utf-8")
    m = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    if not m:
        print("ERROR: __version__ not found in veilguard_client.py", file=sys.stderr)
        sys.exit(1)
    return m.group(1)


def read_iss_version() -> str:
    """Parse #define MyAppVersion "X.Y.Z" from installer.iss to catch drift.

    installer.iss centralises the version in a single ``#define MyAppVersion``
    at the top of the file (every other version reference — AppVersion,
    OutputBaseFilename, VersionInfo*, UninstallDisplayName — derives from
    it). We read that define here so the drift-check vs. __version__ in
    veilguard_client.py stays accurate after the 0.2.5 refactor.
    """
    iss = HERE / "installer.iss"
    if not iss.is_file():
        return ""
    pattern = re.compile(r'^\s*#define\s+MyAppVersion\s+"([^"]+)"\s*$')
    for line in iss.read_text(encoding="utf-8").splitlines():
        m = pattern.match(line)
        if m:
            return m.group(1)
    return ""


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-upload", action="store_true",
                        help="Write version.json locally but don't scp to VM")
    parser.add_argument("--changelog", default="",
                        help="Changelog text to embed in manifest")
    args = parser.parse_args()

    version = read_version()
    iss_version = read_iss_version()
    print(f"[version] __version__  = {version}")
    print(f"[version] installer.iss = {iss_version}")

    if iss_version and iss_version != version:
        print(
            f"WARNING: installer.iss MyAppVersion ({iss_version}) does not match "
            f"__version__ ({version}). Fix before publishing."
        )
        sys.exit(1)

    # Version-stamped filename. installer.iss writes the .exe as
    # VeilguardSetup-<version>.exe; we compute the same name here so
    # the upload + manifest agree. This is the post-0.2.5 convention —
    # see installer.iss [Setup].OutputBaseFilename for why.
    installer_name = f"VeilguardSetup-{version}.exe"
    installer_path = INSTALLER_DIR / installer_name

    if not installer_path.is_file():
        print(f"ERROR: {installer_path} not found. Run build.bat first.", file=sys.stderr)
        sys.exit(1)

    client_mtime = CLIENT_PY.stat().st_mtime
    inst_mtime = installer_path.stat().st_mtime
    if inst_mtime < client_mtime:
        print(
            f"WARNING: {installer_path.name} is older than {CLIENT_PY.name}. "
            f"Did you forget to run build.bat after editing the client?"
        )
        sys.exit(1)

    sha256 = compute_sha256(installer_path)
    size_mb = installer_path.stat().st_size / (1024 * 1024)
    print(f"[installer] {installer_path.name} — {size_mb:.1f}MB")
    print(f"[installer] sha256 = {sha256}")

    manifest = {
        "version": version,
        "url": f"/download/{installer_name}",
        "filename": installer_name,
        "sha256": sha256,
        "size_bytes": installer_path.stat().st_size,
        "min_required": "0.1.0",
        "changelog": args.changelog or f"Release {version}",
        "published_at": int(time.time()),
    }
    MANIFEST_LOCAL.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[manifest] wrote {MANIFEST_LOCAL}")

    if args.skip_upload:
        print("\n--skip-upload — done locally. scp manually to ship:")
        print(f"  gcloud compute scp {installer_path} {VM_NAME}:{REMOTE_DIR}/ --zone={VM_ZONE}")
        print(f"  gcloud compute scp {MANIFEST_LOCAL} {VM_NAME}:{REMOTE_DIR}/ --zone={VM_ZONE}")
        return

    # Upload installer first, then manifest. This ordering matters: if a
    # client hits the manifest endpoint mid-upload, we want it to see
    # the OLD manifest pointing at the OLD installer, not the new
    # manifest pointing at a half-uploaded new installer.
    gcloud = _resolve_gcloud()
    print("\n[upload] Installer first...")
    subprocess.run(
        [gcloud, "compute", "scp",
         str(installer_path).replace("\\", "/"),
         f"{VM_NAME}:{REMOTE_DIR}/{installer_name}",
         f"--zone={VM_ZONE}"],
        check=True,
    )
    print("[upload] Manifest second (this is the atomic flip)...")
    subprocess.run(
        [gcloud, "compute", "scp",
         str(MANIFEST_LOCAL).replace("\\", "/"),
         f"{VM_NAME}:{REMOTE_DIR}/version.json",
         f"--zone={VM_ZONE}"],
        check=True,
    )

    print(f"\n[done] Released v{version} as {installer_name}. All connected daemons will update within 30min.")
    print(f"       Manual smoke: curl https://veilguard.phishield.com/api/sub-agents/api/client/latest")


if __name__ == "__main__":
    main()
