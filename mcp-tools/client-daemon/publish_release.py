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
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLIENT_PY = HERE / "veilguard_client.py"
INSTALLER = HERE / "installer_output" / "VeilguardSetup.exe"
MANIFEST_LOCAL = HERE / "installer_output" / "version.json"

# Remote targets (GCE VM)
VM_NAME = "veilguard-prod"
VM_ZONE = "us-central1-a"
REMOTE_DIR = "/home/rudol/veilguard/downloads"


def read_version() -> str:
    """Parse __version__ = "X.Y.Z" from veilguard_client.py."""
    text = CLIENT_PY.read_text(encoding="utf-8")
    m = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    if not m:
        print("ERROR: __version__ not found in veilguard_client.py", file=sys.stderr)
        sys.exit(1)
    return m.group(1)


def read_iss_version() -> str:
    """Parse AppVersion=X.Y.Z from installer.iss to catch drift."""
    iss = HERE / "installer.iss"
    if not iss.is_file():
        return ""
    for line in iss.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("AppVersion"):
            return line.split("=", 1)[1].strip()
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
            f"WARNING: installer.iss AppVersion ({iss_version}) does not match "
            f"__version__ ({version}). Fix before publishing."
        )
        sys.exit(1)

    if not INSTALLER.is_file():
        print(f"ERROR: {INSTALLER} not found. Run build.bat first.", file=sys.stderr)
        sys.exit(1)

    client_mtime = CLIENT_PY.stat().st_mtime
    inst_mtime = INSTALLER.stat().st_mtime
    if inst_mtime < client_mtime:
        print(
            f"WARNING: {INSTALLER.name} is older than {CLIENT_PY.name}. "
            f"Did you forget to run build.bat after editing the client?"
        )
        sys.exit(1)

    sha256 = compute_sha256(INSTALLER)
    size_mb = INSTALLER.stat().st_size / (1024 * 1024)
    print(f"[installer] {INSTALLER.name} — {size_mb:.1f}MB")
    print(f"[installer] sha256 = {sha256}")

    manifest = {
        "version": version,
        "url": "/download/VeilguardSetup.exe",
        "filename": "VeilguardSetup.exe",
        "sha256": sha256,
        "size_bytes": INSTALLER.stat().st_size,
        "min_required": "0.1.0",
        "changelog": args.changelog or f"Release {version}",
        "published_at": int(time.time()),
    }
    MANIFEST_LOCAL.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[manifest] wrote {MANIFEST_LOCAL}")

    if args.skip_upload:
        print("\n--skip-upload — done locally. scp manually to ship:")
        print(f"  gcloud compute scp {INSTALLER} {VM_NAME}:{REMOTE_DIR}/ --zone={VM_ZONE}")
        print(f"  gcloud compute scp {MANIFEST_LOCAL} {VM_NAME}:{REMOTE_DIR}/ --zone={VM_ZONE}")
        return

    # Upload installer first, then manifest. This ordering matters: if a
    # client hits the manifest endpoint mid-upload, we want it to see
    # the OLD manifest pointing at the OLD installer, not the new
    # manifest pointing at a half-uploaded new installer.
    print("\n[upload] Installer first...")
    subprocess.run(
        ["gcloud", "compute", "scp",
         str(INSTALLER).replace("\\", "/"),
         f"{VM_NAME}:{REMOTE_DIR}/VeilguardSetup.exe",
         f"--zone={VM_ZONE}"],
        check=True,
    )
    print("[upload] Manifest second (this is the atomic flip)...")
    subprocess.run(
        ["gcloud", "compute", "scp",
         str(MANIFEST_LOCAL).replace("\\", "/"),
         f"{VM_NAME}:{REMOTE_DIR}/version.json",
         f"--zone={VM_ZONE}"],
        check=True,
    )

    print(f"\n[done] Released v{version}. All connected daemons will update within 30min.")
    print(f"       Manual smoke: curl https://veilguard.phishield.com/api/sub-agents/api/client/latest")


if __name__ == "__main__":
    main()
