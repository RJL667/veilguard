"""Run all four demo scenarios sequentially and print a summary.

Usage:
    python -m demo.run_all

Each scenario runs in its own subprocess so they don't pollute each
other's process-global script + Lance singleton state.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Force UTF-8 stdout so Windows console doesn't choke on stray bytes.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SCENARIOS = [
    "demo.scenario_pattern_a_solo",
    "demo.scenario_pattern_b_delegation",
    "demo.scenario_pattern_c_fanout",
    "demo.scenario_critic_iterate",
]


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    results = []

    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

    for mod in SCENARIOS:
        print(f"\n{'#' * 72}")
        print(f"# Running {mod}")
        print(f"{'#' * 72}")
        proc = subprocess.run(
            [sys.executable, "-m", mod],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        # Print stdout (the demo's output) and filter out the [log] noise
        # for the summary, but keep the RESULT banner.  Strip non-ASCII
        # control chars that the demo's subprocess may have emitted.
        for line in proc.stdout.splitlines():
            if line.strip().startswith("[log]"):
                continue
            # Replace any remaining replacement-character (�) so
            # cp1252 stdout doesn't choke.
            try:
                print(line)
            except UnicodeEncodeError:
                print(line.encode("ascii", errors="replace").decode("ascii"))
        if proc.returncode != 0:
            print(f"\n*** {mod} EXITED WITH CODE {proc.returncode} ***")
            if proc.stderr:
                err = proc.stderr[-1000:]
                try:
                    print(err)
                except UnicodeEncodeError:
                    print(err.encode("ascii", errors="replace").decode("ascii"))

        pass_or_fail = "PASS" if "[PASS]" in proc.stdout else "FAIL"
        results.append((mod, pass_or_fail, proc.returncode))

    print(f"\n{'=' * 72}")
    print("  RUN_ALL SUMMARY")
    print(f"{'=' * 72}")
    for mod, verdict, code in results:
        print(f"  {mod:50} {verdict}  (exit {code})")

    all_pass = all(v == "PASS" and c == 0 for _, v, c in results)
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'FAILURES'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
