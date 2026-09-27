"""The weekly live cycle in one command: score, refresh, record.

The prospective pick log is the only evidence about the model that was not
built from data it has already seen, and it can only grow one deadline at a
time. GW4 and GW5 of 2026/27 were lost because nothing ran before those
deadlines. This runs the three steps in the only order that keeps the log
honest:

1. Score every finished Gameweek already recorded (official results only).
2. Refresh the live recommendation for the next deadline from the shipped model.
3. Record that pick, which `pick_history.record` refuses once the deadline passes.

Run it any time before a deadline; re-running before the deadline replaces the
draft with the latest recommendation, and a scored Gameweek is never rewritten.

    python analysis/weekly_cycle.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pick_history


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    print("1/3 scoring finished Gameweeks")
    pick_history.score()

    print("\n2/3 refreshing the live recommendation")
    refresh = subprocess.run(
        [sys.executable, str(ROOT / "analysis" / "calibrate_model.py"), "--refresh-current"],
        cwd=ROOT,
    )
    if refresh.returncode != 0:
        print("Refresh failed; not recording a stale pick.")
        return refresh.returncode

    print("\n3/3 recording this deadline's pick")
    try:
        pick_history.record()
    except RuntimeError as error:
        print(f"Not recorded: {error}")
    print()
    pick_history.report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
