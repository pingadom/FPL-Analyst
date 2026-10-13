"""The unattended live job: run it every hour and it decides what is needed.

Two problems with the daily agent-run task motivated this:

* it needed a permission prompt answered, so on some days nothing ran; and
* it recorded whatever the recommendation was at an arbitrary time of day, often
  days before the deadline, so late team news never reached the logged pick.
  Elite managers field about 0.05 starters a week who do not play; the model
  fielded about 0.17, and the difference is mostly late news.

This script has no prompts and is cheap when there is nothing to do, so it can
run hourly from Windows Task Scheduler. Each run:

1. scores any Gameweek FPL has confirmed (cheap);
2. refreshes and records the pick when the deadline is within FINAL_WINDOW
   and no pick has been recorded inside that window yet (the "deadline pick"),
   or when the last recorded draft is older than DRAFT_MAX_AGE (so the site
   stays fresh between deadlines);
3. commits only the three live data files, never pushes, and only on the
   expected branch.

    python analysis/live_job.py            # do whatever is due
    python analysis/live_job.py --dry-run  # say what would be done
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pick_history

ROOT = Path(__file__).resolve().parents[1]
BRANCH = "codex/full-audit-repair"
FINAL_WINDOW = timedelta(hours=3)
DRAFT_MAX_AGE = timedelta(hours=20)
DATA_FILES = (
    "app/data/pick-history.json",
    "app/data/model-results.json",
    "app/data/current-players.json",
)


@dataclass(frozen=True)
class Plan:
    refresh: bool
    reason: str


def decide(now: datetime, deadline: datetime | None, last_recorded: datetime | None) -> Plan:
    """Whether to refresh and record now. Pure, so it can be tested.

    `last_recorded` is when the pick for the coming deadline was last written,
    or None if it has not been recorded at all.
    """
    if deadline is None:
        return Plan(False, "no upcoming deadline known")
    left = deadline - now
    if left <= timedelta(0):
        return Plan(False, "deadline passed; waiting for the next one")
    if left <= FINAL_WINDOW:
        if last_recorded is None or last_recorded < deadline - FINAL_WINDOW:
            return Plan(True, f"deadline pick ({left.total_seconds() / 3600:.1f}h to go)")
        return Plan(False, "deadline pick already recorded")
    if last_recorded is None:
        return Plan(True, "no draft recorded for this deadline yet")
    if now - last_recorded >= DRAFT_MAX_AGE:
        return Plan(True, "draft older than a day")
    return Plan(False, "draft is fresh; deadline not close")


def next_deadline() -> tuple[int | None, datetime | None]:
    """The coming Gameweek and its deadline, from FPL (falls back to the artifact)."""
    try:
        bootstrap = pick_history._get(pick_history.BOOTSTRAP)
        event = next((e for e in bootstrap["events"] if e.get("is_next")), None)
        if event:
            return int(event["id"]), _parse(event["deadline_time"])
    except Exception as error:  # the network is optional for the decision
        print(f"FPL bootstrap unavailable ({error}); using the artifact's deadline")
    headline = json.loads(pick_history.ARTIFACT.read_text(encoding="utf-8-sig")).get("headline") or {}
    deadline = headline.get("deadline")
    return headline.get("gameweek"), _parse(deadline) if deadline else None


def last_recorded_for(gameweek: int | None) -> datetime | None:
    # The coming deadline's entry is the unscored one; a scored entry with the
    # same Gameweek number belongs to an earlier season.
    for entry in pick_history.load_history().get("entries") or []:
        if entry.get("gameweek") == gameweek and not entry.get("scored") and entry.get("recordedAt"):
            return _parse(entry["recordedAt"])
    return None


def _parse(stamp: str) -> datetime:
    return datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def commit_data(message: str) -> bool:
    changed = _git("status", "--porcelain", "--", *DATA_FILES).stdout.strip()
    if not changed:
        return False
    _git("add", "--", *DATA_FILES)
    result = _git(
        "commit", "-q", "-m", message, "-m",
        "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>",
        "--", *DATA_FILES,
    )
    if result.returncode != 0:
        print(f"Commit failed: {result.stderr.strip()}")
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()

    branch = _git("branch", "--show-current").stdout.strip()
    if branch != BRANCH:
        print(f"On branch {branch!r}, not {BRANCH!r}; doing nothing.")
        return 1

    now = datetime.now(timezone.utc)
    gameweek, deadline = next_deadline()
    plan = decide(now, deadline, last_recorded_for(gameweek))
    print(f"{now:%Y-%m-%d %H:%M}Z  GW{gameweek}  {plan.reason}")
    if arguments.dry_run:
        return 0

    before = {entry["gameweek"] for entry in pick_history.load_history().get("entries") or [] if entry.get("scored")}
    pick_history.score()
    scored = sorted(
        {entry["gameweek"] for entry in pick_history.load_history().get("entries") or [] if entry.get("scored")}
        - before
    )
    if plan.refresh:
        refresh = subprocess.run(
            [sys.executable, str(ROOT / "analysis" / "calibrate_model.py"), "--refresh-current"],
            cwd=ROOT,
        )
        if refresh.returncode != 0:
            print("Refresh failed; not recording a stale pick.")
        else:
            try:
                pick_history.record()
            except RuntimeError as error:
                print(f"Not recorded: {error}")

    parts = []
    if scored:
        parts.append("Score GW" + ", GW".join(str(gw) for gw in scored))
    if plan.refresh:
        parts.append(f"Record GW{gameweek} pick" + (" (deadline)" if "deadline pick" in plan.reason else ""))
    if parts and commit_data("; ".join(parts)):
        print("Committed: " + "; ".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
