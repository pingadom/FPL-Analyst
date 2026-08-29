"""Record the squad the model picks each Gameweek, then score it against reality.

The backtest says what the model would have done over ten finished seasons. It
cannot say whether the model is any good *now*, because every number in it comes
from replaying data the model was built on. The only untainted evidence is a
prediction made before a deadline and scored afterwards, and that evidence has to
be accumulated one week at a time — there is no way to manufacture it in a hurry.

So this keeps an append-only log. Before a deadline it records the picked squad,
captain and projected points. After the Gameweek finishes it fetches official
results and scores what was actually picked. Nothing is ever rewritten once a
Gameweek is locked, which is the point: a record that can be revised after the
fact is not evidence.

Autosubs are deliberately not simulated. FPL substitutes a non-playing starter
with the first bench player who played, in the manager's chosen bench order, and
the artifact does not record that order — so any autosub number here would be a
guess dressed as a result. Bench points are reported separately instead, which
brackets the true score: the realised XI is a floor, and the XI plus bench is a
ceiling that no legal autosub can exceed.

    python analysis/pick_history.py            # record this Gameweek's squad
    python analysis/pick_history.py --score    # score every finished Gameweek
    python analysis/pick_history.py --report   # print the running table
"""

from __future__ import annotations

import argparse
import json
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "app" / "data" / "model-results.json"
HISTORY = ROOT / "app" / "data" / "pick-history.json"
BOOTSTRAP = "https://fantasy.premierleague.com/api/bootstrap-static/"
EVENT_LIVE = "https://fantasy.premierleague.com/api/event/{event}/live/"


def _get(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "FPL-Lens/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def load_history() -> dict:
    if HISTORY.exists():
        return json.loads(HISTORY.read_text(encoding="utf-8-sig"))
    return {"schemaVersion": 1, "entries": []}


def save_history(history: dict) -> None:
    HISTORY.write_text(
        json.dumps(history, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def record() -> dict:
    """Append this deadline's squad. Re-running before kick-off refreshes it."""
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8-sig"))
    headline = artifact.get("headline") or {}
    squad = artifact.get("squad") or []
    if not squad:
        raise RuntimeError("The artifact has no squad to record")

    season = str(headline.get("season") or "")
    gameweek = int(headline.get("gameweek") or 0)
    if not season or not gameweek:
        raise RuntimeError("The artifact does not identify its season and Gameweek")

    entry = {
        "season": season,
        "gameweek": gameweek,
        "recordedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "deadline": headline.get("deadline"),
        "formation": headline.get("formation"),
        "captain": headline.get("captain"),
        "vice": headline.get("vice"),
        "projectedPoints": headline.get("projected"),
        "players": [
            {
                "id": int(player["id"]),
                "name": player.get("name"),
                "position": player.get("position"),
                "price": player.get("price"),
                "starter": bool(player.get("starter")),
                "captain": bool(player.get("captain")),
                "vice": bool(player.get("vice")),
                "projected": player.get("projected"),
            }
            for player in squad
        ],
        "scored": False,
    }

    history = load_history()
    kept = [
        item
        for item in history["entries"]
        if not (item["season"] == season and item["gameweek"] == gameweek)
    ]
    replaced = len(kept) != len(history["entries"])
    # A locked Gameweek is evidence and must not be rewritten; an unlocked one is
    # still a draft and should track the latest recommendation.
    locked = [
        item
        for item in history["entries"]
        if item["season"] == season
        and item["gameweek"] == gameweek
        and item.get("scored")
    ]
    if locked:
        raise RuntimeError(
            f"{season} GW{gameweek} is already scored; refusing to overwrite it"
        )
    kept.append(entry)
    kept.sort(key=lambda item: (item["season"], item["gameweek"]))
    history["entries"] = kept
    save_history(history)
    print(
        f"{'Replaced' if replaced else 'Recorded'} {season} GW{gameweek}: "
        f"{entry['formation']}, captain {entry['captain']}, "
        f"projected {entry['projectedPoints']}"
    )
    return entry


def backfill(commit: str) -> None:
    """Record a squad from a past commit, but only if it predates its deadline.

    Git holds earlier versions of the artifact, and it is tempting to treat them
    as a head start on the record. Most of them are not evidence: a squad
    committed after its deadline was written knowing something about how the
    Gameweek went, even if only which players were injured. The check is
    mechanical rather than a matter of judgement — commit time against the
    deadline in the artifact itself — because the temptation to make an exception
    is exactly what the log exists to resist.
    """
    raw = subprocess.run(
        ["git", "show", f"{commit}:app/data/model-results.json"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.lstrip("﻿")
    if not raw.strip():
        raise RuntimeError(f"No artifact found at commit {commit}")
    artifact = json.loads(raw)
    headline = artifact.get("headline") or {}
    deadline = headline.get("deadline")
    if not deadline:
        raise RuntimeError(f"Artifact at {commit} records no deadline")
    committed_at = subprocess.run(
        ["git", "log", "-1", "--format=%aI", commit],
        capture_output=True,
        text=True,
    ).stdout.strip()
    written = datetime.fromisoformat(committed_at)
    due = datetime.fromisoformat(str(deadline).replace("Z", "+00:00"))
    if written >= due:
        raise RuntimeError(
            f"{commit} was committed {written.isoformat(timespec='minutes')}, after the "
            f"GW{headline.get('gameweek')} deadline {due.isoformat(timespec='minutes')}; "
            "it is not a prospective pick and will not be recorded"
        )
    original = ARTIFACT.read_text(encoding="utf-8-sig")
    try:
        ARTIFACT.write_text(raw, encoding="utf-8")
        entry = record()
    finally:
        ARTIFACT.write_text(original, encoding="utf-8")
    entry_note = f"backfilled from {commit[:8]}, committed {written.isoformat(timespec='minutes')}"
    history = load_history()
    for item in history["entries"]:
        if (
            item["season"] == entry["season"]
            and item["gameweek"] == entry["gameweek"]
        ):
            item["provenance"] = entry_note
    save_history(history)
    print(f"  provenance: {entry_note}")


def score() -> None:
    """Score every recorded Gameweek that has since finished."""
    history = load_history()
    if not history["entries"]:
        print("Nothing recorded yet")
        return
    bootstrap = _get(BOOTSTRAP)
    finished = {
        int(event["id"])
        for event in bootstrap["events"]
        if bool(event.get("finished"))
    }
    live_cache: dict[int, dict[int, dict]] = {}
    updated = 0
    for entry in history["entries"]:
        if entry.get("scored") or entry["gameweek"] not in finished:
            continue
        gameweek = int(entry["gameweek"])
        if gameweek not in live_cache:
            payload = _get(EVENT_LIVE.format(event=gameweek))
            live_cache[gameweek] = {
                int(row["id"]): row for row in payload.get("elements", [])
            }
        live = live_cache[gameweek]

        starters = [p for p in entry["players"] if p["starter"]]
        bench = [p for p in entry["players"] if not p["starter"]]
        captain_id = next(
            (int(p["id"]) for p in entry["players"] if p.get("captain")), None
        )

        def points_for(player: dict) -> int:
            row = live.get(int(player["id"]))
            return int(((row or {}).get("stats") or {}).get("total_points", 0))

        def minutes_for(player: dict) -> int:
            row = live.get(int(player["id"]))
            return int(((row or {}).get("stats") or {}).get("minutes", 0))

        xi_points = sum(points_for(p) for p in starters)
        captain_bonus = 0
        if captain_id is not None:
            captain = next(
                (p for p in entry["players"] if int(p["id"]) == captain_id), None
            )
            if captain is not None and minutes_for(captain) > 0:
                captain_bonus = points_for(captain)
            else:
                vice = next(
                    (p for p in entry["players"] if p.get("vice")), None
                )
                if vice is not None and minutes_for(vice) > 0:
                    captain_bonus = points_for(vice)
        entry["realised"] = {
            "xiPoints": xi_points,
            "captainBonus": captain_bonus,
            "total": xi_points + captain_bonus,
            "benchPoints": sum(points_for(p) for p in bench),
            "startersWhoDidNotPlay": sum(1 for p in starters if minutes_for(p) == 0),
            "note": "autosubs not simulated; bench order is not recorded",
        }
        entry["scored"] = True
        updated += 1
    if updated:
        save_history(history)
    print(f"Scored {updated} newly finished Gameweek(s)")


def report() -> None:
    history = load_history()
    entries = history.get("entries") or []
    if not entries:
        print("Nothing recorded yet")
        return
    print(
        f"{'season':<9}{'gw':>4}{'formation':>11}{'captain':>14}"
        f"{'projected':>11}{'actual':>8}{'error':>8}{'bench':>7}"
    )
    running_projected = running_actual = 0.0
    scored = 0
    for entry in entries:
        realised = entry.get("realised") or {}
        actual = realised.get("total")
        projected = entry.get("projectedPoints")
        line = (
            f"{entry['season']:<9}{entry['gameweek']:>4}"
            f"{str(entry.get('formation')):>11}{str(entry.get('captain'))[:13]:>14}"
            f"{(f'{projected:.1f}' if isinstance(projected,(int,float)) else '-'):>11}"
        )
        if actual is None:
            print(line + f"{'pending':>8}{'':>8}{'':>7}")
            continue
        scored += 1
        running_actual += actual
        if isinstance(projected, (int, float)):
            running_projected += float(projected)
        error = (
            actual - float(projected) if isinstance(projected, (int, float)) else None
        )
        print(
            line
            + f"{actual:>8}"
            + (f"{error:>+8.1f}" if error is not None else f"{'':>8}")
            + f"{realised.get('benchPoints', 0):>7}"
        )
    if scored:
        print(
            f"\n  scored Gameweeks {scored}   total {running_actual:.0f}   "
            f"mean {running_actual / scored:.1f} a week"
        )
        if running_projected:
            print(
                f"  projected {running_projected:.1f}   "
                f"bias {running_actual - running_projected:+.1f} "
                f"({(running_actual - running_projected) / scored:+.1f} a week)"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score", action="store_true", help="Score finished Gameweeks.")
    parser.add_argument("--report", action="store_true", help="Print the running table.")
    parser.add_argument(
        "--backfill",
        metavar="COMMIT",
        help="Record a past artifact, if it was committed before its deadline.",
    )
    arguments = parser.parse_args()
    if arguments.backfill:
        backfill(arguments.backfill)
        report()
    elif arguments.score:
        score()
        report()
    elif arguments.report:
        report()
    else:
        record()
        report()


if __name__ == "__main__":
    main()
