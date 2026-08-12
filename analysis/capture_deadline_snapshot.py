"""Capture an immutable official deadline snapshot and minutes intelligence."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from prospective_common import (
    APP_DATA,
    BOOTSTRAP_URL,
    FIXTURES_URL,
    ROOT,
    SNAPSHOT_ROOT,
    atomic_json,
    next_event,
    official_json,
    payload_hash,
    season_label,
    utc_now,
)


OVERRIDE_PATH = ROOT / "analysis" / "inputs" / "deadline_overrides.json"


def load_overrides(captured: datetime) -> dict[int, dict]:
    if not OVERRIDE_PATH.exists():
        return {}
    payload = json.loads(OVERRIDE_PATH.read_text(encoding="utf-8"))
    rows = payload.get("players", [])
    result: dict[int, dict] = {}
    for row in rows:
        player_id = int(row["id"])
        minutes = float(row["expectedMinutes"])
        start = float(row["startProbability"])
        confidence = float(row.get("confidence", 0.75))
        if not 0 <= minutes <= 90 or not 0 <= start <= 1 or not 0 <= confidence <= 1:
            raise ValueError(f"Invalid deadline override for player {player_id}")
        updated = datetime.fromisoformat(str(row["updatedAt"]).replace("Z", "+00:00"))
        if updated.tzinfo is None:
            raise ValueError(f"Override timestamp must include a timezone for player {player_id}")
        if updated.astimezone(timezone.utc) > captured:
            raise ValueError(f"Override timestamp is in the future for player {player_id}")
        result[player_id] = {**row, "updatedAt": updated.astimezone(timezone.utc).isoformat()}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", action="store_true", help="Mark the snapshot as the final pre-deadline lock.")
    args = parser.parse_args()
    captured = utc_now()
    bootstrap = official_json(BOOTSTRAP_URL)
    fixtures = official_json(FIXTURES_URL)
    event = next_event(bootstrap)
    season = season_label(bootstrap)
    gw = int(event["id"])
    teams = {int(team["id"]): team for team in bootstrap["teams"]}
    deadline = datetime.fromisoformat(str(event["deadline_time"]).replace("Z", "+00:00"))
    if args.lock and captured >= deadline:
        raise RuntimeError("A deadline snapshot must be locked before the official deadline.")
    overrides = load_overrides(captured)
    model_players = {
        int(row["id"]): row
        for row in json.loads((APP_DATA / "current-players.json").read_text(encoding="utf-8"))
    }
    intelligence = []
    raw_players = []
    for row in bootstrap["elements"]:
        player_id = int(row["id"])
        team = teams[int(row["team"])]
        raw_players.append(
            {
                key: row.get(key)
                for key in (
                    "id", "code", "first_name", "second_name", "web_name", "team",
                    "element_type", "now_cost", "selected_by_percent", "status",
                    "chance_of_playing_next_round", "news", "news_added", "ep_next",
                    "transfers_in_event", "transfers_out_event", "penalties_order",
                    "corners_and_indirect_freekicks_order", "direct_freekicks_order",
                )
            }
        )
        model = model_players.get(player_id, {})
        model_minutes = float(model.get("expectedMinutes", 0))
        model_start = float(model.get("minutesModel", {}).get("startProbability", 0)) / 100
        official_chance = row.get("chance_of_playing_next_round")
        official_chance = 100 if official_chance is None else float(official_chance)
        chance_factor = max(0.0, min(1.0, official_chance / 100))
        official_minutes = model_minutes * chance_factor
        override = overrides.get(player_id)
        if override:
            confidence = float(override.get("confidence", 0.75))
            expected_minutes = confidence * float(override["expectedMinutes"]) + (1 - confidence) * official_minutes
            start_probability = confidence * float(override["startProbability"]) + (1 - confidence) * model_start * chance_factor
            source = str(override.get("source", "documented override"))
        else:
            expected_minutes = official_minutes
            start_probability = model_start * chance_factor
            source = "Lens minutes + official availability"
        disagreement = abs(expected_minutes - model_minutes)
        intelligence.append(
            {
                "id": player_id,
                "name": row["web_name"],
                "team": team["short_name"],
                "status": row["status"],
                "officialChance": official_chance,
                "officialNews": row.get("news") or "No official flag",
                "newsAdded": row.get("news_added"),
                "modelExpectedMinutes": round(model_minutes, 1),
                "expectedMinutes": round(expected_minutes, 1),
                "startProbability": round(start_probability, 3),
                "minutesDisagreement": round(disagreement, 1),
                "lateNewsPriority": bool(
                    row["status"] != "a"
                    or disagreement >= 12
                    or override is not None
                    or bool(row.get("news"))
                ),
                "source": source,
                "override": override,
            }
        )
    relevant_fixtures = [
        {
            key: fixture.get(key)
            for key in (
                "id", "event", "kickoff_time", "team_h", "team_a", "team_h_difficulty",
                "team_a_difficulty", "started", "finished", "provisional_start_time",
            )
        }
        for fixture in fixtures
    ]
    schedule_fingerprint = payload_hash(relevant_fixtures)
    previous_status_path = APP_DATA / "deadline-status.json"
    previous_status = (
        json.loads(previous_status_path.read_text(encoding="utf-8"))
        if previous_status_path.exists()
        else {}
    )
    previous_fingerprint = previous_status.get("scheduleFingerprint")
    schedule_changed = previous_fingerprint is not None and previous_fingerprint != schedule_fingerprint
    core = {
        "schemaVersion": 1,
        "season": season,
        "gameweek": gw,
        "deadline": event["deadline_time"],
        "capturedAt": captured.isoformat(),
        "status": "locked" if args.lock else "provisional",
        "official": {
            "event": event,
            "gameSettings": bootstrap.get("game_settings", {}),
            "teams": list(teams.values()),
            "players": raw_players,
            "fixtures": relevant_fixtures,
        },
        "deadlineIntelligence": intelligence,
        "overrideFile": str(OVERRIDE_PATH.relative_to(ROOT)),
        "overrideCount": len(overrides),
        "modelCoverage": {
            "officialPlayers": len(intelligence),
            "playersWithFullLensProjection": len(model_players),
            "unmodelledOfficialPlayers": sum(int(row["id"]) not in model_players for row in bootstrap["elements"]),
        },
        "scheduleEvidence": {
            "fingerprint": schedule_fingerprint,
            "previousFingerprint": previous_fingerprint,
            "changedSincePreviousCapture": schedule_changed,
        },
    }
    snapshot_hash = payload_hash(core)
    core["snapshotHash"] = snapshot_hash
    timestamp = captured.strftime("%Y%m%dT%H%M%SZ")
    target = SNAPSHOT_ROOT / season / f"gw-{gw:02d}" / f"{timestamp}-{snapshot_hash[:12]}.json"
    if target.exists():
        raise FileExistsError(f"Immutable snapshot already exists: {target}")
    atomic_json(target, core)
    latest = {
        "schemaVersion": 1,
        "season": season,
        "gameweek": gw,
        "deadline": event["deadline_time"],
        "capturedAt": captured.isoformat(),
        "status": core["status"],
        "snapshotHash": snapshot_hash,
        "snapshotPath": str(target.relative_to(ROOT)).replace("\\", "/"),
        "overrideCount": len(overrides),
        "playersTracked": len(intelligence),
        "playersModelled": len(model_players),
        "unmodelledPlayers": sum(int(row["id"]) not in model_players for row in bootstrap["elements"]),
        "lateNewsCount": sum(row["lateNewsPriority"] for row in intelligence),
        "lateNewsPlayers": sorted(
            [row for row in intelligence if row["lateNewsPriority"]],
            key=lambda row: (row["status"] == "a", -row["minutesDisagreement"]),
        )[:12],
        "scheduleFingerprint": schedule_fingerprint,
        "scheduleChanged": schedule_changed,
        "method": "Immutable official bootstrap/fixture capture plus timestamped expected-minutes overrides. Provisional snapshots can be revised; only --lock is treated as the final pre-deadline decision input.",
    }
    atomic_json(APP_DATA / "deadline-status.json", latest)
    print(f"Captured {core['status']} {season} GW{gw}: {target.relative_to(ROOT)}")
    print(f"Snapshot hash: {snapshot_hash}")


if __name__ == "__main__":
    main()
