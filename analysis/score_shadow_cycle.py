"""Score a locked shadow decision from official FPL event points."""

from __future__ import annotations

import argparse
import json

from prospective_common import APP_DATA, ROOT, SHADOW_ROOT, atomic_json, official_json


EVENT_LIVE = "https://fantasy.premierleague.com/api/event/{gameweek}/live/"


def autosub(manager: dict, live: dict[int, dict]) -> list[int]:
    squad = {int(row["id"]): row for row in manager["squad"]}
    lineup = list(map(int, manager["xiIds"]))
    bench = list(map(int, manager["benchIds"]))
    if manager["chip"] == "Bench Boost":
        return list(squad)
    nonplayers = [player_id for player_id in lineup if live.get(player_id, {}).get("minutes", 0) == 0]
    for outgoing in nonplayers:
        outgoing_position = squad[outgoing]["position"]
        candidates = [
            player_id for player_id in bench
            if live.get(player_id, {}).get("minutes", 0) > 0
            and ((outgoing_position == "GK") == (squad[player_id]["position"] == "GK"))
        ]
        for incoming in candidates:
            proposed = [player_id for player_id in lineup if player_id != outgoing] + [incoming]
            positions = [squad[player_id]["position"] for player_id in proposed]
            if positions.count("GK") == 1 and positions.count("DEF") >= 3 and positions.count("MID") >= 2 and positions.count("FWD") >= 1:
                lineup[lineup.index(outgoing)] = incoming
                bench.remove(incoming)
                break
    return lineup


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gameweek", type=int)
    args = parser.parse_args()
    status = json.loads((APP_DATA / "shadow-status.json").read_text(encoding="utf-8"))
    season = status["season"]
    folder = SHADOW_ROOT / season / f"gw-{args.gameweek:02d}"
    decisions = []
    for path in folder.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("decisionStatus") == "locked":
            decisions.append((path, payload))
    if not decisions:
        raise RuntimeError(f"No locked GW{args.gameweek} shadow decision exists.")
    path, decision = max(decisions, key=lambda item: item[0].stat().st_mtime)
    official = official_json(EVENT_LIVE.format(gameweek=args.gameweek))
    live = {
        int(row["id"]): {
            "points": int(row["stats"]["total_points"]),
            "minutes": int(row["stats"]["minutes"]),
        }
        for row in official["elements"]
    }
    rows = []
    for manager in decision["managers"]:
        lineup = autosub(manager, live)
        captain = int(manager["captainId"])
        vice = int(manager["viceId"])
        if live.get(captain, {}).get("minutes", 0) == 0:
            captain = vice if live.get(vice, {}).get("minutes", 0) > 0 else -1
        multiplier = 3 if manager["chip"] == "Triple Captain" else 2
        points = sum(live.get(player_id, {}).get("points", 0) for player_id in lineup)
        if captain >= 0:
            points += (multiplier - 1) * live.get(captain, {}).get("points", 0)
        hit_cost = sum(int(row.get("cost", 0)) for row in manager["transfers"])
        points -= hit_cost
        rows.append(
            {
                "id": manager["id"],
                "name": manager["name"],
                "points": points,
                "hitCost": hit_cost,
                "captainId": captain if captain >= 0 else None,
                "finalLineupIds": lineup,
                "chip": manager["chip"],
            }
        )
    score_path = SHADOW_ROOT / season / "scores.json"
    scores = json.loads(score_path.read_text(encoding="utf-8")) if score_path.exists() else {"season": season, "gameweeks": []}
    if any(int(row["gameweek"]) == args.gameweek for row in scores["gameweeks"]):
        raise RuntimeError(f"GW{args.gameweek} is already scored; frozen scores are append-only.")
    scores["gameweeks"].append(
        {"gameweek": args.gameweek, "decisionHash": decision["decisionHash"], "decisionPath": str(path.relative_to(ROOT)).replace("\\", "/"), "managers": rows}
    )
    scores["cumulative"] = {
        manager["id"]: sum(
            next(row["points"] for row in week["managers"] if row["id"] == manager["id"])
            for week in scores["gameweeks"]
        )
        for manager in decision["managers"]
    }
    atomic_json(score_path, scores)
    status["completedGameweeks"] = len(scores["gameweeks"])
    status["cumulative"] = scores["cumulative"]
    atomic_json(APP_DATA / "shadow-status.json", status)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
