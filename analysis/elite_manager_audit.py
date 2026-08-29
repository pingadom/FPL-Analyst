"""Compare the current squads of LiveFPL's top all-time managers.

The leaderboard is a selection device, not a forecast feature.  This script keeps
the sample fixed to the first public LiveFPL page (50 managers), then reads their
official FPL picks for one completed/locked Gameweek.  The output is diagnostic:
elite ownership must never be allowed to leak realised points into a deadline
forecast or silently force a player into the model squad.
"""

from __future__ import annotations

import json
import statistics
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "analysis" / "data" / "elite_manager_comparison.json"
PUBLIC_OUTPUT = ROOT / "app" / "data" / "elite-manager-comparison.json"
BOOTSTRAP = "https://fantasy.premierleague.com/api/bootstrap-static/"
PICKS = "https://fantasy.premierleague.com/api/entry/{entry}/event/{event}/picks/"

# LiveFPL all-time leaderboard positions 1-50, captured 29 August 2026.
ELITE_ENTRY_IDS = [
    47394, 53517, 35543, 616, 16499, 18203, 806, 25452, 6098, 881,
    515217, 2086, 82578, 179777, 485455, 1690, 480091, 339434, 9267,
    1826, 262813, 5133, 41, 19797, 112860, 20896, 35602, 3054, 223898,
    195822, 28376, 53543, 123844, 46949, 63377, 243130, 237960, 449246,
    429, 13758, 22493, 4156, 1224, 706408, 13036, 398, 15150, 176400,
    810, 6050,
]


def official_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "FPL-Lens/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def percentile(values: list[float], proportion: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = (len(ordered) - 1) * proportion
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def main(event: int = 2) -> None:
    bootstrap = official_json(BOOTSTRAP)
    elements = {int(row["id"]): row for row in bootstrap["elements"]}
    positional_floor = {
        position: min(
            float(row["now_cost"]) / 10
            for row in bootstrap["elements"]
            if int(row["element_type"]) == position
        )
        for position in (1, 2, 3, 4)
    }

    responses: dict[int, dict] = {}
    errors: list[dict] = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        jobs = {
            executor.submit(official_json, PICKS.format(entry=entry, event=event)): entry
            for entry in ELITE_ENTRY_IDS
        }
        for future in as_completed(jobs):
            entry = jobs[future]
            try:
                responses[entry] = future.result()
            except Exception as error:  # pragma: no cover - network diagnostic
                errors.append({"entry": entry, "error": str(error)})

    managers: list[dict] = []
    ownership: Counter[int] = Counter()
    captaincy: Counter[int] = Counter()
    formations: Counter[str] = Counter()
    chips: Counter[str] = Counter()
    position_squad_spend: dict[int, list[float]] = {position: [] for position in (1, 2, 3, 4)}
    position_xi_spend: dict[int, list[float]] = {position: [] for position in (1, 2, 3, 4)}

    for leaderboard_rank, entry in enumerate(ELITE_ENTRY_IDS, start=1):
        payload = responses.get(entry)
        if not payload:
            continue
        picks = payload.get("picks", [])
        if len(picks) != 15:
            errors.append({"entry": entry, "error": f"Expected 15 picks, received {len(picks)}"})
            continue
        enriched = []
        for pick in picks:
            player = elements[int(pick["element"])]
            row = {
                "id": int(pick["element"]),
                "name": str(player["web_name"]),
                "position": int(pick.get("element_type") or player["element_type"]),
                "slot": int(pick["position"]),
                "price": float(player["now_cost"]) / 10,
                "captain": bool(pick["is_captain"]),
            }
            enriched.append(row)
            ownership[row["id"]] += 1
            if row["captain"]:
                captaincy[row["id"]] += 1

        xi = [row for row in enriched if row["slot"] <= 11]
        bench = [row for row in enriched if row["slot"] > 11]
        formation_counts = Counter(row["position"] for row in xi)
        formation = f"{formation_counts[2]}-{formation_counts[3]}-{formation_counts[4]}"
        formations[formation] += 1
        chip = str(payload.get("active_chip") or "No Chip")
        chips[chip] += 1
        for position in (1, 2, 3, 4):
            position_squad_spend[position].append(
                sum(row["price"] for row in enriched if row["position"] == position)
            )
            position_xi_spend[position].append(
                sum(row["price"] for row in xi if row["position"] == position)
            )
        entry_history = payload.get("entry_history") or {}
        managers.append(
            {
                "leaderboardRank": leaderboard_rank,
                "entry": entry,
                "formation": formation,
                "squadSpend": round(sum(row["price"] for row in enriched), 1),
                "xiSpend": round(sum(row["price"] for row in xi), 1),
                "benchSpend": round(sum(row["price"] for row in bench), 1),
                "benchPremium": round(
                    sum(row["price"] - positional_floor[row["position"]] for row in bench),
                    1,
                ),
                "nearFloorBenchPlayers": sum(
                    row["price"] <= positional_floor[row["position"]] + 0.5 for row in bench
                ),
                "premiumPlayers": sum(row["price"] >= 10 for row in enriched),
                "bank": round(float(entry_history.get("bank", 0)) / 10, 1),
                "teamValue": round(float(entry_history.get("value", 0)) / 10, 1),
                "transfers": int(entry_history.get("event_transfers", 0)),
                "hitCost": int(entry_history.get("event_transfers_cost", 0)),
                "chip": chip,
                "captain": next((row["name"] for row in enriched if row["captain"]), None),
            }
        )

    count = len(managers)
    if count < 40:
        raise RuntimeError(f"Only {count}/50 elite-manager squads were available")

    def describe(key: str) -> dict:
        values = [float(row[key]) for row in managers]
        return {
            "mean": round(statistics.mean(values), 2),
            "median": round(statistics.median(values), 2),
            "p10": round(percentile(values, 0.10), 2),
            "p90": round(percentile(values, 0.90), 2),
        }

    def player_rows(counter: Counter[int], limit: int = 20) -> list[dict]:
        return [
            {
                "id": element,
                "name": str(elements[element]["web_name"]),
                "managers": managers_count,
                "share": round(100 * managers_count / count, 1),
                "price": round(float(elements[element]["now_cost"]) / 10, 1),
                "position": int(elements[element]["element_type"]),
            }
            for element, managers_count in counter.most_common(limit)
        ]

    result = {
        "generatedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "season": "2026/27",
        "gameweek": event,
        "sample": {
            "source": "LiveFPL Elite: first page of the all-time leaderboard",
            "sourceUrl": "https://plan.livefpl.net/elite",
            "officialPicks": "Official FPL public entry/event picks endpoint",
            "requested": len(ELITE_ENTRY_IDS),
            "available": count,
            "errors": errors,
            "selectionCaveat": "Current squads of historically selected managers; diagnostic, not a causal training label.",
        },
        "structure": {
            "squadSpend": describe("squadSpend"),
            "xiSpend": describe("xiSpend"),
            "benchSpend": describe("benchSpend"),
            "benchPremium": describe("benchPremium"),
            "nearFloorBenchPlayers": describe("nearFloorBenchPlayers"),
            "premiumPlayers": describe("premiumPlayers"),
            "bank": describe("bank"),
            "teamValue": describe("teamValue"),
            "formations": [
                {"formation": name, "managers": value, "share": round(100 * value / count, 1)}
                for name, value in formations.most_common()
            ],
            "meanPositionSquadSpend": {
                str(position): round(statistics.mean(values), 2)
                for position, values in position_squad_spend.items()
            },
            "meanPositionXiSpend": {
                str(position): round(statistics.mean(values), 2)
                for position, values in position_xi_spend.items()
            },
        },
        "behaviour": {
            "meanTransfers": round(statistics.mean(row["transfers"] for row in managers), 2),
            "managersTakingHits": sum(row["hitCost"] > 0 for row in managers),
            "meanHitCost": round(statistics.mean(row["hitCost"] for row in managers), 2),
            "chips": [
                {"chip": name, "managers": value, "share": round(100 * value / count, 1)}
                for name, value in chips.most_common()
            ],
        },
        "ownership": player_rows(ownership),
        "captaincy": player_rows(captaincy, 12),
    }
    encoded = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    OUTPUT.write_text(encoded, encoding="utf-8")
    PUBLIC_OUTPUT.write_text(encoded, encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("sample", "structure", "behaviour", "ownership", "captaincy")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
