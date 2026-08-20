"""Create one versioned source of truth for breakthrough model research."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "analysis" / "data"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def main() -> None:
    availability_path = DATA / "availability_leak_audit.json"
    ceilings_path = DATA / "information_ceiling_tournament.json"
    regret_path = DATA / "decision_regret_audit.json"
    premium_path = DATA / "premium_asset_audit.json"
    targets_path = ROOT / "app" / "data" / "model-results.json"
    availability = read_json(availability_path)
    ceilings = read_json(ceilings_path)
    regret = read_json(regret_path)
    premium = read_json(premium_path)
    targets = read_json(targets_path)

    target_by_season = {
        str(row["season"]): float(row["top500Target"])
        for row in targets["backtest"]
    }
    seasons = []
    for row in availability["baseline"]["seasons"]:
        season = str(row["season"])
        target = target_by_season[season]
        seasons.append(
            {
                "season": season,
                "points": int(row["points"]),
                "top500Target": int(target),
                "margin": int(row["points"] - target),
            }
        )

    role_rows = regret["variants"]["lens7_role"]
    weekly_rebuild = round(
        sum(float(row["unlimitedWeeklyRebuild"]) for row in role_rows)
        / len(role_rows),
        1,
    )
    availability_rows = availability["seasons"]
    top_five = {
        str(row["season"]): float(row["topFiveHindsightSquadRate"])
        for row in premium["seasonLeaders"]
    }
    baseline_average = float(availability["baseline"]["average"])
    target_average = round(sum(target_by_season.values()) / len(target_by_season), 1)
    result = {
        "schemaVersion": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": git_value("rev-parse", "HEAD"),
            "branch": git_value("branch", "--show-current"),
            "dirty": bool(git_value("status", "--porcelain")),
        },
        "control": {
            "name": "Leak-free route captain plus audited TC/BB",
            "average": baseline_average,
            "minimum": int(availability["baseline"]["minimum"]),
            "top500Pace": target_average,
            "averageGap": round(baseline_average - target_average, 1),
            "top500Hits": sum(row["margin"] >= 0 for row in seasons),
            "seasons": seasons,
            "firstFourAverageGap": round(
                sum(row["margin"] for row in seasons[:4]) / 4, 1
            ),
            "lastFourAverageGap": round(
                sum(row["margin"] for row in seasons[-4:]) / 4, 1
            ),
        },
        "structuralLoss": {
            "starterNoShows": sum(int(row["starterNoShows"]) for row in availability_rows),
            "autosubsRecovered": sum(int(row["autosubsRecovered"]) for row in availability_rows),
            "unfilledStarterSlots": sum(
                int(row["unfilledStarterSlots"]) for row in availability_rows
            ),
            "twoPointFloorPerSeason": round(
                sum(int(row["twoPointAppearanceFloor"]) for row in availability_rows)
                / len(availability_rows),
                1,
            ),
        },
        "forecastCeilings": {
            "currentWeeklyRebuildAverage": weekly_rebuild,
            "weeklyRebuildGapToPace": round(weekly_rebuild - target_average, 1),
            "informationTournament": ceilings[
                "rankingByRecursiveAuditedChipLift"
            ],
            "warning": "Ceilings overlap and are not additive; the weekly rebuild was refreshed on 2026-08-20 against the frozen current forecast stack.",
        },
        "recentAdaptation": {
            "topFiveSquadRate2023_24": top_five.get("2023/24"),
            "topFiveSquadRate2025_26": top_five.get("2025/26"),
        },
        "promotionGates": {
            "historicalEngineering": {
                "averagePoints": 2300,
                "minimumPositiveSeasons": 6,
                "maximumWorstSeasonDelta": -10,
                "knownBlankXiSlots": 0,
                "illegalStates": 0,
            },
            "governance": "All 2018/19-2025/26 seasons are research-exposed. Historical gates reject weak ideas but cannot prove production performance.",
        },
        "inputHashes": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
            for path in (
                availability_path,
                ceilings_path,
                regret_path,
                premium_path,
                targets_path,
            )
        },
    }
    output = DATA / "breakthrough_benchmark.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
