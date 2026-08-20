"""Attach retrospective absence reasons to held-player droughts.

This script deliberately does not create model features. The source is scraped
after matches and is therefore valid for explaining a no-show, not predicting
one in a causal backtest. Live decisions use official FPL deadline snapshots.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import calibrate_model as lens


SOURCE_ROOT = lens.ROOT / "work" / "availability-data" / "raw" / "GB1"
HELD_AUDIT = lens.ROOT / "analysis" / "data" / "held_player_audit.json"
OUTPUT = lens.ROOT / "analysis" / "data" / "retrospective_availability_audit.json"
UNAVAILABLE = {
    "injured",
    "suspended",
    "national_team",
    "not_at_club",
    "not_included",
}
TEAM_ALIASES = {
    "mancity": "manchestercity",
    "manutd": "manchesterunited",
    "spurs": "tottenhamhotspur",
    "nottmforest": "nottinghamforest",
    "newcastle": "newcastleunited",
    "brighton": "brightonandhovealbion",
    "westham": "westhamunited",
    "wolves": "wolverhamptonwanderers",
}


def normalise(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", text.lower())


def team_key(value: str) -> str:
    key = normalise(value)
    return TEAM_ALIASES.get(key, key)


def name_similarity(short_name: str, full_name: str) -> float:
    short = normalise(short_name)
    full = normalise(full_name)
    if not short or not full:
        return 0.0
    if short == full:
        return 1.0
    if short in full:
        return 0.94
    # FPL web names such as B.Fernandes retain only an initial and surname.
    if len(short) >= 4 and full.endswith(short[1:]) and short[0] == full[0]:
        return 0.92
    return SequenceMatcher(None, short, full).ratio()


def load_season_players(start_year: int) -> list[dict]:
    players: list[dict] = []
    for path in sorted((SOURCE_ROOT / str(start_year)).glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        club = str(payload.get("club", path.stem))
        for competition in payload.get("competitions", []):
            if competition.get("code") != "GB1":
                continue
            for player in competition.get("players", []):
                players.append(
                    {
                        "name": str(player.get("name", "")),
                        "club": club,
                        "matches": player.get("matches", []),
                    }
                )
    return players


def best_player_match(drought: dict, players: list[dict]) -> tuple[dict | None, float]:
    target_team = team_key(drought["team"])
    ranked: list[tuple[float, dict]] = []
    for player in players:
        club = team_key(player["club"])
        team_match = target_team in club or club in target_team
        name_score = name_similarity(drought["player"], player["name"])
        score = name_score + (0.10 if team_match else 0.0)
        ranked.append((score, player))
    if not ranked:
        return None, 0.0
    score, player = max(ranked, key=lambda item: item[0])
    return (player, min(1.0, score)) if score >= 0.68 else (None, score)


def main() -> None:
    held = json.loads(HELD_AUDIT.read_text(encoding="utf-8"))
    season_cache: dict[int, list[dict]] = {}
    rows: list[dict] = []
    status_counts: Counter[str] = Counter()
    explained = 0
    for drought in held["droughts"]:
        start_year = int(str(drought["season"])[:4])
        players = season_cache.setdefault(start_year, load_season_players(start_year))
        match, confidence = best_player_match(drought, players)
        statuses: list[dict] = []
        if match:
            for event in match["matches"]:
                try:
                    round_number = int(str(event.get("round", "")).split(".")[0])
                except ValueError:
                    continue
                if drought["startGw"] <= round_number <= drought["endGw"]:
                    status = str(event.get("status", "unknown"))
                    statuses.append(
                        {
                            "round": round_number,
                            "status": status,
                            "detail": str(event.get("detail", "")),
                        }
                    )
                    status_counts[status] += 1
        absence_rows = [row for row in statuses if row["status"] in UNAVAILABLE]
        if absence_rows:
            explained += 1
        rows.append(
            {
                "season": drought["season"],
                "player": drought["player"],
                "team": drought["team"],
                "startGw": drought["startGw"],
                "endGw": drought["endGw"],
                "matchedPlayer": match["name"] if match else None,
                "matchConfidence": round(confidence, 3),
                "retrospectiveAvailabilityCause": bool(absence_rows),
                "statuses": statuses,
            }
        )
    result = {
        "status": "retrospective explanation only—not a forecast input",
        "source": "withqwerty/availability-data (Transfermarkt-derived)",
        "sourceUrl": "https://github.com/withqwerty/availability-data",
        "methodWarning": "Matchday statuses were scraped after the event. They explain absences but must never enter a deadline-causal historical replay.",
        "droughtsWithAvailabilityCause": explained,
        "totalDroughts": len(rows),
        "statusCounts": dict(status_counts),
        "rows": rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
