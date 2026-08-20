"""Fit the terminal market/team model to an exact current deadline snapshot."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import calibrate_model as lens
from dynamic_match_model_v2 import geometric_blend, probability_blend
from market_lineup_challenger import (
    implied_total_goals,
    load_market_matches,
    model_matrix,
    normalize_team,
    team_perspective,
)


def official_alias(name: str) -> str:
    aliases = {
        "coventry city": "coventry",
        "hull city": "hull",
        "ipswich town": "ipswich",
        "man city": "man city",
        "man utd": "man united",
        "nott'm forest": "nottm forest",
    }
    normal = normalize_team(name)
    return aliases.get(normal, normal)


def main() -> None:
    market_status_path = lens.ROOT / "app" / "data" / "market-deadline-status-v2.json"
    if not market_status_path.exists():
        raise RuntimeError("Run capture_market_snapshot_v2.py before fitting live market scores")
    market_status = json.loads(market_status_path.read_text(encoding="utf-8"))
    if market_status.get("status") == "unavailable":
        result = {
            "schemaVersion": 1,
            "status": "unavailable; forecast-v2 captain falls back to route consensus",
            "season": market_status["season"],
            "gameweek": market_status["gameweek"],
            "capturedAt": market_status["capturedAt"],
            "deadline": market_status["deadline"],
            "reason": market_status.get("reason", "No market source"),
            "fixtures": [],
        }
        output = lens.ROOT / "app" / "data" / "market-scores-v2.json"
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(result["status"])
        return
    snapshot = json.loads((lens.ROOT / market_status["snapshotPath"]).read_text(encoding="utf-8"))
    deadline_status = json.loads((lens.ROOT / "app" / "data" / "deadline-status.json").read_text(encoding="utf-8"))
    official = json.loads((lens.ROOT / deadline_status["snapshotPath"]).read_text(encoding="utf-8"))["official"]
    code_by_name = {official_alias(team["name"]): team["short_name"] for team in official["teams"]}

    current = pd.DataFrame(snapshot["fixtures"])
    current["homeTeam"] = current["homeTeam"].map(normalize_team)
    current["awayTeam"] = current["awayTeam"].map(normalize_team)
    current["marketTotalGoals"] = implied_total_goals(current["over25Probability"].to_numpy(float))
    current["marketSeasonOrder"] = 999
    current["season"] = snapshot["season"]
    current["homeGoals"] = 0.0
    current["awayGoals"] = 0.0

    matches, _ = load_market_matches()
    training = team_perspective(matches)
    train_x = model_matrix(training)
    age = training["marketSeasonOrder"].max() - training["marketSeasonOrder"].to_numpy(int)
    weights = np.power(0.90, np.maximum(age, 0))
    goal_model = make_pipeline(StandardScaler(), PoissonRegressor(alpha=0.35, max_iter=1_000))
    clean_model = make_pipeline(StandardScaler(), LogisticRegression(C=0.45, max_iter=1_000))
    goal_model.fit(train_x, training["goalsFor"].to_numpy(float), poissonregressor__sample_weight=weights)
    clean_model.fit(
        train_x,
        training["goalsAgainst"].eq(0).astype(int).to_numpy(),
        logisticregression__sample_weight=weights,
    )
    sides = team_perspective(current)
    expected_for = np.clip(goal_model.predict(model_matrix(sides)), 0.20, 4.20)
    reverse = sides.copy()
    reverse[["winProbability", "lossProbability"]] = reverse[["lossProbability", "winProbability"]].to_numpy()
    reverse["wasHome"] = 1.0 - reverse["wasHome"]
    expected_against = np.clip(goal_model.predict(model_matrix(reverse)), 0.20, 4.20)
    logistic_clean = clean_model.predict_proba(model_matrix(sides))[:, 1]
    market_clean = np.clip(0.55 * np.exp(-expected_against) + 0.45 * logistic_clean, 0.02, 0.80)

    players = json.loads((lens.ROOT / "app" / "data" / "current-players.json").read_text(encoding="utf-8"))
    structural = {
        (row["team"], row["opponent"], row["venue"] == "H"): (
            float(row["teamContext"]["expectedGoalsFor"]),
            float(row["teamContext"]["expectedGoalsAgainst"]),
            float(row["teamContext"]["cleanSheetProbability"]) / 100.0,
        )
        for row in players
    }
    weight_artifact = json.loads(
        (lens.ROOT / "analysis" / "data" / "dynamic_match_model_v2.json").read_text(encoding="utf-8")
    )
    latest = weight_artifact["weights"][-1]
    rows = []
    for index, side in enumerate(sides.itertuples()):
        team_code = code_by_name.get(official_alias(side.team))
        opponent_code = code_by_name.get(official_alias(side.opponent))
        if not team_code or not opponent_code:
            continue
        base = structural.get((team_code, opponent_code, bool(side.wasHome)))
        if base is None:
            continue
        dynamic_for = geometric_blend(np.array([base[0]]), np.array([expected_for[index]]), float(latest["attackMarketWeight"]))[0]
        dynamic_against = geometric_blend(np.array([base[1]]), np.array([expected_against[index]]), float(latest["defenceMarketWeight"]))[0]
        dynamic_clean = probability_blend(np.array([base[2]]), np.array([market_clean[index]]), float(latest["cleanMarketWeight"]))[0]
        rows.append(
            {
                "team": team_code,
                "opponent": opponent_code,
                "venue": "H" if bool(side.wasHome) else "A",
                "marketExpectedGoalsFor": round(float(expected_for[index]), 4),
                "marketExpectedGoalsAgainst": round(float(expected_against[index]), 4),
                "marketCleanProbability": round(float(market_clean[index]), 5),
                "dynamicExpectedGoalsFor": round(float(dynamic_for), 4),
                "dynamicExpectedGoalsAgainst": round(float(dynamic_against), 4),
                "dynamicCleanProbability": round(float(dynamic_clean), 5),
            }
        )
    result = {
        "schemaVersion": 1,
        "status": "prospective exact-capture shadow input",
        "season": market_status["season"],
        "gameweek": market_status["gameweek"],
        "capturedAt": market_status["capturedAt"],
        "deadline": market_status["deadline"],
        "sourceSnapshotHash": market_status["snapshotHash"],
        "historicalWeightArtifact": "analysis/data/dynamic_match_model_v2.json",
        "fixtures": rows,
    }
    output = lens.ROOT / "app" / "data" / "market-scores-v2.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output.relative_to(lens.ROOT)} with {len(rows)} team-fixture forecasts")


if __name__ == "__main__":
    main()
