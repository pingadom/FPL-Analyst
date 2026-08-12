"""Fit the frozen listwise horizon/captain challengers to the live pool."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import calibrate_model as lens
from captain_ranker_validation import CAPTAIN_FEATURES, captain_matrix
from frontier_ranker_validation import matrix, selectable_frontier
from listwise_ranker_validation import fitted_ranker


def current_frame(current: list[dict]) -> pd.DataFrame:
    rows = []
    for row in current:
        minutes = row["minutesModel"]
        history = row["history"]
        team = row["teamContext"]
        components = row["components"]
        expected_minutes = max(float(row["expectedMinutes"]), 8)
        rows.append(
            {
                "component_xpts": row["projected"],
                "role_ridge_xpts": row["ensemble"]["roleProjection"],
                "expected_minutes": row["expectedMinutes"],
                "minutes_std": minutes["minutesStd"],
                "play_probability": minutes["playProbability"] / 100,
                "start_probability": minutes["startProbability"] / 100,
                "sixty_probability": minutes["sixtyProbability"] / 100,
                "recent_raw": history["average"],
                "long_raw": history["per90"],
                "goal_rate": components["goals"] / expected_minutes,
                "assist_rate": components["assists"] / expected_minutes,
                "bonus_rate": components["bonus"] / expected_minutes,
                "team_expected_goals_for": team["expectedGoalsFor"],
                "team_expected_goals_against": team["expectedGoalsAgainst"],
                "team_clean_probability": team["cleanSheetProbability"] / 100,
                "price": row["price"] * 10,
                "selected": row["ownership"] * 50_000,
                "fixture_count": 1,
                "haul8_probability": row["distribution"]["haul8Probability"] / 100,
                "return5_probability": row["distribution"]["return5Probability"] / 100,
                "team_attack_rating": team["expectedGoalsFor"],
                "opponent_defence_rating": 1.4 / max(float(row["features"]["fixture"]), 0.15),
                "minutes_security": 0.65 * minutes["sixtyProbability"] / 100 + 0.35 * minutes["playProbability"] / 100,
                "fixture_now": row["features"]["fixture"],
                "position_id": {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}[row["position"]],
            }
        )
    return pd.DataFrame(rows)


def horizon_scores(data: pd.DataFrame, live: pd.DataFrame, current: list[dict]) -> np.ndarray:
    frontier = selectable_frontier(data)
    output = np.zeros(len(live))
    for position in lens.SQUAD_QUOTAS:
        train_mask = (
            (data["position_id"].to_numpy(int) == position)
            & frontier
            & (data["fixture_count"].to_numpy(int) > 0)
        )
        test_mask = live["position_id"].to_numpy(int) == position
        train = data.loc[train_mask].sort_values(["season_order", "GW"], kind="stable")
        train_x, medians = matrix(train)
        test_x, _ = matrix(live.loc[test_mask], medians)
        query = pd.factorize(
            train["season_order"].astype(str) + "-" + train["GW"].astype(str), sort=False
        )[0]
        relevance = np.rint(train["horizon_target"].clip(0, 15)).to_numpy(np.int32)
        model = fitted_ranker(520000 + int(position))
        model.fit(train_x, relevance, qid=query)
        output[test_mask] = model.predict(test_x)
    return output


def captain_scores(data: pd.DataFrame, live: pd.DataFrame) -> np.ndarray:
    structural = data["component_xpts"].to_numpy(float)
    group_rank = pd.Series(structural, index=data.index).groupby(
        [data["season"], data["GW"]]
    ).rank(method="first", ascending=False)
    train = data[
        (group_rank <= 40)
        & (data["play_probability"] >= 0.45)
        & (data["fixture_count"] > 0)
    ].sort_values(["season_order", "GW"], kind="stable")
    train_x, medians = captain_matrix(train)
    test_x, _ = captain_matrix(live, medians)
    query = pd.factorize(
        train["season_order"].astype(str) + "-" + train["GW"].astype(str), sort=False
    )[0]
    relevance = np.rint(train["points"].clip(0, 15)).to_numpy(np.int32)
    model = fitted_ranker(530000)
    model.set_params(eval_metric="ndcg@5", lambdarank_num_pair_per_sample=8)
    model.fit(train_x, relevance, qid=query)
    return model.predict(test_x)


def map_within_position(raw: np.ndarray, reference: np.ndarray, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mapped = np.zeros(len(raw))
    percentile = np.zeros(len(raw))
    for position in lens.SQUAD_QUOTAS:
        indices = np.flatnonzero(positions == position)
        order = indices[np.argsort(raw[indices], kind="stable")]
        mapped[order] = np.sort(reference[indices])
        percentile[order] = np.linspace(0, 100, len(indices))
    return mapped, percentile


def main() -> None:
    data, _ = lens.load_or_build_prepared_history()
    data = data.reset_index(drop=True)
    current = json.loads((lens.ROOT / "app" / "data" / "current-players.json").read_text(encoding="utf-8"))
    live = current_frame(current)
    horizon_raw = horizon_scores(data, live, current)
    captain_raw = captain_scores(data, live)
    positions = live["position_id"].to_numpy(int)
    reference_horizon = np.asarray([float(row["sixWeekProjected"]) for row in current])
    horizon_mapped, horizon_percentile = map_within_position(horizon_raw, reference_horizon, positions)
    captain_percentile = pd.Series(captain_raw).rank(pct=True).to_numpy(float) * 100
    structural_captain_percentile = pd.Series([float(row["projected"]) for row in current]).rank(pct=True).to_numpy(float) * 100
    players = []
    for index, row in enumerate(current):
        players.append(
            {
                "id": int(row["id"]),
                "name": row["name"],
                "position": row["position"],
                "horizonRaw": round(float(horizon_raw[index]), 4),
                "horizonMapped": round(float(horizon_mapped[index]), 2),
                "horizonPercentile": round(float(horizon_percentile[index]), 1),
                "planBlend25": round(0.75 * reference_horizon[index] + 0.25 * horizon_mapped[index], 2),
                "captainRaw": round(float(captain_raw[index]), 4),
                "captainPercentile": round(float(captain_percentile[index]), 1),
                "captainBlend50": round(0.50 * structural_captain_percentile[index] + 0.50 * captain_percentile[index], 1),
            }
        )
    validation_path = lens.ROOT / "analysis" / "data" / "listwise_ranker_validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.exists() else {}
    captain_validation_path = lens.ROOT / "analysis" / "data" / "captain_ranker_validation.json"
    captain_validation = json.loads(captain_validation_path.read_text(encoding="utf-8")) if captain_validation_path.exists() else {}
    result = {
        "schemaVersion": 1,
        "status": "shadow challenger",
        "model": "Causal position LambdaMART horizon rank + captain NDCG rank",
        "promotionRule": "Historical improvement can enter the shadow manager, but only frozen prospective decisions can promote it to production.",
        "historicalValidation": validation.get("models", {}),
        "captainValidation": captain_validation.get("models", {}),
        "players": players,
    }
    output = lens.ROOT / "app" / "data" / "listwise-scores.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output.relative_to(lens.ROOT)} with {len(players)} live scores")


if __name__ == "__main__":
    main()
