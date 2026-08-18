"""Hierarchical team clean-sheet challenger for defenders and goalkeepers."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

import calibrate_model as lens
from frontier_ranker_validation import STRATEGY
from multiscale_horizon_validation import add_targets
from probabilistic_minutes_validation import exposure_weights, probability_metrics, season_summary
from wildcard_freehit_ablation import champion_forecasts


CACHE_VERSION = 1
FEATURES = [
    "was_home",
    "GW",
    "team_rest_days",
    "league_goal_rate",
    "team_attack_rating",
    "team_defence_rating",
    "team_form_rating",
    "team_clean_rating",
    "team_rating_confidence",
    "team_regime_shift",
    "opponent_attack_rating",
    "opponent_defence_rating",
    "opponent_form_rating",
    "opponent_clean_rating",
    "opponent_rating_confidence",
    "opponent_regime_shift",
    "team_expected_goals_against",
    "team_expected_goals_for",
    "team_clean_probability",
    "table_points_before",
    "table_goal_difference_before",
    "table_position_before",
]


def team_games(data: pd.DataFrame) -> pd.DataFrame:
    # All player rows on a side share these team/deadline values. Restrict the
    # learner to single fixtures; the structural model remains untouched on
    # doubles where an aggregated binary clean-sheet target is ambiguous.
    columns = ["season", "season_order", "GW", "team_id", "fixture_count", "team_games", "team_clean_sheets", *FEATURES]
    work = data[columns].drop_duplicates(["season", "GW", "team_id"]).copy()
    work = work[work["fixture_count"] == 1].copy()
    work["target"] = (work["team_clean_sheets"] / work["team_games"].clip(lower=1)).clip(0, 1)
    return work


def matrix(frame: pd.DataFrame, medians: pd.Series | None = None) -> tuple[np.ndarray, pd.Series]:
    values = frame[FEATURES].replace([np.inf, -np.inf], np.nan).copy()
    values["GW"] = values["GW"] / 38.0
    values["table_points_before"] = values["table_points_before"] / 90.0
    values["table_goal_difference_before"] = values["table_goal_difference_before"] / 60.0
    values["table_position_before"] = values["table_position_before"] / 20.0
    if medians is None:
        medians = values.median().fillna(0)
    return values.fillna(medians).to_numpy(np.float32), medians


def causal_predictions(data: pd.DataFrame) -> np.ndarray:
    path = lens.CACHE / f"team-clean-challenger-v{CACHE_VERSION}.npz"
    if path.exists():
        cached = np.load(path)
        if len(cached["prediction"]) == len(data):
            return cached["prediction"]
    teams = team_games(data)
    teams["prediction"] = teams["team_clean_probability"]
    seasons = list(dict.fromkeys(data["season"].tolist()))
    for season_order in range(1, len(seasons)):
        train = teams[teams["season_order"] < season_order]
        test_mask = teams["season_order"] == season_order
        test = teams[test_mask]
        train_x, medians = matrix(train)
        test_x, _ = matrix(test, medians)
        age = season_order - train["season_order"].to_numpy(int)
        weight = np.power(0.82, np.maximum(age - 1, 0))
        fitted = XGBRegressor(
            n_estimators=220,
            max_depth=3,
            learning_rate=0.035,
            min_child_weight=15,
            subsample=0.82,
            colsample_bytree=0.82,
            reg_alpha=0.20,
            reg_lambda=4.0,
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            n_jobs=-1,
            random_state=510000 + season_order,
        )
        fitted.fit(train_x, train["target"].to_numpy(float), sample_weight=weight)
        teams.loc[test_mask, "prediction"] = np.clip(fitted.predict(test_x), 0.01, 0.75)
        print(f"Team clean-sheet challenger predicted {seasons[season_order]}", flush=True)
    lookup = {
        (str(row.season), int(row.GW), int(row.team_id)): float(row.prediction)
        for row in teams.itertuples()
    }
    prediction = np.asarray(
        [
            lookup.get((str(season), int(gw), int(team)), np.nan)
            for season, gw, team in data[["season", "GW", "team_id"]].itertuples(index=False, name=None)
        ],
        dtype=float,
    )
    fallback = data["team_clean_probability"].to_numpy(float)
    prediction = np.where(np.isfinite(prediction), prediction, fallback)
    np.savez_compressed(path, prediction=prediction)
    return prediction


def adjust(
    data: pd.DataFrame,
    immediate: np.ndarray,
    plan: np.ndarray,
    candidate: np.ndarray,
    strength: float,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    baseline = data["team_clean_probability"].to_numpy(float)
    delta_probability = candidate - baseline
    if mode == "downside":
        delta_probability = np.minimum(delta_probability, 0)
    clean_points = data["position_id"].map({1: 4.0, 2: 4.0, 3: 1.0, 4: 0.0}).to_numpy(float)
    delta = (
        0.82
        * delta_probability
        * clean_points
        * data["sixty_probability"].to_numpy(float)
        * data["fixture_count"].clip(lower=1).to_numpy(float)
    )
    score = immediate + strength * delta
    # Current clean-sheet evidence is partly persistent but the schedule changes;
    # carry only one fifth of its 4.5-GW value into transfer planning.
    plan_score = plan + strength * 0.9 * delta
    return score, plan_score


def main() -> None:
    original, _ = lens.load_or_build_prepared_history()
    data = add_targets(original.reset_index(drop=True))
    immediate, plan, captain = champion_forecasts(data)
    candidate = causal_predictions(data)
    weights = exposure_weights(data, immediate, plan, captain)
    valid = (
        data["season"].isin(lens.EVALUATION_SEASONS)
        & data["fixture_count"].eq(1)
        & data["position_id"].isin([1, 2])
    ).to_numpy(bool)
    weights *= valid
    actual = (data["team_clean_sheets"] / data["team_games"].clip(lower=1)).to_numpy(float)
    metrics = {
        "baseline": probability_metrics(actual, data["team_clean_probability"].to_numpy(float), weights),
        "challenger": probability_metrics(actual, candidate, weights),
    }
    seasons = list(dict.fromkeys(data["season"].tolist()))
    base_totals, _ = lens.simulate_candidate(data, immediate, STRATEGY, plan_scores=plan, captain_scores=captain)
    base = season_summary(base_totals, seasons)
    variants = []
    for mode, strength in [
        ("symmetric", 0.25), ("symmetric", 0.50), ("symmetric", 0.75), ("symmetric", 1.00),
        ("downside", 0.50), ("downside", 0.75), ("downside", 1.00),
    ]:
        score, plan_score = adjust(data, immediate, plan, candidate, strength, mode)
        totals, _ = lens.simulate_candidate(data, score, STRATEGY, plan_scores=plan_score, captain_scores=captain)
        summary = season_summary(totals, seasons)
        deltas = [row["points"] - old["points"] for row, old in zip(summary["seasons"], base["seasons"])]
        variants.append({
            "name": f"{mode}-{strength:.2f}",
            "mode": mode,
            "strength": strength,
            **summary,
            "averageDelta": round(summary["average"] - base["average"], 1),
            "developmentDelta": round(summary["developmentAverage"] - base["developmentAverage"], 1),
            "holdoutDelta": round(summary["holdoutAverage"] - base["holdoutAverage"], 1),
            "improvedSeasons": int(sum(delta > 0 for delta in deltas)),
            "worseSeasons": int(sum(delta < 0 for delta in deltas)),
            "worstSeasonDelta": int(min(deltas)),
        })
        print("team defence", variants[-1]["name"], variants[-1]["average"], deltas, flush=True)
    eligible = [row for row in variants if row["developmentDelta"] > 0 and row["holdoutDelta"] >= 5 and row["worstSeasonDelta"] >= 0 and row["improvedSeasons"] >= 5]
    selected = max(eligible, key=lambda row: (row["holdoutDelta"], row["developmentDelta"])) if eligible else None
    result = {
        "status": "promoted" if selected else "research-only; robust promotion gate failed",
        "method": "Causal team-game gradient model trained on prior seasons; defender/GK decision-weighted calibration and full recursive validation.",
        "features": FEATURES,
        "decisionWeightedCleanSheetMetrics": metrics,
        "baseline": base,
        "variants": variants,
        "selected": selected,
    }
    output = lens.ROOT / "analysis" / "data" / "team_defence_challenger.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": metrics, "baseline": base, "selected": selected}, indent=2))


if __name__ == "__main__":
    main()
