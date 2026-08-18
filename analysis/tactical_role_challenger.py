"""Causal attacking-role and change-point challenger.

Recent role evidence is built from shifted starts, minutes, xG/xA, threat,
creativity, key passes and open-play crosses.  Set-piece responsibility is only
inferred from the available historical event fields; it is never treated as an
official lineup fact.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

import calibrate_model as lens
from frontier_ranker_validation import STRATEGY
from multiscale_horizon_validation import add_targets
from probabilistic_minutes_validation import exposure_weights, season_summary
from wildcard_freehit_ablation import champion_forecasts


CACHE_VERSION = 1
BASE_FEATURES = [
    "position_id", "GW", "price", "log_selected", "age", "observations",
    "expected_minutes", "start_probability", "sixty_probability", "fixture_now",
    "fixture_count", "team_attack_rating", "opponent_defence_rating",
    "team_expected_goals_for", "goal_rate", "assist_rate", "bonus_rate",
    "recent_raw", "long_raw", "recent_underlying_raw", "long_underlying_raw",
]
SIGNALS = [
    "role_minutes", "role_start", "role_xg", "role_xa", "role_threat",
    "role_creativity", "role_key_pass", "role_big_chance", "role_cross",
    "role_penalty_miss", "role_bps",
]
FEATURES = BASE_FEATURES + [f"{signal}_{window}" for signal in SIGNALS for window in [2, 6]] + [
    f"{signal}_change" for signal in SIGNALS
]


def add_role_features(data: pd.DataFrame) -> pd.DataFrame:
    work = data.copy().sort_values(["player_key", "season_order", "GW"], kind="stable")
    fixture = work["fixture_count"].clip(lower=1)
    minutes = work["minutes"].clip(lower=45)
    observed = work["fixture_count"] > 0
    work["log_selected"] = np.log1p(work["selected"].clip(lower=0))
    raw = {
        "role_minutes": (work["minutes"] / fixture).where(observed),
        "role_start": (work["starts_observed"] / fixture).where(observed),
        "role_xg": (work["expected_goals"] / minutes * 90).where(observed),
        "role_xa": (work["expected_assists"] / minutes * 90).where(observed),
        "role_threat": (work["threat"] / minutes * 90).where(observed),
        "role_creativity": (work["creativity"] / minutes * 90).where(observed),
        "role_key_pass": (work["key_passes"] / minutes * 90).where(observed),
        "role_big_chance": (work["big_chances_created"] / minutes * 90).where(observed),
        "role_cross": (work["open_play_crosses"] / minutes * 90).where(observed),
        "role_penalty_miss": (work["penalties_missed"] / minutes * 90).where(observed),
        "role_bps": (work["bps"] / minutes * 90).where(observed),
    }
    for name, values in raw.items():
        work[f"_{name}"] = values
    grouped = work.groupby("player_key", sort=False)
    for signal in SIGNALS:
        for window in [2, 6]:
            work[f"{signal}_{window}"] = grouped[f"_{signal}"].transform(
                lambda values, n=window: values.rolling(n, min_periods=1).mean().shift(1)
            )
        work[f"{signal}_change"] = work[f"{signal}_2"] - work[f"{signal}_6"]
    work.drop(columns=[f"_{name}" for name in SIGNALS], inplace=True)
    return work.sort_index()


def matrix(frame: pd.DataFrame, medians: pd.Series | None = None) -> tuple[np.ndarray, pd.Series]:
    values = frame[FEATURES].replace([np.inf, -np.inf], np.nan).copy()
    values["GW"] /= 38.0
    values["price"] /= 10.0
    values["age"] /= 35.0
    values["observations"] = np.log1p(values["observations"].clip(lower=0))
    if medians is None:
        medians = values.median().fillna(0)
    return values.fillna(medians).to_numpy(np.float32), medians


def baseline_attack(data: pd.DataFrame) -> np.ndarray:
    group = [data["season"], data["GW"], data["position_id"]]
    goal_vulnerability = (
        data["opponent_goal_vulnerability"]
        / data["opponent_goal_vulnerability"].groupby(group).transform("median").clip(lower=0.01)
    ).clip(0.68, 1.42)
    assist_vulnerability = (
        data["opponent_assist_vulnerability"]
        / data["opponent_assist_vulnerability"].groupby(group).transform("median").clip(lower=0.01)
    ).clip(0.72, 1.35)
    goal_points = data["position_id"].map({1: 6, 2: 6, 3: 5, 4: 4}).to_numpy(float)
    return (
        data["goal_rate"].to_numpy(float) * goal_points * goal_vulnerability.to_numpy(float)
        + data["assist_rate"].to_numpy(float) * 3 * assist_vulnerability.to_numpy(float)
    ) * (data["expected_minutes"].to_numpy(float) / 90) * (0.72 + 0.56 * data["fixture_now"].to_numpy(float)) * data["fixture_count"].clip(lower=1).to_numpy(float)


def causal_prediction(data: pd.DataFrame) -> np.ndarray:
    path = lens.CACHE / f"tactical-role-attack-v{CACHE_VERSION}.npz"
    if path.exists():
        cached = np.load(path)
        if len(cached["prediction"]) == len(data):
            return cached["prediction"]
    work = add_role_features(data)
    baseline = baseline_attack(work)
    goal_points = work["position_id"].map({1: 6, 2: 6, 3: 5, 4: 4}).to_numpy(float)
    target = work["goals"].to_numpy(float) * goal_points + 3 * work["assists"].to_numpy(float)
    prediction = baseline.copy()
    orders = work["season_order"].to_numpy(int)
    observed = work["fixture_count"].to_numpy(int) > 0
    seasons = list(dict.fromkeys(work["season"].tolist()))
    for season_order in range(1, len(seasons)):
        train_mask = (orders < season_order) & observed
        test_mask = orders == season_order
        train = work.loc[train_mask]
        train_x, medians = matrix(train)
        test_x, _ = matrix(work.loc[test_mask], medians)
        age = season_order - train["season_order"].to_numpy(int)
        weight = np.power(0.84, np.maximum(age - 1, 0))
        # A modest emphasis on realised returns prevents a zero-inflated loss
        # from learning that every attacker should be forecast near zero.
        local_target = target[train_mask]
        weight *= np.where(local_target > 0, 1.8, 1.0)
        fitted = XGBRegressor(
            n_estimators=240,
            max_depth=4,
            learning_rate=0.035,
            min_child_weight=18,
            subsample=0.80,
            colsample_bytree=0.78,
            reg_alpha=0.20,
            reg_lambda=4.0,
            objective="reg:pseudohubererror",
            eval_metric="mae",
            tree_method="hist",
            n_jobs=-1,
            random_state=610000 + season_order,
        )
        fitted.fit(train_x, local_target, sample_weight=weight)
        prediction[test_mask] = np.clip(fitted.predict(test_x), 0, 8)
        print(f"Tactical role challenger predicted {seasons[season_order]}", flush=True)
    prediction[data["fixture_count"].to_numpy(int) == 0] = 0
    np.savez_compressed(path, prediction=prediction)
    return prediction


def metrics(actual: np.ndarray, forecast: np.ndarray, weights: np.ndarray) -> dict:
    mask = weights > 0
    error = forecast[mask] - actual[mask]
    return {
        "mae": round(float(np.average(np.abs(error), weights=weights[mask])), 4),
        "bias": round(float(np.average(error, weights=weights[mask])), 4),
        "correlation": round(float(np.corrcoef(forecast[mask], actual[mask])[0, 1]), 4),
    }


def main() -> None:
    original, _ = lens.load_or_build_prepared_history()
    data = add_targets(original.reset_index(drop=True))
    immediate, plan, captain = champion_forecasts(data)
    baseline = baseline_attack(data)
    candidate = causal_prediction(data)
    weights = exposure_weights(data, immediate, plan, captain)
    valid = data["season"].isin(lens.EVALUATION_SEASONS).to_numpy(bool) & data["fixture_count"].gt(0).to_numpy(bool)
    weights *= valid
    goal_points = data["position_id"].map({1: 6, 2: 6, 3: 5, 4: 4}).to_numpy(float)
    actual = data["goals"].to_numpy(float) * goal_points + 3 * data["assists"].to_numpy(float)
    offline = {"baseline": metrics(actual, baseline, weights), "challenger": metrics(actual, candidate, weights)}
    seasons = list(dict.fromkeys(data["season"].tolist()))
    base_totals, _ = lens.simulate_candidate(data, immediate, STRATEGY, plan_scores=plan, captain_scores=captain)
    base_summary = season_summary(base_totals, seasons)
    variants = []
    raw_delta = candidate - baseline
    for mode, strength in [
        ("symmetric", 0.10), ("symmetric", 0.20), ("symmetric", 0.35), ("symmetric", 0.50),
        ("downside", 0.20), ("downside", 0.35), ("downside", 0.50),
    ]:
        delta = np.minimum(raw_delta, 0) if mode == "downside" else raw_delta
        score = immediate + strength * delta
        plan_score = plan + strength * 2.0 * delta
        captain_score = captain + strength * 0.75 * delta
        totals, _ = lens.simulate_candidate(data, score, STRATEGY, plan_scores=plan_score, captain_scores=captain_score)
        summary = season_summary(totals, seasons)
        deltas = [row["points"] - old["points"] for row, old in zip(summary["seasons"], base_summary["seasons"])]
        variants.append({
            "name": f"{mode}-{strength:.2f}", **summary,
            "averageDelta": round(summary["average"] - base_summary["average"], 1),
            "developmentDelta": round(summary["developmentAverage"] - base_summary["developmentAverage"], 1),
            "holdoutDelta": round(summary["holdoutAverage"] - base_summary["holdoutAverage"], 1),
            "improvedSeasons": int(sum(delta > 0 for delta in deltas)),
            "worstSeasonDelta": int(min(deltas)),
        })
        print("tactical role", variants[-1]["name"], variants[-1]["average"], deltas, flush=True)
    eligible = [row for row in variants if row["developmentDelta"] > 0 and row["holdoutDelta"] >= 5 and row["worstSeasonDelta"] >= 0 and row["improvedSeasons"] >= 5]
    selected = max(eligible, key=lambda row: (row["holdoutDelta"], row["developmentDelta"])) if eligible else None
    result = {
        "status": "promoted" if selected else "research-only; robust promotion gate failed",
        "method": "Prior-season causal attacking-route model with shifted short/long tactical-role and inferred set-piece features.",
        "features": FEATURES,
        "decisionWeightedAttackMetrics": offline,
        "baseline": base_summary,
        "variants": variants,
        "selected": selected,
    }
    output = lens.ROOT / "analysis" / "data" / "tactical_role_challenger.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": offline, "baseline": base_summary, "selected": selected}, indent=2))


if __name__ == "__main__":
    main()
