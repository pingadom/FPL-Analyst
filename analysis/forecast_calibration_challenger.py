"""Causal calibration of forecast magnitudes before combinatorial optimisation."""

from __future__ import annotations

import json

import numpy as np
from sklearn.isotonic import IsotonicRegression

import calibrate_model as lens
from captain_ranker_validation import rank_blend
from frontier_ranker_validation import PLAYER_CANDIDATE, STRATEGY
from listwise_ranker_validation import quantile_map


def causal_isotonic(
    data,
    forecast: np.ndarray,
    target_column: str,
) -> np.ndarray:
    """Fit each season only on completed prior seasons, separately by position."""
    result = forecast.copy()
    orders = data["season_order"].to_numpy(int)
    positions = data["position_id"].to_numpy(int)
    observed = data["fixture_count"].to_numpy(int) > 0
    target = data[target_column].to_numpy(float)
    for season_order in sorted(np.unique(orders)):
        if season_order == 0:
            continue
        for position in lens.SQUAD_QUOTAS:
            train = (orders < season_order) & (positions == position) & observed
            test = (orders == season_order) & (positions == position)
            if train.sum() < 250 or not test.any():
                continue
            fitted = IsotonicRegression(
                increasing=True,
                out_of_bounds="clip",
                y_min=0.0,
            )
            fitted.fit(forecast[train], np.clip(target[train], 0.0, None))
            result[test] = fitted.predict(forecast[test])
    result[~observed] = 0.0
    return result


def calibration_error(data, forecast: np.ndarray, target_column: str) -> float:
    mask = (
        data["season"].isin(lens.EVALUATION_SEASONS).to_numpy(bool)
        & (data["fixture_count"].to_numpy(int) > 0)
    )
    actual = data[target_column].to_numpy(float)
    return float(np.mean(np.abs(forecast[mask] - actual[mask])))


def main() -> None:
    data, _ = lens.load_or_build_prepared_history()
    data = data.reset_index(drop=True)
    immediate, horizon, _ = lens.candidate_forecasts(
        data,
        PLAYER_CANDIDATE,
        robust_planning=False,
        schedule_censored=True,
    )
    stable_plan = 0.75 * immediate * 4.5 + 0.25 * horizon
    frontier_raw = np.load(lens.CACHE / "frontier-causal-predictions-v2.npz")[
        "prediction"
    ]
    horizon_raw = np.load(lens.CACHE / "listwise-horizon_target-v1.npz")[
        "prediction"
    ]
    captain_raw = np.load(lens.CACHE / "captain-listwise-v1.npz")["prediction"]
    score = 0.75 * immediate + 0.25 * quantile_map(data, frontier_raw, immediate)
    plan = 0.75 * stable_plan + 0.25 * quantile_map(data, horizon_raw, stable_plan)
    captain = rank_blend(data, immediate, captain_raw, 0.50)
    calibrated_score = causal_isotonic(data, score, "points")
    calibrated_plan = causal_isotonic(data, plan, "horizon_target")
    variants = (
        ("full-stack", score, plan),
        ("immediate-calibrated", calibrated_score, plan),
        ("horizon-calibrated", score, calibrated_plan),
        ("both-calibrated", calibrated_score, calibrated_plan),
    )
    seasons = list(dict.fromkeys(data["season"].tolist()))
    training_count = len(lens.TRAINING_SEASONS)
    rows = []
    for index, (name, local_score, local_plan) in enumerate(variants, start=1):
        print(f"Running {index}/{len(variants)}: {name}", flush=True)
        totals, stats = lens.simulate_candidate(
            data,
            local_score,
            STRATEGY,
            plan_scores=local_plan,
            captain_scores=captain,
            tracked_player_name="Salah",
        )
        training = totals[:training_count]
        evaluation = totals[training_count:]
        rows.append(
            {
                "name": name,
                "trainingStability": round(
                    float(training.mean() - 0.25 * training.std()), 3
                ),
                "evaluationAverage": round(float(evaluation.mean()), 1),
                "evaluationMinimum": int(round(float(evaluation.min()))),
                "seasonTotals": [round(float(value)) for value in totals],
                "transfers": [row["transfers"] for row in stats[training_count:]],
            }
        )
    selected = max(rows, key=lambda row: row["trainingStability"])
    baseline = rows[0]
    result = {
        "status": "training-selected causal calibration challenger",
        "predictiveCalibration": {
            "immediateMaeBefore": round(calibration_error(data, score, "points"), 4),
            "immediateMaeAfter": round(
                calibration_error(data, calibrated_score, "points"), 4
            ),
            "horizonMaeBefore": round(
                calibration_error(data, plan, "horizon_target"), 4
            ),
            "horizonMaeAfter": round(
                calibration_error(data, calibrated_plan, "horizon_target"), 4
            ),
        },
        "selected": selected,
        "lift": {
            "average": round(
                selected["evaluationAverage"] - baseline["evaluationAverage"], 1
            ),
            "minimum": selected["evaluationMinimum"] - baseline["evaluationMinimum"],
        },
        "experiments": rows,
    }
    output = lens.ROOT / "analysis" / "data" / "forecast_calibration_challenger.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
