"""Causal component-by-component forecast challenger.

The challenger predicts the distinct FPL scoring routes rather than asking one
regression to learn a zero-inflated total.  Every season is predicted from prior
seasons only.  Raw component sums are quantile-mapped onto the frozen champion
scale, so the experiment tests ordering rather than arbitrary score magnitude.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

import calibrate_model as lens
from bench_efficiency_validation import championship_forecasts, variant_summary
from frontier_ranker_validation import STRATEGY, selectable_frontier
from listwise_ranker_validation import quantile_map


CACHE_VERSION = 1
FEATURES = [
    "component_xpts_structural",
    "empirical_xpts",
    "role_ridge_xpts",
    "expected_minutes",
    "minutes_std",
    "play_probability",
    "start_probability",
    "sixty_probability",
    "minutes_model_confidence",
    "rotation_volatility",
    "competition_pressure",
    "recent_raw",
    "long_raw",
    "recent_underlying_raw",
    "long_underlying_raw",
    "goal_rate",
    "assist_rate",
    "clean_sheet_rate",
    "save_rate",
    "bonus_rate",
    "defensive_rate",
    "bps_rate",
    "team_attack_rating",
    "team_defence_rating",
    "opponent_attack_rating",
    "opponent_defence_rating",
    "team_expected_goals_for",
    "team_expected_goals_against",
    "team_clean_probability",
    "team_rating_confidence",
    "team_regime_shift",
    "fixture_now",
    "price",
    "selected",
    "observations",
    "fixture_count",
    "position_id",
    "was_home",
]
COMPONENTS = ("appearance", "attack", "defence", "bonus", "other")


def component_targets(data: pd.DataFrame) -> np.ndarray:
    position = data["position_id"].astype(int)
    goal_points = position.map({1: 6, 2: 6, 3: 5, 4: 4}).to_numpy(float)
    clean_points = position.map({1: 4, 2: 4, 3: 1, 4: 0}).to_numpy(float)
    appearance = (
        data["appearances_observed"].to_numpy(float)
        + data["sixty_observed"].to_numpy(float)
    )
    attack = (
        data["goals"].to_numpy(float) * goal_points
        + 3 * data["assists"].to_numpy(float)
    )
    defence = data["clean_sheets"].to_numpy(float) * clean_points
    defence += np.where(
        position.to_numpy() == 1,
        np.floor(data["saves"].to_numpy(float) / 3),
        0.0,
    )
    defence -= np.where(
        position.isin([1, 2]).to_numpy(),
        np.floor(data["goals_conceded"].to_numpy(float) / 2),
        0.0,
    )
    defence += data["current_rule_dc_points"].to_numpy(float)
    bonus = data["bonus"].to_numpy(float)
    other = data["points"].to_numpy(float) - appearance - attack - defence - bonus
    return np.column_stack([appearance, attack, defence, bonus, other])


def feature_matrix(
    frame: pd.DataFrame,
    medians: pd.Series | None = None,
) -> tuple[np.ndarray, pd.Series]:
    values = frame[FEATURES].replace([np.inf, -np.inf], np.nan).astype(float).copy()
    values["expected_minutes"] /= 90.0
    values["minutes_std"] /= 40.0
    values["recent_raw"] /= 8.0
    values["long_raw"] /= 8.0
    values["recent_underlying_raw"] /= 20.0
    values["long_underlying_raw"] /= 20.0
    values["price"] /= 150.0
    values["selected"] = np.log1p(values["selected"].clip(lower=0)) / 16.0
    values["observations"] = np.log1p(values["observations"].clip(lower=0)) / 5.0
    if medians is None:
        medians = values.median().fillna(0)
    return values.fillna(medians).to_numpy(np.float32), medians


def fitted_model(component: str, seed: int) -> XGBRegressor:
    estimators = {
        "appearance": 120,
        "attack": 190,
        "defence": 170,
        "bonus": 140,
        "other": 110,
    }[component]
    return XGBRegressor(
        n_estimators=estimators,
        max_depth=4,
        learning_rate=0.035,
        min_child_weight=12,
        subsample=0.82,
        colsample_bytree=0.82,
        reg_alpha=0.12,
        reg_lambda=2.6,
        objective="reg:squarederror",
        eval_metric="mae",
        n_jobs=-1,
        random_state=seed,
    )


def causal_component_predictions(data: pd.DataFrame) -> tuple[np.ndarray, dict]:
    cache_path = lens.CACHE / f"component-route-predictions-v{CACHE_VERSION}.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        if len(cached["total"]) == len(data):
            return cached["total"], json.loads(str(cached["audit"].item()))

    targets = component_targets(data)
    baseline = data["component_xpts"].to_numpy(float)
    predictions = np.zeros((len(data), len(COMPONENTS)), dtype=float)
    # Until a prior season exists, preserve the baseline exactly in the total.
    predictions[:, 0] = baseline
    frontier = selectable_frontier(data)
    observed = data["fixture_count"].to_numpy(int) > 0
    orders = data["season_order"].to_numpy(int)
    seasons = list(dict.fromkeys(data["season"].tolist()))
    audit: list[dict] = []
    for season_order in range(1, len(seasons)):
        train_mask = (orders < season_order) & frontier & observed
        test_mask = orders == season_order
        train = data.loc[train_mask]
        test = data.loc[test_mask]
        train_x, medians = feature_matrix(train)
        test_x, _ = feature_matrix(test, medians)
        age = season_order - train["season_order"].to_numpy(int)
        base_weight = np.power(0.84, np.maximum(age - 1, 0))
        for component_index, component in enumerate(COMPONENTS):
            target = targets[train_mask, component_index]
            event_weight = 1 + np.minimum(np.abs(target), 12) * (
                0.08 if component in {"appearance", "other"} else 0.16
            )
            sample_weight = base_weight * event_weight
            sample_weight /= sample_weight.mean()
            fitted = fitted_model(
                component,
                520000 + season_order * 10 + component_index,
            )
            fitted.fit(train_x, target, sample_weight=sample_weight)
            predictions[test_mask, component_index] = fitted.predict(test_x)
        print(f"Component routes predicted {seasons[season_order]}", flush=True)
        audit.append(
            {
                "season": seasons[season_order],
                "trainingRows": int(train_mask.sum()),
                "testRows": int(test_mask.sum()),
            }
        )

    predictions[:, 0] = np.clip(predictions[:, 0], 0, 4)
    predictions[:, 1] = np.clip(predictions[:, 1], 0, 30)
    predictions[:, 2] = np.clip(predictions[:, 2], -4, 15)
    predictions[:, 3] = np.clip(predictions[:, 3], 0, 6)
    predictions[:, 4] = np.clip(predictions[:, 4], -6, 4)
    total = predictions.sum(axis=1)
    total[orders == 0] = baseline[orders == 0]
    total[~observed] = 0
    payload = {
        "components": list(COMPONENTS),
        "features": FEATURES,
        "fits": audit,
    }
    np.savez_compressed(
        cache_path,
        total=total,
        components=predictions,
        audit=json.dumps(payload),
    )
    return total, payload


def main() -> None:
    data, _ = lens.load_or_build_prepared_history()
    data = data.reset_index(drop=True)
    scores, plan, captain = championship_forecasts(data)
    raw, audit = causal_component_predictions(data)
    mapped = quantile_map(data, raw, scores)
    seasons = list(dict.fromkeys(data["season"].tolist()))
    actual = data["points"].to_numpy(float)
    observed = data["fixture_count"].to_numpy(int) > 0
    eval_mask = observed & data["season"].isin(lens.EVALUATION_SEASONS).to_numpy()
    blends = (0.10, 0.25, 0.40)
    rows = []
    for share in blends:
        challenger = (1 - share) * scores + share * mapped
        ratio = np.divide(
            challenger,
            scores,
            out=np.ones_like(challenger),
            where=scores > 0.20,
        )
        challenger_plan = plan * np.clip(ratio, 0.72, 1.28)
        print(f"Recursive component blend {share:.2f}", flush=True)
        totals, stats = lens.simulate_candidate(
            data,
            challenger,
            STRATEGY,
            plan_scores=challenger_plan,
            captain_scores=captain,
            tracked_player_name="Salah",
        )
        training = totals[: len(lens.TRAINING_SEASONS)]
        rows.append(
            {
                "share": share,
                "trainingStability": round(
                    float(training.mean() - 0.25 * training.std()), 3
                ),
                "summary": variant_summary(totals, stats, seasons),
            }
        )
    selected = max(rows, key=lambda row: row["trainingStability"])
    result = {
        "status": "training-selected causal component challenger",
        "method": (
            "Five causal XGBoost models predict appearance, attack, defence, "
            "bonus and residual scoring routes from prior seasons only. Their sum "
            "is quantile-mapped to the frozen champion scale before recursive replay."
        ),
        "rawMetrics": {
            "mae": round(float(np.mean(np.abs(raw[eval_mask] - actual[eval_mask]))), 4),
            "championMae": round(
                float(np.mean(np.abs(scores[eval_mask] - actual[eval_mask]))), 4
            ),
            "correlation": round(
                float(np.corrcoef(raw[eval_mask], actual[eval_mask])[0, 1]), 4
            ),
        },
        "trainingAudit": audit,
        "selected": selected,
        "experiments": rows,
    }
    output = lens.ROOT / "analysis" / "data" / "component_forecast_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "metrics": result["rawMetrics"],
                "selected": {
                    "share": selected["share"],
                    "trainingStability": selected["trainingStability"],
                    "average": selected["summary"]["average"],
                    "minimum": selected["summary"]["minimum"],
                },
                "experiments": [
                    {
                        "share": row["share"],
                        "trainingStability": row["trainingStability"],
                        "average": row["summary"]["average"],
                        "minimum": row["summary"]["minimum"],
                    }
                    for row in rows
                ],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
