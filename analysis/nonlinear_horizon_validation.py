"""Causal position-specific tree ensemble for the six-GW planning target."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

import calibrate_model as lens
from nonlinear_ensemble_validation import FEATURE_COLUMNS


HORIZON_FEATURE_COLUMNS = FEATURE_COLUMNS + [
    "component_horizon_censored",
    "causal_horizon_ridge",
    "fixture_censored",
    "horizon_weighted_games_censored",
    "prediction_uncertainty",
]
PROGRESS_VERSION = 1


def feature_matrix(
    frame: pd.DataFrame, medians: pd.Series | None = None
) -> tuple[np.ndarray, pd.Series]:
    values = frame[HORIZON_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).copy()
    values["selected"] = np.log1p(values["selected"].clip(lower=0))
    values["price"] = values["price"] / 10.0
    if medians is None:
        medians = values.median().fillna(0)
    return values.fillna(medians).to_numpy(np.float32), medians


def models(seed: int) -> tuple[RandomForestRegressor, XGBRegressor]:
    return (
        RandomForestRegressor(
            n_estimators=220,
            max_depth=16,
            min_samples_split=6,
            min_samples_leaf=4,
            max_features=0.55,
            bootstrap=True,
            n_jobs=-1,
            random_state=seed,
        ),
        XGBRegressor(
            n_estimators=380,
            max_depth=4,
            learning_rate=0.035,
            min_child_weight=10,
            subsample=0.80,
            colsample_bytree=0.75,
            reg_alpha=0.10,
            reg_lambda=2.0,
            objective="reg:squarederror",
            eval_metric="rmse",
            n_jobs=-1,
            random_state=seed,
        ),
    )


def metrics(frame: pd.DataFrame, forecast: np.ndarray) -> dict[str, float]:
    target = frame["horizon_target"].to_numpy(float)
    error = np.abs(forecast - target)
    correlations: list[float] = []
    top_points: list[float] = []
    for _, indices in frame.groupby(["season", "GW"], sort=False).groups.items():
        local = np.asarray(indices, dtype=int)
        correlation = spearmanr(forecast[local], target[local]).statistic
        if np.isfinite(correlation):
            correlations.append(float(correlation))
        best = local[np.argsort(forecast[local])[-15:]]
        top_points.append(float(target[best].mean()))
    return {
        "mae": float(error.mean()),
        "highReturnMae": float(np.abs(forecast[target >= 18] - target[target >= 18]).mean()),
        "gwSpearman": float(np.mean(correlations)),
        "top15Points": float(np.mean(top_points)),
    }


def main() -> None:
    data, _ = lens.load_or_build_prepared_history()
    work = data.reset_index(drop=True)
    seasons = list(dict.fromkeys(work["season"].tolist()))
    structural = work["component_horizon_censored"].to_numpy(float)
    ridge = work["causal_horizon_ridge"].to_numpy(float)
    rf = structural.copy()
    xgb = structural.copy()
    progress_path = lens.CACHE / "nonlinear-horizon-progress-v1.npz"
    completed_through = 0
    if progress_path.exists():
        cache = np.load(progress_path, allow_pickle=False)
        if (
            int(cache["version"][0]) == PROGRESS_VERSION
            and len(cache["random_forest"]) == len(work)
            and np.array_equal(cache["season_order"], work["season_order"].to_numpy(int))
        ):
            rf = cache["random_forest"]
            xgb = cache["xgboost"]
            completed_through = int(cache["completed_through"][0])
            print(f"Resuming after season order {completed_through}")

    season_order_values = work["season_order"].to_numpy(int)
    positions = work["position_id"].to_numpy(int)
    for season_order in range(1, len(seasons)):
        if season_order <= completed_through:
            continue
        season_mask = season_order_values == season_order
        for position in lens.SQUAD_QUOTAS:
            train_mask = (season_order_values < season_order) & (positions == position)
            test_mask = season_mask & (positions == position)
            train = work.loc[train_mask]
            test = work.loc[test_mask]
            train_x, medians = feature_matrix(train)
            test_x, _ = feature_matrix(test, medians)
            target = train["horizon_target"].clip(-3, 55).to_numpy(float)
            age = season_order - train["season_order"].to_numpy(int)
            weight = np.power(0.84, age - 1)
            weight *= np.where(target >= 24, 1.65, np.where(target >= 14, 1.25, 1.0))
            weight /= weight.mean()
            random_forest, xgboost = models(20260850 + season_order * 10 + position)
            random_forest.fit(train_x, target, sample_weight=weight)
            xgboost.fit(train_x, target, sample_weight=weight)
            rf[test_mask] = np.clip(random_forest.predict(test_x), 0, 55)
            xgb[test_mask] = np.clip(xgboost.predict(test_x), 0, 55)
            print(
                f"Predicted horizon {seasons[season_order]} position {position}: "
                f"{len(train):,} train / {len(test):,} test"
            )
        np.savez_compressed(
            progress_path,
            version=np.array([PROGRESS_VERSION]),
            completed_through=np.array([season_order]),
            season_order=season_order_values,
            random_forest=rf,
            xgboost=xgb,
        )

    ensemble = 0.5 * rf + 0.5 * xgb
    prediction_path = lens.CACHE / "nonlinear-horizon-predictions-v1.npz"
    np.savez_compressed(
        prediction_path,
        structural=structural,
        ridge=ridge,
        random_forest=rf,
        xgboost=xgb,
        causal_ensemble=ensemble,
    )
    evaluation = []
    forecasts = {
        "structural": structural,
        "ridge": ridge,
        "randomForest": rf,
        "xgboost": xgb,
        "causalEnsemble": ensemble,
    }
    for season in seasons[2:]:
        mask = work["season"].to_numpy() == season
        evaluation.append(
            {
                "season": season,
                **{
                    name: metrics(work.loc[mask].reset_index(drop=True), values[mask])
                    for name, values in forecasts.items()
                },
            }
        )
    result = {
        "method": "Position-specific RF/XGBoost horizon target, prior seasons only",
        "features": HORIZON_FEATURE_COLUMNS,
        "evaluation": evaluation,
        "averages": {
            name: {
                metric_name: round(
                    float(np.mean([row[name][metric_name] for row in evaluation])), 4
                )
                for metric_name in next(iter(evaluation))[name]
            }
            for name in forecasts
        },
    }
    output = lens.ROOT / "analysis" / "data" / "nonlinear_horizon_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["averages"], indent=2))
    print(f"Saved {output}")
    print(f"Saved {prediction_path}")


if __name__ == "__main__":
    main()
