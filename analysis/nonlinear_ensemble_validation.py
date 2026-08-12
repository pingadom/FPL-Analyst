"""Causal validation of a position-specific RF/XGBoost FPL challenger.

The challenger is intentionally isolated from the production forecast until it
improves recursive squad decisions.  Every season is predicted by models fitted
only to earlier seasons, structural blanks remain zero, and ensemble blend
weights for a season are selected only from already-predicted seasons.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

import calibrate_model as lens


FEATURE_COLUMNS = [
    # Existing causal forecasts are valuable priors, not labels.
    "component_xpts_structural",
    "empirical_xpts",
    "market_role_xpts",
    "role_ridge_xpts",
    # Availability and rotation.
    "expected_minutes",
    "minutes_std",
    "play_probability",
    "start_probability",
    "sixty_probability",
    "minutes_model_confidence",
    "rotation_volatility",
    "competition_pressure",
    "team_rest_days",
    "team_rotation_rate",
    # Player history at multiple horizons.
    "recent_raw",
    "long_raw",
    "recent_underlying_raw",
    "long_underlying_raw",
    "goal_rate",
    "assist_rate",
    "bonus_rate",
    "bps_rate",
    "save_rate",
    "defensive_rate",
    "defensive_return_probability",
    # Team, opponent, and fixture state known at the deadline.
    "fixture_now",
    "was_home",
    "team_attack_rating",
    "team_defence_rating",
    "team_form_rating",
    "team_clean_rating",
    "team_clean_probability",
    "team_expected_goals_for",
    "team_expected_goals_against",
    "opponent_attack_rating",
    "opponent_defence_rating",
    "opponent_form_rating",
    "opponent_clean_rating",
    "team_rating_confidence",
    "opponent_rating_confidence",
    "team_regime_shift",
    "opponent_regime_shift",
    # Market context is deliberately light; popularity is not treated as truth.
    "price",
    "selected",
    "transfer_pressure_raw",
    "age",
]

BLENDS = [
    (1.00, 0.00, 0.00),
    (0.75, 0.125, 0.125),
    (0.50, 0.25, 0.25),
    (0.35, 0.325, 0.325),
    (0.20, 0.40, 0.40),
    (0.00, 0.50, 0.50),
]

PROGRESS_VERSION = 1


def progress_path() -> Path:
    return lens.CACHE / "nonlinear-causal-progress-v1.npz"


@dataclass(frozen=True)
class PredictiveMetrics:
    mae: float
    high_return_mae: float
    gw_spearman: float
    top15_points: float


def matrix(frame: pd.DataFrame, medians: pd.Series | None = None) -> tuple[np.ndarray, pd.Series]:
    """Build a finite matrix using training-only imputation values."""
    values = frame[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).copy()
    values["selected"] = np.log1p(values["selected"].clip(lower=0))
    values["price"] = values["price"] / 10.0
    if medians is None:
        medians = values.median().fillna(0)
    return values.fillna(medians).to_numpy(np.float32), medians


def sample_weights(frame: pd.DataFrame, target: np.ndarray, prediction_season: int) -> np.ndarray:
    """Recency plus modest high-return weighting, normalized to unit mean."""
    age = prediction_season - frame["season_order"].to_numpy(int)
    weight = np.power(0.84, age - 1)
    weight *= np.where(target >= 8, 1.80, np.where(target > 2, 1.30, 1.0))
    return weight / weight.mean()


def fitted_models(seed: int) -> tuple[RandomForestRegressor, XGBRegressor]:
    return (
        RandomForestRegressor(
            n_estimators=240,
            max_depth=16,
            min_samples_split=5,
            min_samples_leaf=3,
            max_features=0.55,
            bootstrap=True,
            n_jobs=-1,
            random_state=seed,
        ),
        XGBRegressor(
            n_estimators=420,
            max_depth=4,
            learning_rate=0.035,
            min_child_weight=8,
            subsample=0.80,
            colsample_bytree=0.75,
            reg_alpha=0.08,
            reg_lambda=1.8,
            objective="reg:squarederror",
            eval_metric="rmse",
            n_jobs=-1,
            random_state=seed,
        ),
    )


def metrics(frame: pd.DataFrame, forecast: np.ndarray) -> PredictiveMetrics:
    actual = frame["points"].to_numpy(float)
    observed = frame["fixture_count"].to_numpy(int) > 0
    error = np.abs(forecast[observed] - actual[observed])
    high = observed & (actual > 2)
    correlations: list[float] = []
    top_points: list[float] = []
    for _, indices in frame.groupby(["season", "GW"], sort=False).groups.items():
        local = np.asarray(indices, dtype=int)
        local = local[observed[local]]
        if len(local) < 20:
            continue
        correlation = spearmanr(forecast[local], actual[local]).statistic
        if np.isfinite(correlation):
            correlations.append(float(correlation))
        best = local[np.argsort(forecast[local])[-15:]]
        top_points.append(float(actual[best].mean()))
    return PredictiveMetrics(
        mae=float(error.mean()),
        high_return_mae=float(np.abs(forecast[high] - actual[high]).mean()),
        gw_spearman=float(np.mean(correlations)),
        top15_points=float(np.mean(top_points)),
    )


def blend_score(value: PredictiveMetrics) -> float:
    """Selection score aligned to both calibration and useful player ordering."""
    return (
        value.mae
        + 0.35 * value.high_return_mae
        - 1.20 * value.gw_spearman
        - 0.08 * value.top15_points
    )


def causal_predictions(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    work = data.reset_index(drop=True)
    structural = work["component_xpts"].to_numpy(float)
    rf_forecast = structural.copy()
    xgb_forecast = structural.copy()
    seasons = list(dict.fromkeys(work["season"].tolist()))
    audit: list[dict] = []
    completed_through = 0
    progress = progress_path()
    if progress.exists():
        cached = np.load(progress, allow_pickle=False)
        if (
            int(cached["version"][0]) == PROGRESS_VERSION
            and len(cached["random_forest"]) == len(work)
            and np.array_equal(cached["season_order"], work["season_order"].to_numpy(int))
        ):
            rf_forecast = cached["random_forest"]
            xgb_forecast = cached["xgboost"]
            completed_through = int(cached["completed_through"][0])
            print(f"Resuming after season order {completed_through}")
    for season_order in range(1, len(seasons)):
        if season_order <= completed_through:
            continue
        season = seasons[season_order]
        season_mask = work["season_order"].to_numpy(int) == season_order
        for position in lens.SQUAD_QUOTAS:
            training_mask = (
                (work["season_order"].to_numpy(int) < season_order)
                & (work["position_id"].to_numpy(int) == position)
                & (work["fixture_count"].to_numpy(int) > 0)
            )
            test_mask = season_mask & (work["position_id"].to_numpy(int) == position)
            train = work.loc[training_mask]
            test = work.loc[test_mask]
            train_x, medians = matrix(train)
            test_x, _ = matrix(test, medians)
            target = train["points"].clip(-2, 20).to_numpy(float)
            weight = sample_weights(train, target, season_order)
            random_forest, xgboost = fitted_models(20260812 + season_order * 10 + position)
            random_forest.fit(train_x, target, sample_weight=weight)
            xgboost.fit(train_x, target, sample_weight=weight)
            rf_forecast[test_mask] = np.clip(random_forest.predict(test_x), 0.0, 14.0)
            xgb_forecast[test_mask] = np.clip(xgboost.predict(test_x), 0.0, 14.0)
            audit.append(
                {
                    "season": season,
                    "position": int(position),
                    "trainingRows": int(len(train)),
                    "testRows": int(len(test)),
                }
            )
            print(
                f"Predicted {season} position {position}: "
                f"{len(train):,} train / {len(test):,} test"
            )
        # A structural blank is known at the deadline and should remain zero.
        blank = season_mask & (work["fixture_count"].to_numpy(int) == 0)
        rf_forecast[blank] = 0.0
        xgb_forecast[blank] = 0.0
        np.savez_compressed(
            progress,
            version=np.array([PROGRESS_VERSION]),
            completed_through=np.array([season_order]),
            season_order=work["season_order"].to_numpy(int),
            random_forest=rf_forecast,
            xgboost=xgb_forecast,
        )
    return structural, rf_forecast, xgb_forecast, audit


def select_causal_blend(
    data: pd.DataFrame,
    structural: np.ndarray,
    rf_forecast: np.ndarray,
    xgb_forecast: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    work = data.reset_index(drop=True)
    seasons = list(dict.fromkeys(work["season"].tolist()))
    result = structural.copy()
    selections: list[dict] = []
    for season_order in range(2, len(seasons)):
        eligible = (work["season_order"].to_numpy(int) >= 1) & (
            work["season_order"].to_numpy(int) < season_order
        )
        candidates = []
        for blend in BLENDS:
            forecast = blend[0] * structural + blend[1] * rf_forecast + blend[2] * xgb_forecast
            value = metrics(work.loc[eligible].reset_index(drop=True), forecast[eligible])
            candidates.append((blend_score(value), blend, value))
        _, selected, selected_metrics = min(candidates, key=lambda item: item[0])
        season_mask = work["season_order"].to_numpy(int) == season_order
        result[season_mask] = (
            selected[0] * structural[season_mask]
            + selected[1] * rf_forecast[season_mask]
            + selected[2] * xgb_forecast[season_mask]
        )
        selections.append(
            {
                "season": seasons[season_order],
                "selected": list(selected),
                "priorMetrics": asdict(selected_metrics),
            }
        )
    return result, selections


def season_metrics(data: pd.DataFrame, forecasts: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for season in dict.fromkeys(data["season"].tolist()):
        if season not in lens.EVALUATION_SEASONS:
            continue
        mask = data["season"].to_numpy() == season
        row: dict[str, object] = {"season": season}
        for name, values in forecasts.items():
            row[name] = asdict(metrics(data.loc[mask].reset_index(drop=True), values[mask]))
        rows.append(row)
    return rows


def main() -> None:
    data, _ = lens.load_or_build_prepared_history()
    structural, random_forest, xgboost, training_audit = causal_predictions(data)
    causal_ensemble, blend_selections = select_causal_blend(
        data, structural, random_forest, xgboost
    )
    forecasts = {
        "structural": structural,
        "randomForest": random_forest,
        "xgboost": xgboost,
        "causalEnsemble": causal_ensemble,
    }
    per_season = season_metrics(data.reset_index(drop=True), forecasts)
    result = {
        "method": (
            "Position-specific RF/XGBoost; each season trained only on prior "
            "seasons; blend selected from prior out-of-sample forecasts"
        ),
        "features": FEATURE_COLUMNS,
        "blendCandidates": BLENDS,
        "blendSelections": blend_selections,
        "trainingAudit": training_audit,
        "predictiveEvaluation": per_season,
        "evaluationAverages": {
            name: {
                metric_name: round(
                    float(np.mean([row[name][metric_name] for row in per_season])), 4
                )
                for metric_name in asdict(PredictiveMetrics(0, 0, 0, 0))
            }
            for name in forecasts
        },
    }
    output = lens.ROOT / "analysis" / "data" / "nonlinear_ensemble_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    prediction_output = lens.CACHE / "nonlinear-causal-predictions-v1.npz"
    np.savez_compressed(
        prediction_output,
        structural=structural,
        random_forest=random_forest,
        xgboost=xgboost,
        causal_ensemble=causal_ensemble,
    )
    print(json.dumps(result["evaluationAverages"], indent=2))
    print(f"Saved {output}")
    print(f"Saved {prediction_output}")


if __name__ == "__main__":
    main()
