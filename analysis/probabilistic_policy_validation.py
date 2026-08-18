"""Causal residual distributions and correlation-aware squad decisions."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

import calibrate_model as lens
from bench_efficiency_validation import championship_forecasts, variant_summary
from frontier_ranker_validation import FEATURES as FRONTIER_FEATURES
from frontier_ranker_validation import STRATEGY, selectable_frontier


CACHE_VERSION = 1
FEATURES = FRONTIER_FEATURES + [
    "ensemble_disagreement",
    "minutes_model_confidence",
    "rotation_volatility",
    "competition_pressure",
    "observations",
    "team_rating_confidence",
    "team_regime_shift",
    "position_id",
    "was_home",
]
CONFIGS = (
    ("central", 0.00, 0.28),
    ("risk03-independent", 0.03, 0.00),
    ("risk03-correlated", 0.03, 0.28),
    ("risk06-independent", 0.06, 0.00),
    ("risk06-correlated", 0.06, 0.28),
    ("risk10-correlated", 0.10, 0.28),
)


def matrix(
    frame: pd.DataFrame,
    scores: np.ndarray,
    medians: pd.Series | None = None,
) -> tuple[np.ndarray, pd.Series]:
    values = frame[FEATURES].replace([np.inf, -np.inf], np.nan).astype(float).copy()
    values.insert(0, "champion_score", scores[frame.index.to_numpy(int)])
    values["price"] /= 150.0
    values["selected"] = np.log1p(values["selected"].clip(lower=0)) / 16.0
    values["observations"] = np.log1p(values["observations"].clip(lower=0)) / 5.0
    values["expected_minutes"] /= 90.0
    values["minutes_std"] /= 40.0
    if medians is None:
        medians = values.median().fillna(0)
    return values.fillna(medians).to_numpy(np.float32), medians


def causal_quantiles(data: pd.DataFrame, scores: np.ndarray) -> tuple[np.ndarray, dict]:
    cache_path = lens.CACHE / f"champion-residual-quantiles-v{CACHE_VERSION}.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        if len(cached["quantiles"]) == len(data):
            return cached["quantiles"], json.loads(str(cached["audit"].item()))
    residual = data["points"].to_numpy(float) - scores
    orders = data["season_order"].to_numpy(int)
    observed = data["fixture_count"].to_numpy(int) > 0
    frontier = selectable_frontier(data)
    seasons = list(dict.fromkeys(data["season"].tolist()))
    quantiles = np.column_stack(
        [
            -0.80 * data["prediction_uncertainty"].to_numpy(float),
            0.80 * data["prediction_uncertainty"].to_numpy(float),
        ]
    )
    audit = []
    for season_order in range(1, len(seasons)):
        train_mask = (orders < season_order) & observed & frontier
        test_mask = orders == season_order
        train = data.loc[train_mask]
        test = data.loc[test_mask]
        train_x, medians = matrix(train, scores)
        test_x, _ = matrix(test, scores, medians)
        age = season_order - train["season_order"].to_numpy(int)
        sample_weight = np.power(0.84, np.maximum(age - 1, 0))
        sample_weight /= sample_weight.mean()
        model = XGBRegressor(
            n_estimators=260,
            max_depth=4,
            learning_rate=0.035,
            min_child_weight=14,
            subsample=0.84,
            colsample_bytree=0.84,
            reg_alpha=0.12,
            reg_lambda=2.8,
            objective="reg:quantileerror",
            quantile_alpha=np.array([0.10, 0.90]),
            tree_method="hist",
            n_jobs=-1,
            random_state=620000 + season_order,
        )
        model.fit(
            train_x,
            residual[train_mask],
            sample_weight=sample_weight,
        )
        quantiles[test_mask] = model.predict(test_x)
        audit.append(
            {
                "season": seasons[season_order],
                "trainingRows": int(train_mask.sum()),
                "testRows": int(test_mask.sum()),
            }
        )
        print(f"Residual quantiles predicted {seasons[season_order]}", flush=True)
    lower = np.minimum(quantiles[:, 0], quantiles[:, 1])
    upper = np.maximum(quantiles[:, 0], quantiles[:, 1])
    quantiles = np.column_stack([lower, upper])
    payload = {"features": FEATURES, "fits": audit, "quantiles": [0.10, 0.90]}
    np.savez_compressed(cache_path, quantiles=quantiles, audit=json.dumps(payload))
    return quantiles, payload


def calibration(data, scores: np.ndarray, quantiles: np.ndarray) -> dict:
    evaluation = data["season"].isin(lens.EVALUATION_SEASONS).to_numpy()
    observed = data["fixture_count"].to_numpy(int) > 0
    mask = evaluation & observed
    residual = data["points"].to_numpy(float) - scores
    lower = quantiles[:, 0]
    upper = quantiles[:, 1]
    sigma = np.maximum(0.35, (upper - lower) / 2.563)
    bins = pd.qcut(sigma[mask], 5, labels=False, duplicates="drop")
    rows = []
    masked_residual = residual[mask]
    masked_sigma = sigma[mask]
    for bin_id in sorted(np.unique(bins)):
        local = np.asarray(bins == bin_id)
        rows.append(
            {
                "quintile": int(bin_id) + 1,
                "predictedSigma": round(float(masked_sigma[local].mean()), 3),
                "residualStd": round(float(masked_residual[local].std()), 3),
                "rows": int(local.sum()),
            }
        )
    return {
        "lowerCoverage": round(float(np.mean(residual[mask] >= lower[mask])), 4),
        "upperCoverage": round(float(np.mean(residual[mask] <= upper[mask])), 4),
        "intervalCoverage": round(
            float(np.mean((residual[mask] >= lower[mask]) & (residual[mask] <= upper[mask]))),
            4,
        ),
        "sigmaBins": rows,
    }


def main() -> None:
    data, _ = lens.load_or_build_prepared_history()
    data = data.reset_index(drop=True)
    scores, plan, captain = championship_forecasts(data)
    quantiles, audit = causal_quantiles(data, scores)
    sigma = np.maximum(0.35, (quantiles[:, 1] - quantiles[:, 0]) / 2.563)
    horizon_games = data["horizon_weighted_games_censored"].to_numpy(float).clip(1, 6)
    horizon_sigma = sigma * np.sqrt(horizon_games)
    seasons = list(dict.fromkeys(data["season"].tolist()))
    rows = []
    raw_results = {}
    for name, risk_aversion, correlation in CONFIGS:
        strategy = replace(
            STRATEGY,
            name=name,
            squad_risk_aversion=risk_aversion,
            defence_residual_correlation=correlation,
        )
        print(f"Running {name}", flush=True)
        totals, stats = lens.simulate_candidate(
            data,
            scores,
            strategy,
            plan_scores=plan,
            captain_scores=captain,
            tracked_player_name="Salah",
            risk_scores=horizon_sigma,
        )
        training = totals[: len(lens.TRAINING_SEASONS)]
        row = {
            "name": name,
            "riskAversion": risk_aversion,
            "defenceCorrelation": correlation,
            "trainingStability": round(
                float(training.mean() - 0.25 * training.std()), 3
            ),
            "summary": variant_summary(totals, stats, seasons),
        }
        rows.append(row)
        raw_results[name] = (totals, stats, strategy)
    selected = max(rows, key=lambda row: row["trainingStability"])
    selected_totals, _, selected_strategy = raw_results[selected["name"]]
    print(f"Running selected with audited chips: {selected['name']}", flush=True)
    chip_totals, chip_stats = lens.simulate_candidate(
        data,
        scores,
        selected_strategy,
        chip_policy=lens.AUDITED_CHAMPION_CHIP_POLICY,
        plan_scores=plan,
        captain_scores=captain,
        tracked_player_name="Salah",
        risk_scores=horizon_sigma,
    )
    result = {
        "status": "training-selected probabilistic policy challenger",
        "method": (
            "Prior-season 10th/90th residual quantiles create player-specific "
            "horizon risk. Legal squad utility uses portfolio standard deviation "
            "with the empirically measured 0.28 same-team defensive correlation."
        ),
        "distributionAudit": audit,
        "calibration": calibration(data, scores, quantiles),
        "selected": selected,
        "selectedWithAuditedChips": variant_summary(chip_totals, chip_stats, seasons),
        "experiments": rows,
    }
    output = lens.ROOT / "analysis" / "data" / "probabilistic_policy_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "calibration": result["calibration"],
                "selected": selected["name"],
                "selectedAverage": selected["summary"]["average"],
                "selectedWithChips": result["selectedWithAuditedChips"]["average"],
                "experiments": [
                    {
                        "name": row["name"],
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
