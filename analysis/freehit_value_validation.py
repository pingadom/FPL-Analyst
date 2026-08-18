"""Causal Free Hit value model followed by paired recursive policy tests.

The legacy Free Hit trigger compared two noisy point-forecast lineups.  This
challenger first learns the realised one-week FH marginal gain from prior
seasons only, shrinks the prediction by a prior-residual risk penalty, and then
uses that value inside a complete recursive season rerun.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

import calibrate_model as lens
from frontier_ranker_validation import STRATEGY
from wildcard_freehit_ablation import champion_forecasts
from multiscale_horizon_validation import add_targets


FEATURES = (
    "predictedFreeHitImmediateGain",
    "predictedFreeHitGain",
    "permanentTransferValueForegone",
    "freeHitBlankCount",
    "freeHitDoubleCount",
    "freeHitLineupOverlap",
    "currentLineupValue",
    "freeHitLineupValue",
    "currentLineupExpectedMinutes",
    "freeHitLineupExpectedMinutes",
    "currentLineupUncertainty",
    "freeHitLineupUncertainty",
    "windowProgress",
)
THRESHOLDS = (3.0, 6.0, 9.0)
RISK_PENALTIES = (0.0, 0.35, 0.70)


def opportunity_frame(stats: list[dict], seasons: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for season_order, (season, season_stats) in enumerate(zip(seasons, stats)):
        opportunities = season_stats["chipOpportunities"]
        for index, row in enumerate(opportunities):
            if "actualFreeHitGain" not in row:
                continue
            rows.append(
                {
                    **row,
                    "season": season,
                    "seasonOrder": season_order,
                    "windowProgress": index / max(1, len(opportunities) - 1),
                }
            )
    return pd.DataFrame(rows)


def tree(seed: int) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=120,
        max_depth=2,
        learning_rate=0.035,
        min_child_weight=16,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.35,
        reg_lambda=4.0,
        objective="reg:pseudohubererror",
        eval_metric="mae",
        n_jobs=-1,
        random_state=seed,
    )


def causal_predictions(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    prediction = frame["predictedFreeHitImmediateGain"].to_numpy(float).copy()
    residual_scale = np.full(len(frame), 8.0, dtype=float)
    audit: list[dict] = []
    orders = frame["seasonOrder"].to_numpy(int)
    for season_order in sorted(frame["seasonOrder"].unique()):
        if season_order == 0:
            continue
        train_mask = orders < season_order
        test_mask = orders == season_order
        train = frame.loc[train_mask]
        test = frame.loc[test_mask]
        train_x = train[list(FEATURES)].astype(float).to_numpy()
        test_x = test[list(FEATURES)].astype(float).to_numpy()
        target = train["actualFreeHitGain"].to_numpy(float)
        age = season_order - train["seasonOrder"].to_numpy(int)
        weights = np.power(0.88, np.maximum(age - 1, 0))
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_x)
        test_scaled = scaler.transform(test_x)
        ridge = Ridge(alpha=45.0)
        ridge.fit(train_scaled, target, sample_weight=weights)
        fitted = ridge.predict(train_scaled)
        if season_order >= 2:
            nonlinear = tree(262700 + season_order)
            nonlinear.fit(train_x, target, sample_weight=weights)
            fitted = 0.70 * fitted + 0.30 * nonlinear.predict(train_x)
            test_prediction = 0.70 * ridge.predict(test_scaled) + 0.30 * nonlinear.predict(test_x)
        else:
            test_prediction = ridge.predict(test_scaled)
        prediction[test_mask] = np.clip(test_prediction, -15, 35)
        robust_scale = 1.4826 * float(np.median(np.abs(target - fitted)))
        residual_scale[test_mask] = max(4.0, robust_scale)
        audit.append(
            {
                "season": str(test["season"].iloc[0]),
                "trainingSeasons": int(season_order),
                "trainingRows": int(train_mask.sum()),
                "testRows": int(test_mask.sum()),
                "priorResidualScale": round(float(residual_scale[test_mask][0]), 3),
            }
        )
    return prediction, residual_scale, audit


def main() -> None:
    original, _ = lens.load_or_build_prepared_history()
    data = add_targets(original.reset_index(drop=True))
    scores, plan_scores, captain_scores = champion_forecasts(data)
    seasons = list(dict.fromkeys(data["season"].tolist()))
    training_count = len(lens.TRAINING_SEASONS)
    free_hit_squads = lens.precompute_fresh_squads(data, scores)

    print("Collecting every historical Free Hit opportunity", flush=True)
    collector = lens.ChipPolicy(
        1e6, 1e6, 1e6, 1e6, 0.0, 10, 28, ("Free Hit",)
    )
    _, collector_stats = lens.simulate_candidate(
        data,
        scores,
        STRATEGY,
        chip_policy=collector,
        free_hit_squads=free_hit_squads,
        plan_scores=plan_scores,
        captain_scores=captain_scores,
    )
    frame = opportunity_frame(collector_stats, seasons)
    prediction, residual_scale, fit_audit = causal_predictions(frame)
    actual = frame["actualFreeHitGain"].to_numpy(float)
    orders = frame["seasonOrder"].to_numpy(int)
    evaluation_mask = orders >= training_count

    print("Running paired recursive no-chip control", flush=True)
    baseline, _ = lens.simulate_candidate(
        data,
        scores,
        STRATEGY,
        plan_scores=plan_scores,
        captain_scores=captain_scores,
    )
    rows = []
    for risk_penalty in RISK_PENALTIES:
        adjusted = (
            prediction
            - risk_penalty * residual_scale
            - frame["permanentTransferValueForegone"].to_numpy(float)
        )
        overrides = {
            (str(row.season), int(row.gw), "Free Hit"): float(adjusted[index])
            for index, row in frame.iterrows()
        }
        for threshold in THRESHOLDS:
            print(
                f"Running FH learned-value threshold={threshold:g}, risk={risk_penalty:g}",
                flush=True,
            )
            policy = lens.ChipPolicy(
                1e6,
                threshold,
                1e6,
                1e6,
                0.0,
                10,
                28,
                ("Free Hit",),
            )
            totals, stats = lens.simulate_candidate(
                data,
                scores,
                STRATEGY,
                chip_policy=policy,
                free_hit_squads=free_hit_squads,
                plan_scores=plan_scores,
                captain_scores=captain_scores,
                chip_value_overrides=overrides,
            )
            gain = totals - baseline
            training = gain[:training_count]
            evaluation = gain[training_count:]
            rows.append(
                {
                    "threshold": threshold,
                    "riskPenalty": risk_penalty,
                    "trainingStabilityGain": round(
                        float(training.mean() - 0.25 * training.std()), 3
                    ),
                    "evaluationAverageGain": round(float(evaluation.mean()), 1),
                    "evaluationMinimumGain": round(float(evaluation.min()), 1),
                    "evaluationMaximumGain": round(float(evaluation.max()), 1),
                    "seasonGain": gain.round().astype(int).tolist(),
                    "evaluationChoices": [
                        season["chips"] for season in stats[training_count:]
                    ],
                }
            )
    selected = max(rows, key=lambda row: row["trainingStabilityGain"])
    promoted = bool(
        selected["trainingStabilityGain"] > 0
        and selected["evaluationAverageGain"] > 0
        and selected["evaluationMinimumGain"] >= -12
    )
    result = {
        "status": (
            "promotable causal Free Hit value challenger"
            if promoted
            else "rejected causal Free Hit value challenger"
        ),
        "method": (
            "Prior-season-only Ridge/XGBoost predicts realised one-week FH gain. "
            "A prior-residual penalty creates a conservative value signal; each "
            "threshold is then tested in a complete recursive season simulation."
        ),
        "fitAudit": fit_audit,
        "forecastAudit": {
            "evaluationRows": int(evaluation_mask.sum()),
            "rawCorrelation": round(
                float(
                    np.corrcoef(
                        frame.loc[evaluation_mask, "predictedFreeHitImmediateGain"],
                        actual[evaluation_mask],
                    )[0, 1]
                ),
                3,
            ),
            "learnedCorrelation": round(
                float(np.corrcoef(prediction[evaluation_mask], actual[evaluation_mask])[0, 1]),
                3,
            ),
            "rawMae": round(
                float(
                    np.mean(
                        np.abs(
                            frame.loc[evaluation_mask, "predictedFreeHitImmediateGain"].to_numpy(float)
                            - actual[evaluation_mask]
                        )
                    )
                ),
                2,
            ),
            "learnedMae": round(
                float(np.mean(np.abs(prediction[evaluation_mask] - actual[evaluation_mask]))),
                2,
            ),
        },
        "baselineEvaluationAverage": round(float(baseline[training_count:].mean()), 1),
        "promotionRule": (
            "Training stability > 0, evaluation mean > 0 and evaluation minimum "
            "not below -12; otherwise Free Hit remains manually supervised."
        ),
        "promoted": promoted,
        "selected": selected,
        "experiments": rows,
    }
    output = lens.ROOT / "analysis" / "data" / "freehit_value_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "forecastAudit": result["forecastAudit"],
                "selected": selected,
                "all": [
                    {
                        "threshold": row["threshold"],
                        "risk": row["riskPenalty"],
                        "training": row["trainingStabilityGain"],
                        "evaluation": row["evaluationAverageGain"],
                        "minimum": row["evaluationMinimumGain"],
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
