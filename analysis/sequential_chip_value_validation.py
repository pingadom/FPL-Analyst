"""Causal optimal-stopping policy for Triple Captain and Bench Boost.

The existing chip engine had two limitations: it required a confirmed double,
and its advertised continuation value was hard-coded to zero.  This challenger
allows strong single-Gameweek opportunities (important when chips expire at
GW19), predicts realised marginal chip value, and compares use-now with an
empirical reservation value learned only from earlier seasons.

TC and BB do not alter squad or transfer state, so their realised marginal gains
can be evaluated exactly on the same recursive no-chip path.  Wildcard and Free
Hit are excluded here because they require paired continuation rollouts.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

import calibrate_model as lens
from bench_efficiency_validation import championship_forecasts
from decision_focused_horizon_validation import causal_online_prediction
from frontier_ranker_validation import STRATEGY
from listwise_ranker_validation import quantile_map
from multiscale_horizon_validation import (
    adaptive_value,
    add_targets,
    causal_online_ridge_horizons,
    causal_ridge_horizons,
    structural_horizons,
)
from multiscale_phase_validation import event_number


CHIPS = ("Triple Captain", "Bench Boost")
RESERVATION_QUANTILES = (0.50, 0.65, 0.75, 0.85)
RISK_PENALTIES = (0.0, 0.15, 0.30)
TC_FEATURES = (
    "predictedTripleCaptainGain",
    "captainFixtureCount",
    "captainPlayProbability",
    "captainStartProbability",
    "captainSixtyProbability",
    "captainExpectedMinutes",
    "captainUncertainty",
    "captainReturnProbability",
    "captainHaulProbability",
    "captainPrice",
    "windowProgress",
)
BB_FEATURES = (
    "predictedBenchBoostGain",
    "benchFixtureCount",
    "benchDoubleCount",
    "benchBlankCount",
    "benchPlayProbability",
    "benchSixtyProbability",
    "benchExpectedMinutes",
    "benchUncertainty",
    "benchMinimumPlayProbability",
    "windowProgress",
)


@dataclass(frozen=True)
class Policy:
    quantile: float
    risk_penalty: float


def legal_windows(season: str, weeks: list[int]) -> list[tuple[int, int]]:
    if season == "2025-26":
        return [(1, 19), (20, 38)]
    return [(int(weeks[0]), int(weeks[-1]))]


def opportunity_frame(stats: list[dict], seasons: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for season_order, (season, season_stats) in enumerate(zip(seasons, stats)):
        opportunities = season_stats["chipOpportunities"]
        weeks = [int(row["gw"]) for row in opportunities]
        event_position = {gw: index for index, gw in enumerate(weeks)}
        for row in opportunities:
            gw = int(row["gw"])
            window = next(
                (pair for pair in legal_windows(season, weeks) if pair[0] <= gw <= pair[1]),
                (weeks[0], weeks[-1]),
            )
            local_weeks = [week for week in weeks if window[0] <= week <= window[1]]
            local_index = local_weeks.index(gw)
            rows.append(
                {
                    **row,
                    "season": season,
                    "seasonOrder": season_order,
                    "eventNumber": event_position[gw] + 1,
                    "windowStart": window[0],
                    "windowEnd": window[1],
                    "windowIndex": local_index,
                    "windowLength": len(local_weeks),
                    "windowProgress": local_index / max(1, len(local_weeks) - 1),
                }
            )
    return pd.DataFrame(rows)


def target_column(chip: str) -> str:
    return (
        "actualTripleCaptainGain"
        if chip == "Triple Captain"
        else "actualBenchBoostGain"
    )


def raw_signal_column(chip: str) -> str:
    return (
        "predictedTripleCaptainGain"
        if chip == "Triple Captain"
        else "predictedBenchBoostGain"
    )


def features(chip: str) -> tuple[str, ...]:
    return TC_FEATURES if chip == "Triple Captain" else BB_FEATURES


def xgb(seed: int) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=140,
        max_depth=2,
        learning_rate=0.035,
        min_child_weight=14,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.20,
        reg_lambda=3.0,
        objective="reg:pseudohubererror",
        eval_metric="mae",
        n_jobs=-1,
        random_state=seed,
    )


def causal_value_predictions(frame: pd.DataFrame) -> tuple[dict[str, np.ndarray], list[dict]]:
    predictions = {
        chip: frame[raw_signal_column(chip)].to_numpy(float).copy()
        for chip in CHIPS
    }
    audit: list[dict] = []
    orders = frame["seasonOrder"].to_numpy(int)
    for season_order in sorted(frame["seasonOrder"].unique()):
        if season_order == 0:
            continue
        train_mask = orders < season_order
        test_mask = orders == season_order
        train = frame.loc[train_mask]
        test = frame.loc[test_mask]
        for chip_index, chip in enumerate(CHIPS):
            columns = list(features(chip))
            train_x = train[columns].astype(float).to_numpy()
            test_x = test[columns].astype(float).to_numpy()
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train_x)
            test_scaled = scaler.transform(test_x)
            target = train[target_column(chip)].to_numpy(float)
            age = season_order - train["seasonOrder"].to_numpy(int)
            weights = np.power(0.88, np.maximum(age - 1, 0))
            ridge = Ridge(alpha=35.0)
            ridge.fit(train_scaled, target, sample_weight=weights)
            tree = xgb(261000 + season_order * 10 + chip_index)
            tree.fit(train_x, target, sample_weight=weights)
            prediction = 0.55 * ridge.predict(test_scaled) + 0.45 * tree.predict(test_x)
            predictions[chip][test_mask] = np.clip(prediction, -3, 35)
            audit.append(
                {
                    "season": str(test["season"].iloc[0]),
                    "chip": chip,
                    "trainingSeasons": int(season_order),
                    "trainingRows": int(train_mask.sum()),
                    "testRows": int(test_mask.sum()),
                }
            )
    return predictions, audit


def risk_proxy(row: pd.Series, chip: str) -> float:
    if chip == "Triple Captain":
        return float(row["captainUncertainty"]) * (
            1.15 - 0.35 * float(row["captainPlayProbability"])
        )
    return float(row["benchUncertainty"]) + 1.5 * float(row["benchBlankCount"])


def choose_sequentially(
    season_frame: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    prior_frame: pd.DataFrame,
    prior_predictions: dict[str, np.ndarray],
    policy: Policy,
) -> list[dict]:
    choices: list[dict] = []
    season = str(season_frame["season"].iloc[0])
    weeks = season_frame.sort_values("eventNumber")["gw"].astype(int).tolist()
    for start, end in legal_windows(season, weeks):
        local = season_frame[
            season_frame["gw"].astype(int).between(start, end)
        ].sort_values("eventNumber")
        unused = set(CHIPS)
        for local_position, (index, row) in enumerate(local.iterrows()):
            remaining_weeks = len(local) - local_position
            candidates: list[tuple[float, str, float, float]] = []
            must_use = remaining_weeks <= len(unused)
            for chip in sorted(unused):
                current = float(predictions[chip][index]) - policy.risk_penalty * risk_proxy(row, chip)
                prior_scores = (
                    prior_predictions[chip]
                    - policy.risk_penalty
                    * prior_frame.apply(lambda item: risk_proxy(item, chip), axis=1).to_numpy(float)
                )
                if len(prior_scores):
                    # Early in a window demand the configured upper quantile.
                    # The reservation quantile decays continuously to zero as
                    # expiry approaches, making an otherwise-unused chip cheap.
                    remaining_fraction = max(0.0, (remaining_weeks - len(unused)) / max(1, len(local)))
                    effective_quantile = policy.quantile * remaining_fraction
                    threshold = float(np.quantile(prior_scores, effective_quantile))
                    scale = max(1.0, float(np.std(prior_scores)))
                else:
                    threshold = 6.0 if chip == "Triple Captain" else 8.0
                    scale = 4.0
                if must_use or current >= threshold:
                    urgency = (current - threshold) / scale
                    candidates.append((urgency, chip, current, threshold))
            if not candidates:
                continue
            _, chip, score, threshold = max(candidates, key=lambda item: item[0])
            choices.append(
                {
                    "chip": chip,
                    "gw": int(row["gw"]),
                    "predictedValue": round(score, 3),
                    "reservationValue": round(threshold, 3),
                    "actualGain": float(row[target_column(chip)]),
                    "fixtureCount": int(
                        row["captainFixtureCount"]
                        if chip == "Triple Captain"
                        else row["benchFixtureCount"]
                    ),
                    "forcedByExpiry": bool(must_use and score < threshold),
                }
            )
            unused.remove(chip)
            if not unused:
                break
    return choices


def best_assignment(frame: pd.DataFrame, score_columns: dict[str, str]) -> tuple[float, list[dict]]:
    """Diagnostic full-window assignment with one chip per GW."""
    total = 0.0
    choices: list[dict] = []
    season = str(frame["season"].iloc[0])
    weeks = frame.sort_values("eventNumber")["gw"].astype(int).tolist()
    for start, end in legal_windows(season, weeks):
        local = frame[frame["gw"].astype(int).between(start, end)]
        best: tuple[float, tuple[int, int]] | None = None
        for tc_index, bb_index in itertools.product(local.index, repeat=2):
            if tc_index == bb_index:
                continue
            value = float(local.loc[tc_index, score_columns["Triple Captain"]]) + float(
                local.loc[bb_index, score_columns["Bench Boost"]]
            )
            if best is None or value > best[0]:
                best = (value, (int(tc_index), int(bb_index)))
        if best is None:
            continue
        total += best[0]
        for chip, index in zip(CHIPS, best[1]):
            choices.append(
                {
                    "chip": chip,
                    "gw": int(frame.loc[index, "gw"]),
                    "value": round(float(frame.loc[index, score_columns[chip]]), 2),
                }
            )
    return total, choices


def main() -> None:
    original, _ = lens.load_or_build_prepared_history()
    data = add_targets(original.reset_index(drop=True))
    scores, baseline_plan, captain = championship_forecasts(data)
    structural = structural_horizons(data, scores)
    learned, _ = causal_ridge_horizons(data, structural)
    online, _ = causal_online_ridge_horizons(data, structural, learned)
    structural_adaptive = adaptive_value(data, structural, 3.0)
    ridge_value = quantile_map(
        data, adaptive_value(data, online, 3.0), baseline_plan
    )
    direct_raw, _ = causal_online_prediction(data, structural_adaptive)
    direct_value = quantile_map(data, direct_raw, baseline_plan)
    decision_ensemble = 0.50 * ridge_value + 0.50 * direct_value
    active = 0.15 * (event_number(data) >= 13)
    plan = baseline_plan + active * (decision_ensemble - baseline_plan)
    seasons = list(dict.fromkeys(data["season"].tolist()))
    training_count = len(lens.TRAINING_SEASONS)

    print("Running causal multi-timescale no-chip path", flush=True)
    no_chip_totals, no_chip_stats = lens.simulate_candidate(
        data,
        scores,
        STRATEGY,
        plan_scores=plan,
        captain_scores=captain,
        tracked_player_name="Salah",
    )
    frame = opportunity_frame(no_chip_stats, seasons)
    predictions, fit_audit = causal_value_predictions(frame)
    for chip in CHIPS:
        frame[f"causal{chip.replace(' ', '')}Value"] = predictions[chip]

    policies = []
    for quantile in RESERVATION_QUANTILES:
        for penalty in RISK_PENALTIES:
            policy = Policy(quantile, penalty)
            season_rows = []
            gains = []
            for season_order, season in enumerate(seasons):
                season_frame = frame[frame["seasonOrder"] == season_order]
                prior_frame = frame[frame["seasonOrder"] < season_order]
                local_predictions = {
                    chip: predictions[chip] for chip in CHIPS
                }
                prior_predictions = {
                    chip: predictions[chip][prior_frame.index.to_numpy(int)]
                    for chip in CHIPS
                }
                choices = choose_sequentially(
                    season_frame,
                    local_predictions,
                    prior_frame,
                    prior_predictions,
                    policy,
                )
                gain = float(sum(row["actualGain"] for row in choices))
                gains.append(gain)
                season_rows.append(
                    {
                        "season": season.replace("-", "/"),
                        "gain": round(gain, 1),
                        "points": round(float(no_chip_totals[season_order] + gain)),
                        "choices": choices,
                    }
                )
            training = np.asarray(gains[:training_count], dtype=float)
            evaluation = np.asarray(gains[training_count:], dtype=float)
            policies.append(
                {
                    "quantile": quantile,
                    "riskPenalty": penalty,
                    "trainingStabilityGain": round(
                        float(training.mean() - 0.25 * training.std()), 3
                    ),
                    "evaluationAverageGain": round(float(evaluation.mean()), 1),
                    "evaluationMinimumGain": round(float(evaluation.min()), 1),
                    "evaluationAveragePoints": round(
                        float(no_chip_totals[training_count:].mean() + evaluation.mean()), 1
                    ),
                    "seasons": season_rows,
                }
            )
    selected = max(policies, key=lambda row: row["trainingStabilityGain"])
    selected_penalty = float(selected["riskPenalty"])
    reservation_calibration = {}
    for chip in CHIPS:
        adjusted = predictions[chip] - selected_penalty * frame.apply(
            lambda row: risk_proxy(row, chip), axis=1
        ).to_numpy(float)
        reservation_calibration[chip] = {
            f"q{int(100 * quantile):02d}": round(
                float(np.quantile(adjusted, quantile)), 3
            )
            for quantile in (0.25, 0.50, 0.65, 0.75, 0.85, 0.90)
        }

    print("Running old audited TC/BB policy on the same path", flush=True)
    old_totals, old_stats = lens.simulate_candidate(
        data,
        scores,
        STRATEGY,
        chip_policy=lens.AUDITED_CHAMPION_CHIP_POLICY,
        plan_scores=plan,
        captain_scores=captain,
        tracked_player_name="Salah",
    )
    old_gain = old_totals - no_chip_totals

    predicted_bound = []
    oracle_bound = []
    for season_order, season in enumerate(seasons):
        season_frame = frame[frame["seasonOrder"] == season_order]
        predicted_total, predicted_choices = best_assignment(
            season_frame,
            {
                "Triple Captain": "causalTripleCaptainValue",
                "Bench Boost": "causalBenchBoostValue",
            },
        )
        oracle_total, oracle_choices = best_assignment(
            season_frame,
            {
                "Triple Captain": "actualTripleCaptainGain",
                "Bench Boost": "actualBenchBoostGain",
            },
        )
        predicted_bound.append(
            {
                "season": season.replace("-", "/"),
                "predictedObjective": round(predicted_total, 1),
                "choices": predicted_choices,
            }
        )
        oracle_bound.append(
            {
                "season": season.replace("-", "/"),
                "actualGain": round(oracle_total, 1),
                "choices": oracle_choices,
            }
        )

    selected_eval = selected["seasons"][training_count:]
    result = {
        "status": "training-selected causal optimal-stopping chip challenger",
        "currentRules": (
            "2026/27 has two complete sets of Wildcard, Free Hit, Triple Captain "
            "and Bench Boost; the first set expires at the GW19 deadline."
        ),
        "method": (
            "TC/BB marginal values are learned from prior seasons only. Each week "
            "compares current risk-adjusted value with an empirical reservation "
            "value that decays toward expiry. Strong single fixtures are eligible; "
            "unused chips are forced into distinct final weeks."
        ),
        "fitAudit": fit_audit,
        "historicalReservationCalibration": reservation_calibration,
        "noChipEvaluationAverage": round(float(no_chip_totals[training_count:].mean()), 1),
        "oldAuditedPolicy": {
            "evaluationAverageGain": round(float(old_gain[training_count:].mean()), 1),
            "evaluationMinimumGain": round(float(old_gain[training_count:].min()), 1),
            "evaluationAveragePoints": round(float(old_totals[training_count:].mean()), 1),
            "seasonGain": old_gain[training_count:].round().astype(int).tolist(),
            "choices": [row["chips"] for row in old_stats[training_count:]],
        },
        "selected": selected,
        "pairedVsOld": [
            {
                "season": row["season"],
                "oldGain": round(float(old_gain[training_count + index]), 1),
                "newGain": row["gain"],
                "delta": round(row["gain"] - float(old_gain[training_count + index]), 1),
            }
            for index, row in enumerate(selected_eval)
        ],
        "diagnosticBounds": {
            "warning": "Full-window predicted and hindsight assignments are noncausal and cannot be promoted.",
            "predicted": predicted_bound,
            "oracle": oracle_bound,
            "oracleEvaluationAverageGain": round(
                float(np.mean([row["actualGain"] for row in oracle_bound[training_count:]])), 1
            ),
        },
        "policies": policies,
    }
    output = lens.ROOT / "analysis" / "data" / "sequential_chip_value_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "noChipAverage": result["noChipEvaluationAverage"],
                "oldAuditedPolicy": result["oldAuditedPolicy"],
                "selected": {
                    key: selected[key]
                    for key in (
                        "quantile",
                        "riskPenalty",
                        "trainingStabilityGain",
                        "evaluationAverageGain",
                        "evaluationMinimumGain",
                        "evaluationAveragePoints",
                    )
                },
                "pairedVsOld": result["pairedVsOld"],
                "oracleEvaluationAverageGain": result["diagnosticBounds"]["oracleEvaluationAverageGain"],
                "all": [
                    {
                        "quantile": row["quantile"],
                        "riskPenalty": row["riskPenalty"],
                        "training": row["trainingStabilityGain"],
                        "evaluation": row["evaluationAverageGain"],
                        "minimum": row["evaluationMinimumGain"],
                    }
                    for row in policies
                ],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
