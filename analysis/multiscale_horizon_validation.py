"""Replace the fixed six-GW planning assumption with player-specific tenure.

This is a research challenger, not a production promotion.  It separates four
questions which the former six-week label mixed together:

* what a player is worth now (one GW);
* whether a short fixture punt is worth holding for three GWs;
* whether a normal transfer survives for six GWs; and
* whether a durable anchor/premium retains value over ten GWs.

All learned predictions for a season are fitted on earlier seasons only.  The
oracle rows deliberately use future results and are reported only as diagnostic
upper bounds; they can never be selected or promoted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import calibrate_model as lens
from bench_efficiency_validation import championship_forecasts, variant_summary
from frontier_ranker_validation import FEATURES as IMMEDIATE_FEATURES
from frontier_ranker_validation import STRATEGY, selectable_frontier
from listwise_ranker_validation import quantile_map


HORIZONS = (1, 3, 6, 10)
DISCOUNT = 0.86
CACHE_VERSION = 2
ONLINE_CHECKPOINTS = (13, 25)
FEATURES = list(
    dict.fromkeys(
        IMMEDIATE_FEATURES
        + [
            "component_horizon_censored",
            "causal_horizon_ridge",
            "horizon_weighted_games_censored",
            "fixture_censored",
            "fixture_now",
            "team_context",
            "team_attack",
            "team_defence",
            "recent_underlying",
            "long_underlying",
            "recent_value",
            "long_value",
            "minutes_model_confidence",
            "observations",
            "rotation_volatility",
            "team_rating_confidence",
            "team_regime_shift",
            "transfer_pressure_rank",
            "price_rise_probability",
            "price_fall_probability",
            "competition_pressure",
            "prediction_uncertainty",
            "position_id",
            "GW",
        ]
    )
)


@dataclass(frozen=True)
class HorizonAudit:
    horizon: int
    mae: float
    spearman: float
    top15_target: float


def add_targets(data: pd.DataFrame) -> pd.DataFrame:
    """Add discounted, inclusive forward labels for each declared horizon."""
    work = data.copy()
    points = work.groupby(["season", "player_key"], sort=False)["points"]
    gw = work.groupby(["season", "player_key"], sort=False)["GW"]
    for horizon in HORIZONS:
        target = pd.Series(0.0, index=work.index)
        target_end = work["GW"].astype(int).copy()
        for offset in range(horizon):
            target += (DISCOUNT**offset) * points.shift(-offset).fillna(0)
            shifted_gw = gw.shift(-offset)
            target_end = np.maximum(
                target_end,
                shifted_gw.fillna(target_end).astype(int),
            )
        work[f"target_h{horizon}"] = target
        work[f"target_h{horizon}_end_gw"] = target_end.astype(int)
    return work


def durability(data: pd.DataFrame) -> np.ndarray:
    """Deadline-known probability that a transfer remains useful.

    Availability dominates.  Evidence depth and team stability prevent a tiny
    hot sample from being treated like an established 90-minute role.
    """
    observations = 1 - np.exp(-data["observations"].to_numpy(float) / 16.0)
    minutes_confidence = np.clip(
        data["minutes_model_confidence"].to_numpy(float) / 0.64, 0, 1
    )
    availability = (
        0.38 * data["play_probability"].to_numpy(float)
        + 0.24 * data["start_probability"].to_numpy(float)
        + 0.18 * data["sixty_probability"].to_numpy(float)
        + 0.20 * (1 - data["rotation_volatility"].to_numpy(float))
    )
    evidence = (
        0.40 * minutes_confidence
        + 0.30 * observations
        + 0.30 * data["team_rating_confidence"].to_numpy(float)
    )
    regime_penalty = 0.12 * data["team_regime_shift"].to_numpy(float)
    return np.clip(0.78 * availability + 0.22 * evidence - regime_penalty, 0.12, 0.96)


def expected_tenure(data: pd.DataFrame) -> np.ndarray:
    """Map durability to a two-to-ten-GW expected holding period."""
    stable = durability(data)
    return np.clip(2.0 + 8.0 * stable, 2.0, 10.0)


def remaining_events(data: pd.DataFrame) -> np.ndarray:
    """Number of replay scoring events left, including the current event.

    This is based on event order, not numeric GW subtraction: the 2019/20
    source resumes at GW39 after GW29, while still containing 38 scoring
    events.  The season length is part of the competition format; no future
    opponent or result enters this value.
    """
    remaining = np.zeros(len(data), dtype=float)
    for _, season_frame in data.groupby("season", sort=False):
        weeks = list(dict.fromkeys(season_frame["GW"].astype(int).tolist()))
        week_position = {gw: index for index, gw in enumerate(weeks)}
        local = season_frame["GW"].astype(int).map(week_position).to_numpy(int)
        remaining[season_frame.index.to_numpy(int)] = len(weeks) - local
    return remaining


def structural_horizons(
    data: pd.DataFrame, immediate: np.ndarray
) -> dict[int, np.ndarray]:
    """Construct schedule-censored forecasts on comparable cumulative scales."""
    six_weights = sum(DISCOUNT**offset for offset in range(6))
    six_total = data["component_horizon_censored"].to_numpy(float)
    structural_immediate = data["component_xpts"].to_numpy(float)
    ratio = np.divide(
        immediate,
        structural_immediate,
        out=np.ones_like(immediate),
        where=structural_immediate > 0.20,
    )
    six_total = six_total * np.clip(ratio, 0.60, 1.45)
    steady_rate = np.divide(
        six_total,
        six_weights,
        out=np.maximum(immediate, 0.0),
        where=six_total > 0,
    )
    remaining = remaining_events(data)
    forecasts: dict[int, np.ndarray] = {}
    for horizon in HORIZONS:
        if horizon == 1:
            forecasts[horizon] = immediate.copy()
            continue
        effective = np.minimum(remaining, float(horizon)).astype(int)
        future_weight = np.asarray(
            [sum(DISCOUNT**offset for offset in range(1, count)) for count in effective],
            dtype=float,
        )
        forecasts[horizon] = np.clip(
            immediate + steady_rate * future_weight, 0, 70
        )
    return forecasts


def interpolate_horizon(
    tenure: np.ndarray, horizons: dict[int, np.ndarray]
) -> np.ndarray:
    """Linearly interpolate each row between the 1/3/6/10-GW value functions."""
    result = np.empty(len(tenure), dtype=float)
    knots = np.asarray(HORIZONS, dtype=float)
    stacked = np.column_stack([horizons[horizon] for horizon in HORIZONS])
    for row_index, hold in enumerate(tenure):
        upper_index = int(np.searchsorted(knots, hold, side="right"))
        if upper_index == 0:
            result[row_index] = stacked[row_index, 0]
        elif upper_index >= len(knots):
            result[row_index] = stacked[row_index, -1]
        else:
            lower_index = upper_index - 1
            fraction = (hold - knots[lower_index]) / (
                knots[upper_index] - knots[lower_index]
            )
            result[row_index] = (
                (1 - fraction) * stacked[row_index, lower_index]
                + fraction * stacked[row_index, upper_index]
            )
    return result


def adaptive_value(
    data: pd.DataFrame,
    horizons: dict[int, np.ndarray],
    exit_cost_scale: float,
) -> np.ndarray:
    raw_tenure = expected_tenure(data)
    remaining = remaining_events(data)
    tenure = np.minimum(raw_tenure, remaining)
    value = interpolate_horizon(tenure, horizons)
    # A short-lived transfer consumes a future free transfer.  The cost fades
    # to zero at a normal six-GW tenure and is never charged to durable anchors.
    replacement_expected = remaining > raw_tenure
    exit_cost = (
        exit_cost_scale
        * np.clip((6.0 - raw_tenure) / 4.0, 0, 1)
        * replacement_expected
    )
    return value - exit_cost


def feature_matrix(
    frame: pd.DataFrame,
    medians: pd.Series | None = None,
) -> tuple[np.ndarray, pd.Series]:
    values = frame[FEATURES].replace([np.inf, -np.inf], np.nan).astype(float).copy()
    values["price"] /= 150.0
    values["selected"] = np.log1p(values["selected"].clip(lower=0)) / 16.0
    values["component_horizon_censored"] /= 35.0
    values["causal_horizon_ridge"] /= 35.0
    values["horizon_weighted_games_censored"] /= 6.0
    values["expected_minutes"] /= 90.0
    values["minutes_std"] /= 40.0
    values["observations"] = np.log1p(values["observations"]) / 6.0
    values["GW"] /= 38.0
    if medians is None:
        medians = values.median().fillna(0)
    return values.fillna(medians).to_numpy(np.float32), medians


def causal_ridge_horizons(
    data: pd.DataFrame,
    structural: dict[int, np.ndarray],
) -> tuple[dict[int, np.ndarray], list[dict]]:
    """Prior-season-only multi-target ridge, preserving structural scale."""
    cache_path = lens.CACHE / f"multiscale-ridge-v{CACHE_VERSION}.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        if len(cached["h3"]) == len(data):
            return (
                {horizon: cached[f"h{horizon}"] for horizon in HORIZONS},
                json.loads(str(cached["audit"].item())),
            )

    raw = {horizon: structural[horizon].copy() for horizon in HORIZONS}
    seasons = list(dict.fromkeys(data["season"].tolist()))
    orders = data["season_order"].to_numpy(int)
    positions = data["position_id"].to_numpy(int)
    observed = data["fixture_count"].to_numpy(int) > 0
    frontier = selectable_frontier(data)
    audit: list[dict] = []
    target_columns = [f"target_h{horizon}" for horizon in HORIZONS[1:]]
    for season_order in range(1, len(seasons)):
        season_mask = orders == season_order
        for position in lens.SQUAD_QUOTAS:
            train_mask = (
                (orders < season_order)
                & (positions == position)
                & observed
                & frontier
            )
            test_mask = season_mask & (positions == position)
            train = data.loc[train_mask]
            test = data.loc[test_mask]
            train_x, medians = feature_matrix(train)
            test_x, _ = feature_matrix(test, medians)
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train_x)
            test_scaled = scaler.transform(test_x)
            target = train[target_columns].to_numpy(float)
            age = season_order - train["season_order"].to_numpy(int)
            weights = np.power(0.88, np.maximum(age - 1, 0))
            fitted = Ridge(alpha=80.0)
            fitted.fit(train_scaled, target, sample_weight=weights)
            prediction = fitted.predict(test_scaled)
            for column_index, horizon in enumerate(HORIZONS[1:]):
                raw[horizon][test_mask] = np.clip(
                    prediction[:, column_index], 0, 70
                )
            raw[1][test_mask] = structural[1][test_mask]
            audit.append(
                {
                    "season": seasons[season_order],
                    "position": int(position),
                    "trainingRows": int(train_mask.sum()),
                    "testRows": int(test_mask.sum()),
                    "latestTrainingSeason": seasons[season_order - 1],
                }
            )
            print(
                f"Multi-scale ridge {seasons[season_order]} position {position}",
                flush=True,
            )
        for horizon in HORIZONS:
            raw[horizon][season_mask & ~observed] = 0

    mapped = {1: raw[1]}
    for horizon in HORIZONS[1:]:
        mapped[horizon] = quantile_map(data, raw[horizon], structural[horizon])
    np.savez_compressed(
        cache_path,
        **{f"h{horizon}": mapped[horizon] for horizon in HORIZONS},
        audit=json.dumps(audit),
    )
    return mapped, audit


def causal_online_ridge_horizons(
    data: pd.DataFrame,
    structural: dict[int, np.ndarray],
    static: dict[int, np.ndarray],
) -> tuple[dict[int, np.ndarray], list[dict]]:
    """Update each horizon only when its same-season labels have matured."""
    cache_path = lens.CACHE / f"multiscale-online-ridge-v{CACHE_VERSION}.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        if len(cached["h3"]) == len(data):
            return (
                {horizon: cached[f"h{horizon}"] for horizon in HORIZONS},
                json.loads(str(cached["audit"].item())),
            )

    raw = {horizon: static[horizon].copy() for horizon in HORIZONS}
    seasons = list(dict.fromkeys(data["season"].tolist()))
    orders = data["season_order"].to_numpy(int)
    gameweeks = data["GW"].to_numpy(int)
    positions = data["position_id"].to_numpy(int)
    observed = data["fixture_count"].to_numpy(int) > 0
    frontier = selectable_frontier(data)
    audit: list[dict] = []
    for season_order in range(1, len(seasons)):
        season_mask = orders == season_order
        for checkpoint_index, checkpoint in enumerate(ONLINE_CHECKPOINTS):
            interval_end = (
                ONLINE_CHECKPOINTS[checkpoint_index + 1] - 1
                if checkpoint_index + 1 < len(ONLINE_CHECKPOINTS)
                else 99
            )
            for position in lens.SQUAD_QUOTAS:
                position_mask = positions == position
                test_mask = (
                    season_mask
                    & position_mask
                    & (gameweeks >= checkpoint)
                    & (gameweeks <= interval_end)
                )
                test = data.loc[test_mask]
                if test.empty:
                    continue
                test_x, _ = feature_matrix(test)
                for horizon in HORIZONS[1:]:
                    matured = data[f"target_h{horizon}_end_gw"].to_numpy(int) < checkpoint
                    train_mask = (
                        position_mask
                        & observed
                        & frontier
                        & ((orders < season_order) | (season_mask & matured))
                    )
                    train = data.loc[train_mask]
                    train_x, medians = feature_matrix(train)
                    test_x, _ = feature_matrix(test, medians)
                    scaler = StandardScaler()
                    train_scaled = scaler.fit_transform(train_x)
                    test_scaled = scaler.transform(test_x)
                    age = season_order - train["season_order"].to_numpy(int)
                    weights = np.power(0.88, np.maximum(age - 1, 0))
                    # Current-season evidence receives enough weight to react,
                    # without letting three early GWs erase several seasons.
                    weights *= np.where(
                        train["season_order"].to_numpy(int) == season_order,
                        1.35,
                        1.0,
                    )
                    fitted = Ridge(alpha=80.0)
                    fitted.fit(
                        train_scaled,
                        train[f"target_h{horizon}"].to_numpy(float),
                        sample_weight=weights,
                    )
                    raw[horizon][test_mask] = np.clip(
                        fitted.predict(test_scaled), 0, 70
                    )
                    audit.append(
                        {
                            "season": seasons[season_order],
                            "checkpoint": checkpoint,
                            "horizon": horizon,
                            "position": int(position),
                            "trainingRows": int(train_mask.sum()),
                            "maturedCurrentSeasonRows": int(
                                (train_mask & season_mask).sum()
                            ),
                            "testRows": int(test_mask.sum()),
                        }
                    )
            print(
                f"Online multi-scale ridge {seasons[season_order]} GW{checkpoint}",
                flush=True,
            )
        for horizon in HORIZONS:
            raw[horizon][season_mask & ~observed] = 0

    mapped = {1: raw[1]}
    for horizon in HORIZONS[1:]:
        mapped[horizon] = quantile_map(data, raw[horizon], structural[horizon])
    np.savez_compressed(
        cache_path,
        **{f"h{horizon}": mapped[horizon] for horizon in HORIZONS},
        audit=json.dumps(audit),
    )
    return mapped, audit


def forecast_audit(
    data: pd.DataFrame,
    forecasts: dict[int, np.ndarray],
) -> list[HorizonAudit]:
    rows: list[HorizonAudit] = []
    observed = data["fixture_count"].to_numpy(int) > 0
    evaluation = data["season"].isin(lens.EVALUATION_SEASONS).to_numpy(bool)
    valid = observed & evaluation
    for horizon in HORIZONS:
        actual = data[f"target_h{horizon}"].to_numpy(float)
        prediction = forecasts[horizon]
        correlations: list[float] = []
        top_targets: list[float] = []
        for _, indices in data.groupby(["season", "GW", "position_id"], sort=False).groups.items():
            local = np.asarray(indices, dtype=int)
            local = local[valid[local]]
            if len(local) < 10:
                continue
            correlation = spearmanr(prediction[local], actual[local]).statistic
            if np.isfinite(correlation):
                correlations.append(float(correlation))
            top = local[np.argsort(prediction[local])[-15:]]
            top_targets.append(float(actual[top].mean()))
        rows.append(
            HorizonAudit(
                horizon=horizon,
                mae=round(float(np.abs(prediction[valid] - actual[valid]).mean()), 4),
                spearman=round(float(np.mean(correlations)), 4),
                top15_target=round(float(np.mean(top_targets)), 4),
            )
        )
    return rows


def main() -> None:
    original, _ = lens.load_or_build_prepared_history()
    data = add_targets(original.reset_index(drop=True))
    scores, baseline_plan, captain = championship_forecasts(data)
    structural = structural_horizons(data, scores)
    learned, fit_audit = causal_ridge_horizons(data, structural)
    online, online_fit_audit = causal_online_ridge_horizons(
        data, structural, learned
    )
    tenure = expected_tenure(data)

    structural_adaptive = adaptive_value(data, structural, exit_cost_scale=3.0)
    learned_adaptive = adaptive_value(data, learned, exit_cost_scale=3.0)
    online_adaptive = adaptive_value(data, online, exit_cost_scale=3.0)
    structural_mapped = quantile_map(data, structural_adaptive, baseline_plan)
    learned_mapped = quantile_map(data, learned_adaptive, baseline_plan)
    online_mapped = quantile_map(data, online_adaptive, baseline_plan)

    oracle_horizons = {
        horizon: data[f"target_h{horizon}"].to_numpy(float)
        for horizon in HORIZONS
    }
    oracle_adaptive = adaptive_value(data, oracle_horizons, exit_cost_scale=0.0)
    oracle_mapped = quantile_map(data, oracle_adaptive, baseline_plan)

    configs: list[tuple[str, np.ndarray, bool]] = [("baseline", baseline_plan, False)]
    for share in (0.05, 0.10, 0.25, 0.40):
        configs.append(
            (
                f"structuralAdaptive{int(share * 100)}",
                (1 - share) * baseline_plan + share * structural_mapped,
                False,
            )
        )
        configs.append(
            (
                f"onlineAdaptive{int(share * 100)}",
                (1 - share) * baseline_plan + share * online_mapped,
                False,
            )
        )
        configs.append(
            (
                f"learnedAdaptive{int(share * 100)}",
                (1 - share) * baseline_plan + share * learned_mapped,
                False,
            )
        )
    # Diagnostic only.  It proves whether the objective has exploitable ceiling;
    # it is excluded from every selection comparison below.
    configs.append(
        ("oracleAdaptive25DiagnosticOnly", 0.75 * baseline_plan + 0.25 * oracle_mapped, True)
    )

    seasons = list(dict.fromkeys(data["season"].tolist()))
    rows = []
    raw_results: dict[str, tuple[np.ndarray, list[dict]]] = {}
    for name, plan, oracle in configs:
        print(f"Running {name}", flush=True)
        totals, stats = lens.simulate_candidate(
            data,
            scores,
            STRATEGY,
            plan_scores=plan,
            captain_scores=captain,
            tracked_player_name="Salah",
        )
        training = totals[: len(lens.TRAINING_SEASONS)]
        row = {
            "name": name,
            "oracle": oracle,
            "trainingStability": None
            if oracle
            else round(float(training.mean() - 0.25 * training.std()), 3),
            "summary": variant_summary(totals, stats, seasons),
        }
        rows.append(row)
        raw_results[name] = (totals, stats)

    eligible = [row for row in rows if not row["oracle"]]
    selected = max(eligible, key=lambda row: row["trainingStability"])
    baseline = next(row for row in rows if row["name"] == "baseline")
    selected_totals, _ = raw_results[selected["name"]]
    baseline_totals, _ = raw_results["baseline"]
    evaluation_mask = np.asarray(
        [season in lens.EVALUATION_SEASONS for season in seasons], dtype=bool
    )
    paired_delta = selected_totals[evaluation_mask] - baseline_totals[evaluation_mask]
    robust_promotion = bool(
        selected["name"] != "baseline"
        and selected["summary"]["average"] > baseline["summary"]["average"]
        and selected["summary"]["minimum"] >= baseline["summary"]["minimum"]
        and int((paired_delta > 0).sum()) >= 5
    )

    result = {
        "status": "multi-timescale research challenger",
        "method": (
            "Causal 1/3/6/10-GW value functions. Expected tenure is computed "
            "from deadline-known availability, role stability, evidence depth, "
            "and team stability; low-tenure transfers pay an exit/option cost. "
            "Ridge models use prior seasons only and are quantile-mapped to the "
            "structural planning scale."
        ),
        "fixedSixWeekDiagnosis": {
            "sameTenureForEveryPlayer": True,
            "futureScheduleTreatment": (
                "Only the current slate is observed in the censored backtest; "
                "later slots are neutral."
            ),
            "replacementCostInOldLabel": False,
            "playerSpecificExitProbabilityInOldLabel": False,
        },
        "tenure": {
            "mean": round(float(tenure.mean()), 3),
            "p10": round(float(np.quantile(tenure, 0.10)), 3),
            "median": round(float(np.median(tenure)), 3),
            "p90": round(float(np.quantile(tenure, 0.90)), 3),
        },
        "forecastAudit": {
            "structural": [row.__dict__ for row in forecast_audit(data, structural)],
            "causalRidge": [row.__dict__ for row in forecast_audit(data, learned)],
            "causalOnlineRidge": [
                row.__dict__ for row in forecast_audit(data, online)
            ],
        },
        "fitAudit": fit_audit,
        "onlineFitAudit": online_fit_audit,
        "selectionRule": (
            "Choose by the two untouched calibration seasons only. Promotion "
            "also requires a higher evaluation average, no lower minimum, and "
            "improvement in at least five of eight evaluation seasons. Oracle "
            "rows are never eligible."
        ),
        "selected": selected,
        "robustPromotion": robust_promotion,
        "experiments": rows,
    }
    output = lens.ROOT / "analysis" / "data" / "multiscale_horizon_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "selected": selected["name"],
                "robustPromotion": robust_promotion,
                "tenure": result["tenure"],
                "forecastAudit": result["forecastAudit"],
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
