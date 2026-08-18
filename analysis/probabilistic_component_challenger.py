"""Causal scoring-route distributions for player selection.

This is deliberately a challenger, not a silent replacement for the frozen
champion.  Each test season is fitted only on earlier seasons.  The model
learns the event counts which generate FPL points, converts them through the
actual position-specific scoring rules, and then lets a small causal ridge
stacker decide how much information survives beyond the champion projection.

The old component experiment regressed five already-bundled point totals.  It
therefore had neither valid event probabilities nor a coherent variance.  This
version predicts appearances, 60-minute appearances, goals, assists, clean
sheets, saves, goals conceded, bonus and defensive-contribution points as
non-negative counts, plus a signed residual for cards/own-goals/penalties.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

import calibrate_model as lens
from bench_efficiency_validation import variant_summary
from component_forecast_validation import FEATURES as BASE_FEATURES
from feasible_decision_audit import decision_metrics
from frontier_ranker_validation import STRATEGY, selectable_frontier
from listwise_ranker_validation import quantile_map
from multiscale_horizon_validation import add_targets
from wildcard_freehit_ablation import champion_forecasts


CACHE_VERSION = 2
COUNT_ROUTES = (
    "appearances_observed",
    "sixty_observed",
    "goals",
    "assists",
    "clean_sheets",
    "saves",
    "goals_conceded",
    "bonus",
    "current_rule_dc_points",
)
ROUTE_LABELS = (
    "appearance",
    "sixty",
    "goals",
    "assists",
    "cleanSheets",
    "saves",
    "goalsConceded",
    "bonus",
    "defensiveContribution",
    "other",
)
FEATURES = list(
    dict.fromkeys(
        BASE_FEATURES
        + [
            "defensive_return_probability",
            "defensive_event_coverage",
            "opponent_goal_vulnerability",
            "opponent_assist_vulnerability",
            "team_clean_rating",
            "opponent_clean_rating",
            "league_goal_rate",
            "table_goal_difference_before",
        ]
    )
)


def scoring_weights(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    position = data["position_id"].astype(int)
    goals = position.map({1: 6, 2: 6, 3: 5, 4: 4}).to_numpy(float)
    clean = position.map({1: 4, 2: 4, 3: 1, 4: 0}).to_numpy(float)
    return goals, clean


def known_route_points(data: pd.DataFrame) -> np.ndarray:
    """Observed points represented by the explicit routes."""
    goal_weight, clean_weight = scoring_weights(data)
    position = data["position_id"].astype(int).to_numpy()
    points = data["appearances_observed"].to_numpy(float).copy()
    points += data["sixty_observed"].to_numpy(float)
    points += goal_weight * data["goals"].to_numpy(float)
    points += 3 * data["assists"].to_numpy(float)
    points += clean_weight * data["clean_sheets"].to_numpy(float)
    points += np.where(position == 1, np.floor(data["saves"].to_numpy(float) / 3), 0)
    points -= np.where(
        np.isin(position, [1, 2]),
        np.floor(data["goals_conceded"].to_numpy(float) / 2),
        0,
    )
    points += data["bonus"].to_numpy(float)
    points += data["current_rule_dc_points"].to_numpy(float)
    return points


def feature_matrix(
    frame: pd.DataFrame, medians: pd.Series | None = None
) -> tuple[np.ndarray, pd.Series]:
    values = frame[FEATURES].replace([np.inf, -np.inf], np.nan).astype(float).copy()
    for column in ("expected_minutes",):
        values[column] /= 90.0
    values["minutes_std"] /= 40.0
    values["price"] /= 150.0
    values["selected"] = np.log1p(values["selected"].clip(lower=0)) / 16.0
    values["observations"] = np.log1p(values["observations"].clip(lower=0)) / 6.0
    for column in ("recent_raw", "long_raw"):
        values[column] /= 8.0
    for column in ("recent_underlying_raw", "long_underlying_raw"):
        values[column] /= 20.0
    if medians is None:
        medians = values.median().fillna(0)
    return values.fillna(medians).to_numpy(np.float32), medians


def count_model(seed: int) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=125,
        max_depth=3,
        learning_rate=0.045,
        min_child_weight=18,
        subsample=0.82,
        colsample_bytree=0.80,
        reg_alpha=0.16,
        reg_lambda=3.0,
        objective="count:poisson",
        eval_metric="poisson-nloglik",
        max_delta_step=0.7,
        n_jobs=-1,
        random_state=seed,
    )


def residual_model(seed: int) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=135,
        max_depth=3,
        learning_rate=0.04,
        min_child_weight=20,
        subsample=0.82,
        colsample_bytree=0.80,
        reg_alpha=0.18,
        reg_lambda=3.2,
        objective="reg:pseudohubererror",
        eval_metric="mae",
        n_jobs=-1,
        random_state=seed,
    )


def poisson_floor_moments(rate: np.ndarray, divisor: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact E[floor(X/divisor)] and variance for Poisson X.

    Rates in this application are small.  A recurrence through 50 events is
    effectively exact while staying vectorised over every player-week.
    """
    local = np.clip(np.asarray(rate, dtype=float), 0, 30)
    probability = np.exp(-local)
    mean = np.zeros_like(local)
    second = np.zeros_like(local)
    for count in range(1, 51):
        probability = probability * local / count
        value = count // divisor
        mean += probability * value
        second += probability * value * value
    return mean, np.maximum(0, second - mean * mean)


def route_distribution(
    data: pd.DataFrame, means: np.ndarray, other_variance: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    goal_weight, clean_weight = scoring_weights(data)
    position = data["position_id"].astype(int).to_numpy()
    save_mean, save_var = poisson_floor_moments(means[:, 5], 3)
    conceded_mean, conceded_var = poisson_floor_moments(means[:, 6], 2)
    keeper = position == 1
    defender = np.isin(position, [1, 2])

    total = means[:, 0] + means[:, 1]
    total += goal_weight * means[:, 2] + 3 * means[:, 3]
    total += clean_weight * means[:, 4]
    total += keeper * save_mean - defender * conceded_mean
    total += means[:, 7] + means[:, 8] + means[:, 9]

    # Independent event variance is a useful first-order distribution.  A
    # causal scale correction below absorbs dependence and residual dispersion.
    variance = means[:, 0] + means[:, 1]
    variance += goal_weight**2 * means[:, 2] + 9 * means[:, 3]
    variance += clean_weight**2 * means[:, 4]
    variance += keeper * save_var + defender * conceded_var
    variance += means[:, 7] + means[:, 8] + other_variance
    return total, np.maximum(variance, 0.16)


def causal_route_predictions(
    data: pd.DataFrame,
    champion: np.ndarray,
    seed_offset: int = 0,
) -> tuple[dict, dict]:
    seed_suffix = "" if seed_offset == 0 else f"-seed{seed_offset}"
    cache_path = lens.CACHE / (
        f"probabilistic-routes-v{CACHE_VERSION}{seed_suffix}.npz"
    )
    if cache_path.exists():
        cached = np.load(cache_path)
        if len(cached["total"]) == len(data):
            return (
                {key: cached[key] for key in ("means", "total", "stacked", "sigma")},
                json.loads(str(cached["audit"].item())),
            )

    orders = data["season_order"].to_numpy(int)
    observed = data["fixture_count"].to_numpy(int) > 0
    frontier = selectable_frontier(data)
    seasons = list(dict.fromkeys(data["season"].tolist()))
    target_counts = data[list(COUNT_ROUTES)].to_numpy(float)
    other_target = data["points"].to_numpy(float) - known_route_points(data)
    means = np.zeros((len(data), len(ROUTE_LABELS)), dtype=float)
    # This initialisation is used only where there is no prior season.  Those
    # rows are never presented as a learned component result.
    means[:, 0] = np.maximum(champion, 0)
    other_variance = np.full(len(data), 2.0, dtype=float)
    fit_audit: list[dict] = []

    for season_order in range(1, len(seasons)):
        train_mask = (orders < season_order) & observed & frontier
        test_mask = orders == season_order
        train = data.loc[train_mask]
        test = data.loc[test_mask]
        train_x, medians = feature_matrix(train)
        test_x, _ = feature_matrix(test, medians)
        age = season_order - train["season_order"].to_numpy(int)
        base_weight = np.power(0.86, np.maximum(age - 1, 0))
        for route_index, route in enumerate(COUNT_ROUTES):
            target = target_counts[train_mask, route_index]
            event_weight = 1 + np.minimum(target, 5) * 0.08
            fitted = count_model(
                730000
                + 100000 * seed_offset
                + 100 * season_order
                + route_index
            )
            fitted.fit(train_x, target, sample_weight=base_weight * event_weight)
            means[test_mask, route_index] = np.clip(fitted.predict(test_x), 0, 15)

        other = residual_model(735000 + 100000 * seed_offset + season_order)
        other.fit(train_x, other_target[train_mask], sample_weight=base_weight)
        means[test_mask, 9] = np.clip(other.predict(test_x), -5, 3)

        # Estimate irreducible signed-route noise only from the fitting period.
        # Position-level estimates are more stable than another variance tree.
        train_other_prediction = other.predict(train_x)
        for position in lens.SQUAD_QUOTAS:
            local_train = train["position_id"].to_numpy(int) == position
            local_test = test["position_id"].to_numpy(int) == position
            if local_train.any():
                residual = other_target[train_mask][local_train] - train_other_prediction[local_train]
                other_variance[np.flatnonzero(test_mask)[local_test]] = max(
                    0.25, float(np.mean(residual**2))
                )
        fit_audit.append(
            {
                "season": seasons[season_order],
                "trainingRows": int(train_mask.sum()),
                "testRows": int(test_mask.sum()),
            }
        )
        print(f"Probabilistic routes predicted {seasons[season_order]}", flush=True)

    total, variance = route_distribution(data, means, other_variance)
    total[orders == 0] = champion[orders == 0]
    total[~observed] = 0

    # Calibrate dispersion causally by position.  Event independence otherwise
    # understates the very long tail created by correlated minutes and returns.
    sigma = np.sqrt(variance)
    for season_order in range(1, len(seasons)):
        for position in lens.SQUAD_QUOTAS:
            train_mask = (
                (orders < season_order)
                & (orders > 0)
                & observed
                & frontier
                & (data["position_id"].to_numpy(int) == position)
            )
            test_mask = (orders == season_order) & (data["position_id"].to_numpy(int) == position)
            if train_mask.sum() < 200:
                continue
            residual_variance = np.mean((data.loc[train_mask, "points"].to_numpy(float) - total[train_mask]) ** 2)
            predicted_variance = np.mean(variance[train_mask])
            scale = np.sqrt(residual_variance / max(predicted_variance, 0.1))
            sigma[test_mask] *= np.clip(scale, 0.65, 2.5)

    # OOF route estimates become features in a conservative causal stacker.
    stacked = total.copy()
    positions = data["position_id"].to_numpy(int)
    stack_features = np.column_stack([champion, total, means, sigma])
    for season_order in range(2, len(seasons)):
        for position in lens.SQUAD_QUOTAS:
            train_mask = (
                (orders > 0)
                & (orders < season_order)
                & observed
                & frontier
                & (positions == position)
            )
            test_mask = (orders == season_order) & (positions == position)
            scaler = StandardScaler()
            train_x = scaler.fit_transform(stack_features[train_mask])
            test_x = scaler.transform(stack_features[test_mask])
            age = season_order - orders[train_mask]
            weights = np.power(0.88, np.maximum(age - 1, 0))
            fitted = Ridge(alpha=80.0)
            fitted.fit(
                train_x,
                data.loc[train_mask, "points"].to_numpy(float),
                sample_weight=weights,
            )
            stacked[test_mask] = fitted.predict(test_x)
    stacked[orders < 2] = total[orders < 2]
    stacked[~observed] = 0

    audit = {
        "routes": list(ROUTE_LABELS),
        "features": FEATURES,
        "fits": fit_audit,
        "seedOffset": seed_offset,
        "causal": "Every route, variance scale and stacker uses prior seasons only.",
    }
    np.savez_compressed(
        cache_path,
        means=means,
        total=total,
        stacked=stacked,
        sigma=sigma,
        audit=json.dumps(audit),
    )
    return {"means": means, "total": total, "stacked": stacked, "sigma": sigma}, audit


def terminal_live_route_predictions(
    data: pd.DataFrame,
    live: pd.DataFrame,
    historical_champion: np.ndarray,
    live_champion: np.ndarray,
    seed_offset: int = 0,
) -> tuple[dict, dict]:
    """Fit scoring routes through the final completed season for a live deadline.

    Historical validation uses prior-season-only predictions.  At a genuinely
    later live deadline every completed historical row is available, so this is
    the matching terminal fit.  Dispersion and the route stack are calibrated
    from historical out-of-fold predictions rather than optimistic in-sample
    residuals.
    """
    observed = data["fixture_count"].to_numpy(int) > 0
    frontier = selectable_frontier(data)
    train_mask = observed & frontier
    train = data.loc[train_mask]
    train_x, medians = feature_matrix(train)
    live_x, _ = feature_matrix(live, medians)
    orders = data["season_order"].to_numpy(int)
    age = orders.max() - orders[train_mask]
    base_weight = np.power(0.86, np.maximum(age, 0))
    target_counts = data[list(COUNT_ROUTES)].to_numpy(float)
    other_target = data["points"].to_numpy(float) - known_route_points(data)
    means = np.zeros((len(live), len(ROUTE_LABELS)), dtype=float)
    fits: list[dict] = []

    for route_index, route in enumerate(COUNT_ROUTES):
        target = target_counts[train_mask, route_index]
        event_weight = 1 + np.minimum(target, 5) * 0.08
        fitted = count_model(840000 + 100000 * seed_offset + route_index)
        fitted.fit(train_x, target, sample_weight=base_weight * event_weight)
        means[:, route_index] = np.clip(fitted.predict(live_x), 0, 15)
        fits.append({"route": route, "trainingRows": int(train_mask.sum())})

    other = residual_model(845000 + 100000 * seed_offset)
    other.fit(train_x, other_target[train_mask], sample_weight=base_weight)
    means[:, 9] = np.clip(other.predict(live_x), -5, 3)
    train_other_prediction = other.predict(train_x)
    other_variance = np.full(len(live), 2.0, dtype=float)
    live_position = live["position_id"].to_numpy(int)
    train_position = train["position_id"].to_numpy(int)
    for position in lens.SQUAD_QUOTAS:
        local_train = train_position == position
        local_live = live_position == position
        if local_train.any():
            residual = (
                other_target[train_mask][local_train]
                - train_other_prediction[local_train]
            )
            other_variance[local_live] = max(0.25, float(np.mean(residual**2)))

    total, variance = route_distribution(live, means, other_variance)
    sigma = np.sqrt(variance)
    historical_component, _ = causal_route_predictions(
        data, historical_champion, seed_offset=seed_offset
    )
    historical_position = data["position_id"].to_numpy(int)
    out_of_fold = (orders > 0) & observed & frontier
    for position in lens.SQUAD_QUOTAS:
        scale_mask = out_of_fold & (historical_position == position)
        live_mask = live_position == position
        if scale_mask.sum() < 200:
            continue
        residual_variance = np.mean(
            (
                data.loc[scale_mask, "points"].to_numpy(float)
                - historical_component["total"][scale_mask]
            )
            ** 2
        )
        predicted_variance = np.mean(
            historical_component["sigma"][scale_mask] ** 2
        )
        sigma[live_mask] *= np.clip(
            np.sqrt(residual_variance / max(predicted_variance, 0.1)),
            0.65,
            2.5,
        )

    stacked = total.copy()
    historical_stack = np.column_stack(
        [
            historical_champion,
            historical_component["total"],
            historical_component["means"],
            historical_component["sigma"],
        ]
    )
    live_stack = np.column_stack([live_champion, total, means, sigma])
    for position in lens.SQUAD_QUOTAS:
        stack_mask = out_of_fold & (historical_position == position)
        live_mask = live_position == position
        if stack_mask.sum() < 200 or not live_mask.any():
            continue
        scaler = StandardScaler()
        train_stack = scaler.fit_transform(historical_stack[stack_mask])
        test_stack = scaler.transform(live_stack[live_mask])
        recency = np.power(
            0.88,
            np.maximum(orders.max() - orders[stack_mask], 0),
        )
        fitted = Ridge(alpha=80.0)
        fitted.fit(
            train_stack,
            data.loc[stack_mask, "points"].to_numpy(float),
            sample_weight=recency,
        )
        stacked[live_mask] = fitted.predict(test_stack)

    return (
        {"means": means, "total": total, "stacked": stacked, "sigma": sigma},
        {
            "terminal": True,
            "causal": "Fit only on completed historical rows; scale and stack use out-of-fold historical route predictions.",
            "trainingRows": int(train_mask.sum()),
            "seedOffset": seed_offset,
            "routes": list(ROUTE_LABELS),
            "fits": fits,
        },
    )


def probability_metrics(data: pd.DataFrame, mean: np.ndarray, sigma: np.ndarray) -> dict:
    observed = data["fixture_count"].to_numpy(int) > 0
    evaluation = data["season"].isin(lens.EVALUATION_SEASONS).to_numpy()
    mask = observed & evaluation
    actual = data["points"].to_numpy(float)
    safe_sigma = np.maximum(sigma, 0.35)
    blank = ndtr((2.5 - mean) / safe_sigma)
    return5 = 1 - ndtr((4.5 - mean) / safe_sigma)
    haul8 = 1 - ndtr((7.5 - mean) / safe_sigma)
    return {
        "mae": round(float(np.mean(np.abs(mean[mask] - actual[mask]))), 4),
        "correlation": round(float(np.corrcoef(mean[mask], actual[mask])[0, 1]), 4),
        "blankBrier": round(float(np.mean((blank[mask] - (actual[mask] <= 2)) ** 2)), 4),
        "return5Brier": round(float(np.mean((return5[mask] - (actual[mask] >= 5)) ** 2)), 4),
        "haul8Brier": round(float(np.mean((haul8[mask] - (actual[mask] >= 8)) ** 2)), 4),
        "interval80Coverage": round(
            float(
                np.mean(
                    np.abs(actual[mask] - mean[mask])
                    <= 1.2816 * safe_sigma[mask]
                )
            ),
            4,
        ),
    }


def main() -> None:
    original, _ = lens.load_or_build_prepared_history()
    data = add_targets(original.reset_index(drop=True))
    scores, champion_plan, captain = champion_forecasts(data)
    predictions, audit = causal_route_predictions(data, scores)
    mapped_raw = quantile_map(data, predictions["total"], scores)
    mapped_stacked = quantile_map(data, predictions["stacked"], scores)
    observed = data["fixture_count"].to_numpy(int) > 0
    evaluation = data["season"].isin(lens.EVALUATION_SEASONS).to_numpy()
    frontier = selectable_frontier(data)
    actual = data["points"].to_numpy(float)
    decision_mask = observed & evaluation & frontier

    variants: dict[str, np.ndarray] = {"champion": scores}
    for share in (0.025, 0.05, 0.10, 0.15):
        variants[f"routes{int(share * 1000):03d}"] = (
            (1 - share) * scores + share * mapped_raw
        )
        variants[f"stacked{int(share * 1000):03d}"] = (
            (1 - share) * scores + share * mapped_stacked
        )

    seasons = list(dict.fromkeys(data["season"].tolist()))
    rows = []
    for name, challenger in variants.items():
        ratio = np.divide(challenger, scores, out=np.ones_like(scores), where=scores > 0.20)
        plan = champion_plan * np.clip(ratio, 0.80, 1.20)
        print(f"Recursive probabilistic challenger {name}", flush=True)
        totals, stats = lens.simulate_candidate(
            data,
            challenger,
            STRATEGY,
            plan_scores=plan,
            captain_scores=captain,
            tracked_player_name="Salah",
        )
        development = totals[2:6]
        holdout = totals[6:]
        rows.append(
            {
                "name": name,
                "developmentStability": round(float(development.mean() - 0.25 * development.std()), 3),
                "holdoutAverage": round(float(holdout.mean()), 1),
                "summary": variant_summary(totals, stats, seasons),
            }
        )

    champion_row = rows[0]
    best_challenger = max(rows[1:], key=lambda row: row["developmentStability"])
    selected = max(rows, key=lambda row: row["developmentStability"])
    paired = [
        {
            "season": old["season"],
            "champion": old["points"],
            "challenger": new["points"],
            "delta": new["points"] - old["points"],
        }
        for old, new in zip(champion_row["summary"]["seasons"], best_challenger["summary"]["seasons"])
    ]
    result = {
        "status": "causal probabilistic component challenger",
        "selectionRule": "Blend selected on 2018/19-2021/22 stability; 2022/23-2025/26 is untouched holdout.",
        "audit": audit,
        "probabilityMetrics": {
            "routeRaw": probability_metrics(data, predictions["total"], predictions["sigma"]),
            "routeStacked": probability_metrics(data, predictions["stacked"], predictions["sigma"]),
            "championMae": round(float(np.mean(np.abs(scores[observed & evaluation] - actual[observed & evaluation]))), 4),
        },
        "decisionMetrics": {
            "champion": decision_metrics(data, champion_plan, actual, decision_mask),
            "routeRaw": decision_metrics(data, mapped_raw, actual, decision_mask),
            "routeStacked": decision_metrics(data, mapped_stacked, actual, decision_mask),
        },
        "selected": selected,
        "bestChallenger": best_challenger,
        "pairedBestChallengerVsChampion": paired,
        "robustPromotion": bool(
            best_challenger["developmentStability"] > champion_row["developmentStability"]
            and best_challenger["summary"]["average"] > champion_row["summary"]["average"]
            and best_challenger["holdoutAverage"] >= champion_row["holdoutAverage"]
            and sum(row["delta"] > 0 for row in paired) >= 5
        ),
        "experiments": rows,
    }
    output = lens.ROOT / "analysis" / "data" / "probabilistic_component_challenger.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "probabilityMetrics": result["probabilityMetrics"],
                "decisionMetrics": result["decisionMetrics"],
                "selected": selected["name"],
                "bestChallenger": best_challenger["name"],
                "robustPromotion": result["robustPromotion"],
                "paired": paired,
                "experiments": [
                    {
                        "name": row["name"],
                        "developmentStability": row["developmentStability"],
                        "average": row["summary"]["average"],
                        "minimum": row["summary"]["minimum"],
                        "holdoutAverage": row["holdoutAverage"],
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
