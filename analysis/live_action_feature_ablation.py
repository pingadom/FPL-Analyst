"""Locate the information required by the late near-price action ranker.

The full historical challenger is not automatically live-deployable: its
matrix contains two learned horizon forecasts and ten learned scoring-route
forecasts.  This audit removes those families in stages, always retaining the
same causal checkpoint fitting and recursive squad simulation.  A variant is
eligible for the prospective shadow only when its inputs have a deadline-live
equivalent and its paired recursive result clears the declared gate.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import calibrate_model as lens
from bench_efficiency_validation import variant_summary
from decision_focused_horizon_validation import CHECKPOINTS, target_and_maturity
from frontier_ranker_validation import FEATURES as BASE_FEATURES
from frontier_ranker_validation import STRATEGY, selectable_frontier
from listwise_ranker_validation import quantile_map
from multiscale_horizon_validation import FEATURES as MULTISCALE_FEATURES
from multiscale_horizon_validation import add_targets
from multiscale_phase_validation import event_number
from probabilistic_component_challenger import causal_route_predictions
from transfer_action_ranker_validation import PRICE_BAND, SHIFTS, model, query_order
from wildcard_freehit_ablation import champion_forecasts


CACHE_VERSION = 2
TERMINAL_FIT_HORIZONS = {
    "component_horizon_censored",
    "causal_horizon_ridge",
}
LIVE_EXTENDED_FEATURES = [
    feature for feature in MULTISCALE_FEATURES if feature not in TERMINAL_FIT_HORIZONS
]
VARIANTS = {
    "base18": (BASE_FEATURES, False),
    "base18Routes": (BASE_FEATURES, True),
    "liveExtended": (LIVE_EXTENDED_FEATURES, False),
    "liveExtendedRoutes": (LIVE_EXTENDED_FEATURES, True),
    "componentHorizonRoutes": (
        LIVE_EXTENDED_FEATURES + ["component_horizon_censored"],
        True,
    ),
    "causalHorizonRoutes": (
        LIVE_EXTENDED_FEATURES + ["causal_horizon_ridge"],
        True,
    ),
    "fullRoutes": (MULTISCALE_FEATURES, True),
}


def named_matrix(
    frame: pd.DataFrame,
    features: list[str],
    medians: pd.Series | None = None,
) -> tuple[np.ndarray, pd.Series]:
    """Apply the frozen multiscale normalisation to an explicit subset."""
    values = frame[features].replace([np.inf, -np.inf], np.nan).astype(float).copy()
    divisors = {
        "price": 150.0,
        "horizon_weighted_games_censored": 6.0,
        "expected_minutes": 90.0,
        "minutes_std": 40.0,
        "GW": 38.0,
    }
    for column, divisor in divisors.items():
        if column in values:
            values[column] /= divisor
    if "selected" in values:
        values["selected"] = np.log1p(values["selected"].clip(lower=0)) / 16.0
    if "observations" in values:
        values["observations"] = np.log1p(values["observations"].clip(lower=0)) / 6.0
    if medians is None:
        medians = values.median().fillna(0)
    return values.fillna(medians).to_numpy(np.float32), medians


def route_additions(
    frame: pd.DataFrame,
    champion_plan: np.ndarray,
    component: dict,
    medians: pd.Series | None = None,
) -> tuple[np.ndarray, pd.Series]:
    indices = frame.index.to_numpy(int)
    means = component["means"][indices]
    values = pd.DataFrame(
        np.column_stack(
            [
                champion_plan[indices] / 35.0,
                component["total"][indices] / 8.0,
                component["stacked"][indices] / 8.0,
                component["sigma"][indices] / 6.0,
                means[:, 0] / 2.0,
                means[:, 1] / 2.0,
                means[:, 2] / 0.5,
                means[:, 3] / 0.5,
                means[:, 4] / 1.2,
                means[:, 5] / 5.0,
                means[:, 6] / 4.0,
                means[:, 7] / 2.0,
                means[:, 8] / 2.0,
                means[:, 9] / 2.0,
            ]
        ),
        index=frame.index,
        columns=[
            "championPlan",
            "routeTotal",
            "routeStacked",
            "routeSigma",
            "routeAppearance",
            "routeSixty",
            "routeGoals",
            "routeAssists",
            "routeClean",
            "routeSaves",
            "routeConceded",
            "routeBonus",
            "routeDefensiveContribution",
            "routeOther",
        ],
    ).replace([np.inf, -np.inf], np.nan)
    if medians is None:
        medians = values.median().fillna(0)
    return values.fillna(medians).to_numpy(np.float32), medians


def action_matrix(
    frame: pd.DataFrame,
    features: list[str],
    include_routes: bool,
    champion_plan: np.ndarray,
    component: dict,
    medians: tuple[pd.Series, pd.Series | None] | None = None,
) -> tuple[np.ndarray, tuple[pd.Series, pd.Series | None]]:
    base, base_medians = named_matrix(
        frame, features, medians[0] if medians is not None else None
    )
    if not include_routes:
        return base, (base_medians, None)
    additions, addition_medians = route_additions(
        frame,
        champion_plan,
        component,
        medians[1] if medians is not None else None,
    )
    return np.column_stack([base, additions]), (base_medians, addition_medians)


def causal_predictions(
    name: str,
    data: pd.DataFrame,
    features: list[str],
    include_routes: bool,
    champion_plan: np.ndarray,
    component: dict,
) -> np.ndarray:
    path = lens.CACHE / f"live-action-ablation-{name}-v{CACHE_VERSION}.npz"
    if path.exists():
        cached = np.load(path)
        if len(cached["shifts"]) == len(data):
            return cached["shifts"]

    target, maturity = target_and_maturity(data)
    output = np.tile(champion_plan[:, None], (1, len(SHIFTS)))
    seasons = list(dict.fromkeys(data["season"].tolist()))
    orders = data["season_order"].to_numpy(int)
    gws = data["GW"].to_numpy(int)
    positions = data["position_id"].to_numpy(int)
    observed = data["fixture_count"].to_numpy(int) > 0
    frontier = selectable_frontier(data)
    for season_order in range(1, len(seasons)):
        season_mask = orders == season_order
        for checkpoint_index, checkpoint in enumerate(CHECKPOINTS):
            interval_end = (
                CHECKPOINTS[checkpoint_index + 1] - 1
                if checkpoint_index + 1 < len(CHECKPOINTS)
                else 99
            )
            for position in lens.SQUAD_QUOTAS:
                position_mask = positions == position
                train_mask = (
                    position_mask
                    & observed
                    & frontier
                    & ((orders < season_order) | (season_mask & (maturity < checkpoint)))
                )
                test_mask = (
                    season_mask
                    & position_mask
                    & (gws >= checkpoint)
                    & (gws <= interval_end)
                )
                if not test_mask.any():
                    continue
                train = data.loc[train_mask]
                test = data.loc[test_mask]
                train_x, medians = action_matrix(
                    train, features, include_routes, champion_plan, component
                )
                test_x, _ = action_matrix(
                    test, features, include_routes, champion_plan, component, medians
                )
                for shift_index, shift in enumerate(SHIFTS):
                    order, qid = query_order(train, shift)
                    fitted = model(
                        910000
                        + 10000 * list(VARIANTS).index(name)
                        + 1000 * season_order
                        + 100 * checkpoint_index
                        + 10 * position
                        + shift_index
                    )
                    fitted.fit(train_x[order], target[train_mask][order], qid=qid)
                    output[test_mask, shift_index] = fitted.predict(test_x)
            print(f"{name}: {seasons[season_order]} GW{checkpoint}", flush=True)
    np.savez_compressed(
        path,
        shifts=output,
        features=json.dumps(features),
        include_routes=include_routes,
    )
    return output


def recursive_summary(
    data: pd.DataFrame,
    immediate: np.ndarray,
    champion_plan: np.ndarray,
    captain: np.ndarray,
    shifts: np.ndarray,
    seasons: list[str],
) -> tuple[dict, list[dict]]:
    mapped = np.column_stack(
        [quantile_map(data, shifts[:, index], champion_plan) for index in range(len(SHIFTS))]
    )
    delta = mapped - champion_plan[:, None]
    agreement = np.sign(delta[:, 0]) == np.sign(delta[:, 1])
    consensus = mapped.mean(axis=1)
    active = (
        (event_number(data) >= 25)
        & agreement
        & (data["position_id"].to_numpy(int) != 1)
    )
    plan = champion_plan + 0.05 * active * (consensus - champion_plan)
    totals, stats = lens.simulate_candidate(
        data,
        immediate,
        STRATEGY,
        plan_scores=plan,
        captain_scores=captain,
        tracked_player_name="Salah",
    )
    return variant_summary(totals, stats, seasons), totals.tolist()


def main() -> None:
    original, _ = lens.load_or_build_prepared_history()
    data = add_targets(original.reset_index(drop=True))
    immediate, champion_plan, captain = champion_forecasts(data)
    component, _ = causal_route_predictions(data, immediate)
    seasons = list(dict.fromkeys(data["season"].tolist()))
    base_totals, base_stats = lens.simulate_candidate(
        data,
        immediate,
        STRATEGY,
        plan_scores=champion_plan,
        captain_scores=captain,
        tracked_player_name="Salah",
    )
    baseline = variant_summary(base_totals, base_stats, seasons)
    rows = []
    for name, (features, include_routes) in VARIANTS.items():
        shifts = causal_predictions(
            name, data, features, include_routes, champion_plan, component
        )
        summary, totals = recursive_summary(
            data, immediate, champion_plan, captain, shifts, seasons
        )
        paired = [
            {
                "season": season,
                "before": int(before),
                "after": int(after),
                "delta": int(after - before),
            }
            for season, before, after in zip(seasons, base_totals, totals)
        ]
        development = np.asarray(totals[2:6], dtype=float)
        base_development = base_totals[2:6]
        holdout = np.asarray(totals[6:], dtype=float)
        base_holdout = base_totals[6:]
        shadow = bool(
            development.mean() - 0.25 * development.std()
            > base_development.mean() - 0.25 * base_development.std()
            and holdout.mean() >= base_holdout.mean()
            and summary["minimum"] >= baseline["minimum"]
        )
        rows.append(
            {
                "name": name,
                "featureCount": len(features) + (14 if include_routes else 0),
                "includeLearnedRoutes": include_routes,
                "deadlineLiveFeatureSet": name == "liveExtended",
                "summary": summary,
                "paired": paired,
                "developmentStability": round(
                    float(development.mean() - 0.25 * development.std()), 3
                ),
                "holdoutAverage": round(float(holdout.mean()), 1),
                "prospectiveShadowGate": shadow and name == "liveExtended",
            }
        )
        print(
            json.dumps(
                {
                    "name": name,
                    "average": summary["average"],
                    "minimum": summary["minimum"],
                    "holdout": rows[-1]["holdoutAverage"],
                    "deltas": [row["delta"] for row in paired],
                }
            ),
            flush=True,
        )
    result = {
        "status": "feature-family ablation; no automatic promotion",
        "method": "Causal GW13/GW25 near-price LambdaMART; recursive policy active GW25+, non-GK, 5% only when overlapping price bands agree.",
        "baseline": baseline,
        "terminalFitHorizonFeatures": sorted(TERMINAL_FIT_HORIZONS),
        "liveExtendedFeatures": LIVE_EXTENDED_FEATURES,
        "variants": rows,
        "productionPromotion": False,
    }
    output = lens.ROOT / "analysis" / "data" / "live_action_feature_ablation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output.relative_to(lens.ROOT)}")


if __name__ == "__main__":
    main()
