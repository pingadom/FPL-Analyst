"""Audit horizon forecasts on affordable decisions, not global player metrics."""

from __future__ import annotations

import json

import numpy as np

import calibrate_model as lens
from bench_efficiency_validation import championship_forecasts
from frontier_ranker_validation import selectable_frontier
from multiscale_horizon_validation import (
    adaptive_value,
    add_targets,
    causal_online_ridge_horizons,
    causal_ridge_horizons,
    structural_horizons,
)


def decision_metrics(
    data,
    forecast: np.ndarray,
    actual: np.ndarray,
    mask: np.ndarray,
) -> dict:
    pair_correct = 0.0
    pair_total = 0.0
    cap_regrets: list[float] = []
    top3_regrets: list[float] = []
    for _, group_indices in data.groupby(
        ["season", "GW", "position_id"], sort=False
    ).groups.items():
        indices = np.asarray(group_indices, dtype=int)
        indices = indices[mask[indices]]
        if len(indices) < 5:
            continue
        prices = data.loc[indices, "price"].to_numpy(int)
        # A real transfer normally compares affordable near-substitutes.  Count
        # only pairs no more than £1.5m apart and ignore outcome ties.
        for left in range(len(indices)):
            comparable = np.flatnonzero(
                (np.abs(prices - prices[left]) <= 15)
                & (np.arange(len(indices)) > left)
            )
            for right in comparable:
                actual_delta = actual[indices[left]] - actual[indices[right]]
                if abs(actual_delta) < 0.25:
                    continue
                forecast_delta = forecast[indices[left]] - forecast[indices[right]]
                pair_correct += float(np.sign(actual_delta) == np.sign(forecast_delta))
                pair_total += 1.0

        minimum = int(np.ceil(prices.min() / 10) * 10)
        maximum = int(np.floor(prices.max() / 10) * 10)
        for cap in range(minimum, maximum + 1, 10):
            affordable = indices[prices <= cap]
            if len(affordable) < 5:
                continue
            forecast_order = affordable[
                np.argsort(forecast[affordable], kind="stable")
            ]
            actual_order = affordable[np.argsort(actual[affordable], kind="stable")]
            cap_regrets.append(
                float(actual[actual_order[-1]] - actual[forecast_order[-1]])
            )
            count = min(3, len(affordable))
            top3_regrets.append(
                float(
                    actual[actual_order[-count:]].sum()
                    - actual[forecast_order[-count:]].sum()
                )
            )
    return {
        "nearPricePairAccuracy": round(pair_correct / max(1, pair_total), 4),
        "pairs": int(pair_total),
        "meanAffordableTop1Regret": round(float(np.mean(cap_regrets)), 4),
        "p90AffordableTop1Regret": round(float(np.quantile(cap_regrets, 0.90)), 4),
        "meanAffordableTop3Regret": round(float(np.mean(top3_regrets)), 4),
        "budgetComparisons": len(cap_regrets),
    }


def main() -> None:
    original, _ = lens.load_or_build_prepared_history()
    data = add_targets(original.reset_index(drop=True))
    scores, baseline_plan, _ = championship_forecasts(data)
    structural = structural_horizons(data, scores)
    learned, _ = causal_ridge_horizons(data, structural)
    online, _ = causal_online_ridge_horizons(data, structural, learned)
    actual_horizons = {
        horizon: data[f"target_h{horizon}"].to_numpy(float)
        for horizon in (1, 3, 6, 10)
    }
    actual = adaptive_value(data, actual_horizons, 0.0)
    forecasts = {
        "oldFixedPlan": baseline_plan,
        "adaptiveStructural": adaptive_value(data, structural, 3.0),
        "adaptivePriorSeasonRidge": adaptive_value(data, learned, 3.0),
        "adaptiveOnlineRidge": adaptive_value(data, online, 3.0),
    }
    frontier = selectable_frontier(data)
    observed = data["fixture_count"].to_numpy(int) > 0
    evaluation = data["season"].isin(lens.EVALUATION_SEASONS).to_numpy(bool)
    base_mask = frontier & observed & evaluation
    result = {
        "status": "decision-frontier diagnostic",
        "method": (
            "Forecasts are judged on near-price pairs and the best affordable "
            "player under repeated price caps. This approximates the alternatives "
            "the legal squad optimiser can actually exchange."
        ),
        "overall": {
            name: decision_metrics(data, forecast, actual, base_mask)
            for name, forecast in forecasts.items()
        },
        "byPosition": {
            lens.POSITION_LABELS[position]: {
                name: decision_metrics(
                    data,
                    forecast,
                    actual,
                    base_mask & (data["position_id"].to_numpy(int) == position),
                )
                for name, forecast in forecasts.items()
            }
            for position in lens.SQUAD_QUOTAS
        },
        "bySeason": {
            season.replace("-", "/"): {
                name: decision_metrics(
                    data,
                    forecast,
                    actual,
                    base_mask & (data["season"].to_numpy() == season),
                )
                for name, forecast in forecasts.items()
            }
            for season in lens.EVALUATION_SEASONS
        },
    }
    output = lens.ROOT / "analysis" / "data" / "feasible_decision_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["overall"], indent=2))
    print(
        json.dumps(
            {
                position: {
                    name: metrics["meanAffordableTop1Regret"]
                    for name, metrics in rows.items()
                }
                for position, rows in result["byPosition"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
