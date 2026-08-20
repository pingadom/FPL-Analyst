"""OpenFPL-inspired causal, position-specific player ensemble.

The benchmark combines four genuinely different views of a deadline:

* the stable structural champion;
* the causal role ridge;
* a selectable-frontier tree;
* a scoring-route distribution model.

Weights are fitted on within-deadline percentiles using earlier seasons only,
separately by FPL position.  The final values are quantile-mapped to the stable
forecast scale so the experiment changes ordering, not the optimizer's units.
This is an independent implementation inspired by the OpenFPL design, not a
claim of reproducing its private pipeline or data exactly.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import calibrate_model as lens
from captain_fixture_history_validation import add_fixture_history
from frontier_ranker_validation import (
    causal_predictions as frontier_predictions,
    frontier_metrics,
    quantile_map,
    selectable_frontier,
)
from multiscale_horizon_validation import add_targets
from probabilistic_component_challenger import (
    FEATURES as ROUTE_FEATURES,
    causal_route_predictions,
)
from wildcard_freehit_ablation import champion_forecasts


MODEL_NAMES = ("structural", "role", "frontier", "routes")
FORBIDDEN_POST_MATCH_FEATURES = frozenset(
    {"expected_goals", "expected_assists", "expected_goals_conceded"}
)
MINIMUM_TRAINING_SEASONS = 3


def grouped_percentile(data: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    result = np.zeros(len(data), dtype=float)
    series = pd.Series(np.asarray(values, float), index=data.index)
    for _, indices in data.groupby(["season", "GW", "position_id"], sort=False).groups.items():
        local = series.loc[indices]
        result[np.asarray(indices, int)] = local.rank(method="average", pct=True).to_numpy(float)
    return result


def actual_percentile(data: pd.DataFrame) -> np.ndarray:
    return grouped_percentile(data, data["points"].to_numpy(float))


def _normalised_positive_coefficients(model: Ridge) -> np.ndarray:
    coefficient = np.maximum(np.asarray(model.coef_, float), 0.0)
    if coefficient.sum() <= 1e-8:
        return np.full(len(MODEL_NAMES), 1.0 / len(MODEL_NAMES))
    return coefficient / coefficient.sum()


def causal_ensemble(
    data: pd.DataFrame,
    structural: np.ndarray,
    frontier: np.ndarray,
    routes: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    role = data["role_ridge_xpts"].to_numpy(float)
    raw_models = np.column_stack([structural, role, frontier, routes])
    rank_models = np.column_stack(
        [grouped_percentile(data, raw_models[:, index]) for index in range(raw_models.shape[1])]
    )
    target = actual_percentile(data)
    orders = data["season_order"].to_numpy(int)
    positions = data["position_id"].to_numpy(int)
    observed = data["fixture_count"].to_numpy(int) > 0
    frontier_mask = selectable_frontier(data)
    ensemble_rank = rank_models[:, 0].copy()
    audit: list[dict] = []
    seasons = list(dict.fromkeys(data["season"].tolist()))
    for season_order in range(MINIMUM_TRAINING_SEASONS, len(seasons)):
        for position in lens.SQUAD_QUOTAS:
            train = (
                (orders < season_order)
                & observed
                & frontier_mask
                & (positions == position)
            )
            test = (orders == season_order) & (positions == position)
            fitted = Ridge(alpha=90.0, positive=True)
            age = season_order - orders[train]
            weights = np.power(0.88, np.maximum(age - 1, 0))
            fitted.fit(rank_models[train], target[train], sample_weight=weights)
            coefficients = _normalised_positive_coefficients(fitted)
            ensemble_rank[test] = rank_models[test] @ coefficients
            audit.append(
                {
                    "season": seasons[season_order],
                    "position": int(position),
                    "trainingRows": int(train.sum()),
                    "weights": {
                        name: round(float(value), 4)
                        for name, value in zip(MODEL_NAMES, coefficients)
                    },
                }
            )
    mapped = quantile_map(data, ensemble_rank, structural)
    mapped[~observed] = 0.0
    return mapped, audit


def build_ensemble(
    data: pd.DataFrame, structural: np.ndarray
) -> tuple[np.ndarray, dict]:
    leaked = sorted(FORBIDDEN_POST_MATCH_FEATURES.intersection(ROUTE_FEATURES))
    if leaked:
        raise RuntimeError(f"Post-match route features are forbidden: {leaked}")
    frontier, frontier_audit = frontier_predictions(data)
    route, route_audit = causal_route_predictions(data, structural)
    ensemble, weight_audit = causal_ensemble(
        data, structural, frontier, route["stacked"]
    )
    metrics = {
        "structural": vars(frontier_metrics(data, structural)),
        "frontier": vars(frontier_metrics(data, quantile_map(data, frontier, structural))),
        "routes": vars(frontier_metrics(data, quantile_map(data, route["stacked"], structural))),
        "positionEnsemble": vars(frontier_metrics(data, ensemble)),
    }
    return ensemble, {
        "status": "causal OpenFPL-inspired challenger",
        "method": (
            "Position-specific non-negative ridge over within-deadline ranks; "
            "each test season uses earlier seasons only and the output is "
            "quantile-mapped to the structural scale."
        ),
        "modelNames": list(MODEL_NAMES),
        "informationBoundary": {
            "forbiddenPostMatchFeatures": sorted(FORBIDDEN_POST_MATCH_FEATURES),
            "forbiddenFeaturesPresent": leaked,
        },
        "metrics": metrics,
        "weightAudit": weight_audit,
        "frontierFitCount": len(frontier_audit),
        "routeAudit": route_audit,
    }


def main() -> None:
    original, _ = lens.load_or_build_prepared_history()
    data = add_fixture_history(add_targets(original.reset_index(drop=True)))
    structural, _, _ = champion_forecasts(data)
    _, result = build_ensemble(data, structural)
    output = lens.ROOT / "analysis" / "data" / "openfpl_position_ensemble_v2.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
