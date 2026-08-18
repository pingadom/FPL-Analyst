"""Leak-free scoring-route consensus for fine-margin captain decisions.

The captain challenger is deliberately conservative.  It keeps 85% of the
frozen armband rank, adds 15% of a causal scoring-route distribution rank and
applies only a half-percentile defender tie-break.  Route models use prior
seasons only and exclude same-fixture event fields such as raw xG/xA.
"""

from __future__ import annotations

import json

import numpy as np

import calibrate_model as lens
from captain_fixture_history_validation import (
    add_fixture_history,
    decision_evaluation,
)
from frontier_ranker_validation import STRATEGY
from multiscale_horizon_validation import add_targets
from premium_captain_validation import captain_variants, percentile
from probabilistic_component_challenger import (
    FEATURES as ROUTE_FEATURES,
    causal_route_predictions,
)
from probabilistic_minutes_validation import season_summary
from wildcard_freehit_ablation import champion_forecasts


SEED_OFFSETS = (0, 1, 2, 3, 4)
SELECTED_SIGMA = 0.30
SELECTED_SHARE = 0.15
DEFENDER_TIE_BREAK = 0.005
MINIMUM_COMPLETED_SEASONS = 3


def weekly_percentile(data, values: np.ndarray) -> np.ndarray:
    return percentile(data, values, ["season", "GW"])


def captain_metric(
    frozen: np.ndarray,
    route_rank: np.ndarray,
    positions: np.ndarray,
    active: np.ndarray,
    share: float = SELECTED_SHARE,
    defender_tie_break: float = DEFENDER_TIE_BREAK,
) -> np.ndarray:
    # Goalkeepers receive one tenth of the defender tie-break.  This is enough
    # to resolve exact/near ties without banning a genuine double-GW standout.
    position_adjustment = defender_tie_break * (positions == 2)
    position_adjustment += 0.10 * defender_tie_break * (positions == 1)
    challenger = (1 - share) * frozen + share * route_rank - position_adjustment
    return np.where(active, challenger, frozen)


def selected_consensus_metric(
    data,
    immediate: np.ndarray,
    frozen_captain: np.ndarray,
) -> np.ndarray:
    """Build the selected leak-free five-seed captain metric for reuse."""
    frozen = captain_variants(data, immediate, frozen_captain)["frozen"]
    ranks = []
    for seed_offset in SEED_OFFSETS:
        component, _ = causal_route_predictions(
            data, immediate, seed_offset=seed_offset
        )
        ranks.append(
            weekly_percentile(
                data,
                component["stacked"] + SELECTED_SIGMA * component["sigma"],
            )
        )
    active = data["season_order"].to_numpy(int) >= MINIMUM_COMPLETED_SEASONS
    return captain_metric(
        frozen,
        np.column_stack(ranks).mean(axis=1),
        data["position_id"].to_numpy(int),
        active,
    )


def evaluate_recursive(
    data,
    immediate: np.ndarray,
    plan: np.ndarray,
    frozen_metric: np.ndarray,
    challenger_metric: np.ndarray,
    seasons: list[str],
) -> dict:
    rows = {}
    for label, metric, chip_policy in (
        ("frozenNoChips", frozen_metric, None),
        ("routeCaptainNoChips", challenger_metric, None),
        ("frozenAuditedChips", frozen_metric, lens.AUDITED_CHAMPION_CHIP_POLICY),
        (
            "routeCaptainAuditedChips",
            challenger_metric,
            lens.AUDITED_CHAMPION_CHIP_POLICY,
        ),
    ):
        totals, _ = lens.simulate_candidate(
            data,
            immediate,
            STRATEGY,
            chip_policy=chip_policy,
            plan_scores=plan,
            captain_scores=metric,
        )
        rows[label] = {**season_summary(totals, seasons), "totals": totals}

    comparisons = {}
    for label, new_key, old_key in (
        ("noChips", "routeCaptainNoChips", "frozenNoChips"),
        (
            "auditedChips",
            "routeCaptainAuditedChips",
            "frozenAuditedChips",
        ),
    ):
        delta = rows[new_key]["totals"][2:] - rows[old_key]["totals"][2:]
        comparisons[label] = {
            "averageDelta": round(float(delta.mean()), 1),
            "minimumDelta": int(delta.min()),
            "positiveSeasons": int((delta > 0).sum()),
            "negativeSeasons": int((delta < 0).sum()),
            "seasonDeltas": delta.astype(int).tolist(),
        }
    for row in rows.values():
        del row["totals"]
    return {"models": rows, "comparisons": comparisons}


def main() -> None:
    forbidden = {"expected_goals", "expected_assists", "expected_goals_conceded"}
    leaked = sorted(forbidden.intersection(ROUTE_FEATURES))
    if leaked:
        raise RuntimeError(f"Post-match route features are forbidden: {leaked}")

    original, _ = lens.load_or_build_prepared_history()
    data = add_fixture_history(add_targets(original.reset_index(drop=True)))
    immediate, plan, captain = champion_forecasts(data)
    frozen = captain_variants(data, immediate, captain)["frozen"]
    positions = data["position_id"].to_numpy(int)
    active = data["season_order"].to_numpy(int) >= MINIMUM_COMPLETED_SEASONS
    seasons = list(dict.fromkeys(data["season"].tolist()))
    base_totals, stats = lens.simulate_candidate(
        data,
        immediate,
        STRATEGY,
        plan_scores=plan,
        captain_scores=captain,
        audit_selections=True,
    )

    route_components = []
    route_ranks = []
    for seed_offset in SEED_OFFSETS:
        component, _ = causal_route_predictions(
            data, immediate, seed_offset=seed_offset
        )
        route_components.append(component)
        route_ranks.append(
            weekly_percentile(
                data,
                component["stacked"] + SELECTED_SIGMA * component["sigma"],
            )
        )
    rank_matrix = np.column_stack(route_ranks)

    variants = {"frozen": frozen}
    for seed_index, seed_offset in enumerate(SEED_OFFSETS):
        variants[f"routeSeed{seed_offset}"] = captain_metric(
            frozen, rank_matrix[:, seed_index], positions, active
        )
    variants["routeMean"] = captain_metric(
        frozen, rank_matrix.mean(axis=1), positions, active
    )
    variants["routeMedian"] = captain_metric(
        frozen, np.median(rank_matrix, axis=1), positions, active
    )

    # A small local neighbourhood guards against a single tuned coefficient.
    for sigma in (0.20, 0.30, 0.40):
        sigma_ranks = np.column_stack(
            [
                weekly_percentile(
                    data,
                    component["stacked"] + sigma * component["sigma"],
                )
                for component in route_components
            ]
        ).mean(axis=1)
        for share in (0.125, 0.15, 0.175):
            variants[f"neighbourZ{sigma:.3f}S{share:.3f}"] = captain_metric(
                frozen, sigma_ranks, positions, active, share=share
            )
    variants["routeMeanNoDefenderTieBreak"] = captain_metric(
        frozen,
        rank_matrix.mean(axis=1),
        positions,
        active,
        defender_tie_break=0.0,
    )

    by_name, rows = decision_evaluation(
        data, stats, base_totals, frozen, variants
    )
    selected = by_name["routeMean"]
    seed_rows = [by_name[f"routeSeed{seed}"] for seed in SEED_OFFSETS]
    neighbourhood = [
        row for row in rows if row["name"].startswith("neighbour")
    ]
    seed_average = np.asarray(
        [row["averageDelta"] for row in seed_rows], dtype=float
    )
    neighbour_average = np.asarray(
        [row["averageDelta"] for row in neighbourhood], dtype=float
    )
    stable = bool(
        selected["developmentDelta"] > 0
        and selected["holdoutDelta"] >= 0
        and selected["worstSeasonDelta"] >= -8
        and np.quantile(seed_average, 0.20) >= 0
        and np.quantile(neighbour_average, 0.20) >= 0
    )

    recursive = evaluate_recursive(
        data,
        immediate,
        plan,
        frozen,
        variants["routeMean"],
        seasons,
    )
    result = {
        "status": (
            "prospective captain shadow gate passed"
            if stable
            else "research-only; captain stability gate failed"
        ),
        "method": (
            "Five-seed prior-season scoring-route consensus: 85% frozen rank, "
            "15% expected-route-plus-0.30-sigma rank, and a 0.005 defender "
            "tie-break, activated after three complete training seasons. Raw "
            "same-fixture xG/xA/xGA are excluded."
        ),
        "informationBoundary": {
            "forbiddenPostMatchFeatures": sorted(forbidden),
            "forbiddenFeaturesPresent": leaked,
            "routeFeatureCount": len(ROUTE_FEATURES),
            "minimumCompletedSeasons": MINIMUM_COMPLETED_SEASONS,
        },
        "selected": selected,
        "seedVariants": seed_rows,
        "ensembleVariants": [by_name["routeMean"], by_name["routeMedian"]],
        "neighbourhood": neighbourhood,
        "tieBreakAblation": by_name["routeMeanNoDefenderTieBreak"],
        "stability": {
            "seedP20AverageDelta": round(float(np.quantile(seed_average, 0.20)), 1),
            "neighbourP20AverageDelta": round(
                float(np.quantile(neighbour_average, 0.20)), 1
            ),
            "passed": stable,
        },
        "recursive": recursive,
        "fixtureHistoryFinding": (
            "Direct opponent-history ranks remain excluded from the selected "
            "policy because their causal standalone and blended gates failed."
        ),
        "productionPromotion": False,
    }
    output = lens.ROOT / "analysis" / "data" / "captain_route_consensus_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected": {
                    key: selected[key]
                    for key in (
                        "averageDelta",
                        "developmentDelta",
                        "holdoutDelta",
                        "worstSeasonDelta",
                        "seasonDeltas",
                    )
                },
                "stability": result["stability"],
                "recursive": recursive["comparisons"],
            },
            indent=2,
        )
    )
    print(f"Wrote {output.relative_to(lens.ROOT)}")


if __name__ == "__main__":
    main()
