"""Seed and model-consensus audit for the fixture captain challenger."""

from __future__ import annotations

import json

import numpy as np

import calibrate_model as lens
from captain_fixture_history_validation import (
    add_fixture_history,
    causal_predictions,
    decision_evaluation,
    percentile,
)
from frontier_ranker_validation import STRATEGY
from multiscale_horizon_validation import add_targets
from premium_captain_validation import captain_variants
from wildcard_freehit_ablation import champion_forecasts


def main() -> None:
    original, _ = lens.load_or_build_prepared_history()
    data = add_fixture_history(add_targets(original.reset_index(drop=True)))
    immediate, plan, captain = champion_forecasts(data)
    frozen = captain_variants(data, immediate, captain)["frozen"]
    context = causal_predictions(data, immediate, False)
    history = causal_predictions(data, immediate, True)
    context_ranks = np.column_stack(
        [percentile(data, context[:, seed]) for seed in range(context.shape[1])]
    )
    history_ranks = np.column_stack(
        [percentile(data, history[:, seed]) for seed in range(history.shape[1])]
    )
    variants = {}
    for seed in range(context.shape[1]):
        variants[f"context20Seed{seed}"] = 0.80 * frozen + 0.20 * context_ranks[:, seed]
        variants[f"history15Seed{seed}"] = 0.85 * frozen + 0.15 * history_ranks[:, seed]
        variants[f"history20Seed{seed}"] = 0.80 * frozen + 0.20 * history_ranks[:, seed]
    context_mean = context_ranks.mean(axis=1)
    history_mean = history_ranks.mean(axis=1)
    context_median = np.median(context_ranks, axis=1)
    history_median = np.median(history_ranks, axis=1)
    variants.update(
        {
            "context20Mean": 0.80 * frozen + 0.20 * context_mean,
            "context20Median": 0.80 * frozen + 0.20 * context_median,
            "history15Mean": 0.85 * frozen + 0.15 * history_mean,
            "history20Mean": 0.80 * frozen + 0.20 * history_mean,
            "history20Median": 0.80 * frozen + 0.20 * history_median,
            "crossModel15Mean": 0.85 * frozen
            + 0.15 * (0.50 * context_mean + 0.50 * history_mean),
            "crossModel20Mean": 0.80 * frozen
            + 0.20 * (0.50 * context_mean + 0.50 * history_mean),
            "crossModel20Conservative": 0.80 * frozen
            + 0.20 * np.minimum(context_mean, history_mean),
        }
    )
    base_totals, stats = lens.simulate_candidate(
        data,
        immediate,
        STRATEGY,
        plan_scores=plan,
        captain_scores=captain,
        audit_selections=True,
    )
    _, rows = decision_evaluation(
        data, stats, base_totals, frozen, {"frozen": frozen, **variants}
    )
    for row in rows:
        print(row["name"], row["averageDelta"], row["seasonDeltas"], flush=True)
    seed_rows = [row for row in rows if "Seed" in row["name"]]
    ensemble_rows = [row for row in rows if "Seed" not in row["name"] and row["name"] != "frozen"]
    stable = [
        row
        for row in ensemble_rows
        if row["developmentDelta"] > 0
        and row["holdoutDelta"] >= 0
        and row["worstSeasonDelta"] >= -8
        and row["declinedSeasons"] <= 2
    ]
    selected = max(
        stable,
        key=lambda row: (
            row["averageDelta"] - 0.20 * abs(row["worstSeasonDelta"]),
            -row["oracleRegretPerWeek"],
        ),
        default=None,
    )
    result = {
        "status": "recursive finalists identified" if selected else "research-only; seed/consensus gate failed",
        "method": "Three independently seeded checkpoint rankers, mean/median and cross-model captain rank consensus; fixed squad/XI decision evaluation.",
        "seedVariants": seed_rows,
        "ensembleVariants": ensemble_rows,
        "stable": [row["name"] for row in stable],
        "selected": selected["name"] if selected else None,
        "productionPromotion": False,
    }
    output = lens.ROOT / "analysis" / "data" / "captain_fixture_seed_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected": result["selected"], "stable": result["stable"]}, indent=2))
    print(f"Wrote {output.relative_to(lens.ROOT)}")


if __name__ == "__main__":
    main()
