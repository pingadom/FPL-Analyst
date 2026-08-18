"""Freeze and audit the strongest decision-focused multi-timescale challenger."""

from __future__ import annotations

import json

import numpy as np

import calibrate_model as lens
from bench_efficiency_validation import championship_forecasts
from decision_focused_horizon_validation import causal_online_prediction
from frontier_ranker_validation import STRATEGY
from hybrid_decision_validation import summarize
from listwise_ranker_validation import quantile_map
from multiscale_horizon_validation import (
    adaptive_value,
    add_targets,
    causal_online_ridge_horizons,
    causal_ridge_horizons,
    structural_horizons,
)
from multiscale_phase_validation import event_number


def main() -> None:
    original, _ = lens.load_or_build_prepared_history()
    data = add_targets(original.reset_index(drop=True))
    scores, baseline_plan, captain = championship_forecasts(data)
    structural = structural_horizons(data, scores)
    structural_adaptive = adaptive_value(data, structural, 3.0)
    learned, _ = causal_ridge_horizons(data, structural)
    online_ridge, ridge_audit = causal_online_ridge_horizons(
        data, structural, learned
    )
    ridge_plan = quantile_map(
        data, adaptive_value(data, online_ridge, 3.0), baseline_plan
    )
    direct_raw, direct_audit = causal_online_prediction(data, structural_adaptive)
    direct_plan = quantile_map(data, direct_raw, baseline_plan)
    decision_ensemble = 0.50 * ridge_plan + 0.50 * direct_plan
    active_share = 0.15 * (event_number(data) >= 13)
    champion_plan = baseline_plan + active_share * (
        decision_ensemble - baseline_plan
    )

    print("Running frozen decision-focused champion with audited chips", flush=True)
    totals, stats = lens.simulate_candidate(
        data,
        scores,
        STRATEGY,
        chip_policy=lens.AUDITED_CHAMPION_CHIP_POLICY,
        plan_scores=champion_plan,
        captain_scores=captain,
        tracked_player_name="Salah",
    )
    seasons = list(dict.fromkeys(data["season"].tolist()))
    benchmarks = json.loads(
        (lens.ROOT / "analysis" / "data" / "historical_rank_benchmarks.json").read_text(
            encoding="utf-8"
        )
    )
    targets = {
        row["season"].replace("/", "-"): int(row["points"])
        for row in benchmarks["seasons"]
    }
    with_chips = summarize(totals, seasons, targets, stats)

    policy_results = json.loads(
        (
            lens.ROOT
            / "analysis"
            / "data"
            / "decision_focused_horizon_validation.json"
        ).read_text(encoding="utf-8")
    )
    no_chip = next(
        row["summary"]
        for row in policy_results["experiments"]
        if row["name"] == "ridgeDirectEnsemble15"
    )
    no_chip_margins = [
        int(row["points"] - targets[row["season"].replace("/", "-")])
        for row in no_chip["seasons"]
    ]
    no_chip = {
        **no_chip,
        "targetHits": int(sum(margin >= 0 for margin in no_chip_margins)),
        "averageMargin": round(float(np.mean(no_chip_margins)), 1),
    }
    old_stack = json.loads(
        (
            lens.ROOT
            / "analysis"
            / "data"
            / "championship_stack_validation.json"
        ).read_text(encoding="utf-8")
    )["variants"]["hybridCaptain50AuditedChips"]
    no_chip_delta = np.asarray(
        [41, 42, 23, 68, -18, 27, -12, 32], dtype=float
    )
    rng = np.random.default_rng(260813)
    bootstrap = rng.choice(no_chip_delta, size=(200_000, len(no_chip_delta)), replace=True)
    bootstrap_means = bootstrap.mean(axis=1)
    result = {
        "status": "frozen historical research champion; prospective shadow required",
        "architecture": {
            "GW1to12": "Established championship planner",
            "GW13to38": (
                "85% established planner + 15% equal blend of causal multi-output "
                "ridge and direct nonlinear holding-value models"
            ),
            "horizons": [1, 3, 6, 10],
            "holdingPeriod": "Player-specific 2-10 GWs, capped by season end",
            "exitCost": "Up to 3 planning points for expected sub-six-GW churn",
            "updates": [13, 25],
            "chips": lens.AUDITED_CHAMPION_CHIP_POLICY.as_dict(),
        },
        "causalAudit": {
            "ridgeFits": ridge_audit,
            "directFits": direct_audit,
            "rule": (
                "Every season prediction uses earlier seasons plus only same-season "
                "targets whose complete player-specific horizon ended before the "
                "checkpoint."
            ),
        },
        "selectionExposureWarning": (
            "The architecture is walk-forward causal, but the 15% blend was chosen "
            "after inspecting historical evaluation results. Its historical score "
            "is not an unbiased estimate of future rank."
        ),
        "noChips": no_chip,
        "withAuditedChips": with_chips,
        "oldChampionWithAuditedChips": old_stack,
        "lift": {
            "noChipAverageVsOldNoChip": round(no_chip["average"] - 2149.5, 1),
            "chipAverageVsOldChip": round(
                with_chips["average"] - old_stack["average"], 1
            ),
            "noChipSeasonDeltas": no_chip_delta.astype(int).tolist(),
            "improvedSeasons": int((no_chip_delta > 0).sum()),
            "worseSeasons": int((no_chip_delta < 0).sum()),
            "seasonBootstrap95": [
                round(float(np.quantile(bootstrap_means, 0.025)), 1),
                round(float(np.quantile(bootstrap_means, 0.975)), 1),
            ],
            "bootstrapProbabilityPositive": round(
                float((bootstrap_means > 0).mean()), 4
            ),
        },
    }
    output = (
        lens.ROOT
        / "analysis"
        / "data"
        / "decision_focused_champion_audit.json"
    )
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "noChips": {
                    key: no_chip[key]
                    for key in ("average", "minimum", "targetHits", "averageMargin")
                },
                "withAuditedChips": {
                    key: with_chips[key]
                    for key in ("average", "minimum", "targetHits", "averageMargin")
                },
                "oldChampionWithAuditedChips": {
                    key: old_stack[key]
                    for key in ("average", "minimum", "targetHits", "averageMargin")
                },
                "lift": result["lift"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
