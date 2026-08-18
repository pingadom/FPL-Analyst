"""Paired recursive isolation of Wildcard and Free Hit policies."""

from __future__ import annotations

import json

import numpy as np

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


POLICIES = tuple(
    (
        f"wc{threshold}-{phase}",
        lens.ChipPolicy(
            threshold,
            1e6,
            1e6,
            1e6,
            0.55,
            minimum,
            second,
            ("Wildcard",),
        ),
    )
    for threshold in (45, 60, 75, 90)
    for phase, minimum, second in (("early", 6, 20), ("late", 10, 28))
) + tuple(
    (
        f"fh{threshold}",
        lens.ChipPolicy(
            1e6,
            threshold,
            1e6,
            1e6,
            0.0,
            10,
            28,
            ("Free Hit",),
        ),
    )
    for threshold in (5, 10, 15, 20, 30)
)


def champion_forecasts(data):
    scores, baseline_plan, captain = championship_forecasts(data)
    structural = structural_horizons(data, scores)
    structural_adaptive = adaptive_value(data, structural, 3.0)
    learned, _ = causal_ridge_horizons(data, structural)
    online, _ = causal_online_ridge_horizons(data, structural, learned)
    ridge = quantile_map(
        data, adaptive_value(data, online, 3.0), baseline_plan
    )
    direct_raw, _ = causal_online_prediction(data, structural_adaptive)
    direct = quantile_map(data, direct_raw, baseline_plan)
    ensemble = 0.50 * ridge + 0.50 * direct
    active = 0.15 * (event_number(data) >= 13)
    plan = baseline_plan + active * (ensemble - baseline_plan)
    return scores, plan, captain


def main() -> None:
    original, _ = lens.load_or_build_prepared_history()
    data = add_targets(original.reset_index(drop=True))
    scores, plan_scores, captain_scores = champion_forecasts(data)
    seasons = list(dict.fromkeys(data["season"].tolist()))
    training_count = len(lens.TRAINING_SEASONS)
    print("Running no-chip paired baseline", flush=True)
    baseline, _ = lens.simulate_candidate(
        data,
        scores,
        STRATEGY,
        plan_scores=plan_scores,
        captain_scores=captain_scores,
    )
    fresh = lens.precompute_fresh_squads(data, plan_scores)
    free_hits = lens.precompute_fresh_squads(data, scores)
    rows = []
    for index, (name, policy) in enumerate(POLICIES, start=1):
        print(f"Running {index}/{len(POLICIES)}: {name}", flush=True)
        totals, stats = lens.simulate_candidate(
            data,
            scores,
            STRATEGY,
            chip_policy=policy,
            fresh_squads=fresh,
            free_hit_squads=free_hits,
            plan_scores=plan_scores,
            captain_scores=captain_scores,
        )
        gain = totals - baseline
        training = gain[:training_count]
        evaluation = gain[training_count:]
        rows.append(
            {
                "name": name,
                "chip": policy.enabled_chips[0],
                "policy": policy.as_dict(),
                "trainingStabilityGain": round(
                    float(training.mean() - 0.25 * training.std()), 3
                ),
                "evaluationAverageGain": round(float(evaluation.mean()), 1),
                "evaluationMinimumGain": int(round(float(evaluation.min()))),
                "evaluationMaximumGain": int(round(float(evaluation.max()))),
                "seasonGain": [round(float(value)) for value in gain],
                "evaluationUsage": [
                    [entry["chip"] for entry in row["chips"]]
                    for row in stats[training_count:]
                ],
            }
        )
    selected = {}
    for chip in ("Wildcard", "Free Hit"):
        options = [row for row in rows if row["chip"] == chip]
        selected[chip] = max(options, key=lambda row: row["trainingStabilityGain"])
    result = {
        "status": "training-selected paired recursive chip ablation on frozen multi-timescale champion",
        "warning": (
            "Every candidate is a full recursive rerun. Threshold selection uses "
            "only the calibration seasons; evaluation seasons are untouched. "
            "Future blank/double assignments remain censored when they were not "
            "known at the historical deadline."
        ),
        "baselineEvaluationAverage": round(float(baseline[training_count:].mean()), 1),
        "selected": selected,
        "experiments": rows,
    }
    output = lens.ROOT / "analysis" / "data" / "wildcard_freehit_ablation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "baseline": result["baselineEvaluationAverage"],
                "selected": {
                    chip: {
                        "name": row["name"],
                        "trainingStabilityGain": row["trainingStabilityGain"],
                        "evaluationAverageGain": row["evaluationAverageGain"],
                        "minimumGain": row["evaluationMinimumGain"],
                    }
                    for chip, row in selected.items()
                },
                "all": [
                    {
                        "name": row["name"],
                        "training": row["trainingStabilityGain"],
                        "evaluation": row["evaluationAverageGain"],
                        "minimum": row["evaluationMinimumGain"],
                    }
                    for row in rows
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
