"""Measure the value of multi-GW schedule information without promoting leakage."""

from __future__ import annotations

import json

import numpy as np
from scipy.stats import spearmanr

import calibrate_model as lens
from bench_efficiency_validation import championship_forecasts, variant_summary
from frontier_ranker_validation import STRATEGY
from listwise_ranker_validation import quantile_map
from multiscale_horizon_validation import (
    adaptive_value,
    add_targets,
    structural_horizons,
)
from multiscale_phase_validation import event_number


def main() -> None:
    original, _ = lens.load_or_build_prepared_history()
    data = add_targets(original.reset_index(drop=True))
    scores, baseline_plan, captain = championship_forecasts(data)
    censored = structural_horizons(data, scores)
    optimistic_data = data.copy()
    optimistic_data["component_horizon_censored"] = optimistic_data[
        "component_horizon"
    ]
    uncensored = structural_horizons(optimistic_data, scores)
    censored_value = adaptive_value(data, censored, 3.0)
    uncensored_value = adaptive_value(data, uncensored, 3.0)
    uncensored_plan = quantile_map(data, uncensored_value, baseline_plan)
    events = event_number(data)
    configs = {
        "baselineCensored": baseline_plan,
        "finalSchedule10AllSeasonDiagnostic": (
            0.90 * baseline_plan + 0.10 * uncensored_plan
        ),
        "finalSchedule10AfterGW13Diagnostic": (
            baseline_plan
            + 0.10 * (events >= 13) * (uncensored_plan - baseline_plan)
        ),
        "finalSchedule25AllSeasonDiagnostic": (
            0.75 * baseline_plan + 0.25 * uncensored_plan
        ),
    }
    seasons = list(dict.fromkeys(data["season"].tolist()))
    rows = []
    for name, plan in configs.items():
        print(f"Running {name}", flush=True)
        totals, stats = lens.simulate_candidate(
            data,
            scores,
            STRATEGY,
            plan_scores=plan,
            captain_scores=captain,
            tracked_player_name="Salah",
        )
        rows.append(
            {
                "name": name,
                "diagnosticOnly": name != "baselineCensored",
                "summary": variant_summary(totals, stats, seasons),
            }
        )

    targets = {
        horizon: data[f"target_h{horizon}"].to_numpy(float)
        for horizon in (1, 3, 6, 10)
    }
    actual = adaptive_value(data, targets, 0.0)
    evaluation = data["season"].isin(lens.EVALUATION_SEASONS).to_numpy(bool)
    observed = data["fixture_count"].to_numpy(int) > 0
    mask = evaluation & observed
    result = {
        "status": "information bound only; finalized schedule is not causal",
        "warning": (
            "The uncensored rows use the final historical GW assignment. They "
            "therefore know later postponements and rearrangements and can never "
            "be promoted. The experiment only estimates whether archived deadline "
            "fixture snapshots would be valuable."
        ),
        "rankCorrelation": {
            "censored": round(float(spearmanr(censored_value[mask], actual[mask]).statistic), 4),
            "finalSchedule": round(
                float(spearmanr(uncensored_value[mask], actual[mask]).statistic), 4
            ),
        },
        "experiments": rows,
    }
    output = lens.ROOT / "analysis" / "data" / "schedule_information_bound.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "rankCorrelation": result["rankCorrelation"],
                "experiments": [
                    {
                        "name": row["name"],
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
