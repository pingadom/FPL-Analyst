"""Trace how a stronger opening squad becomes worse through recursive decisions."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

import calibrate_model as lens
from bench_efficiency_validation import championship_forecasts, variant_summary
from frontier_ranker_validation import STRATEGY


def first_negative_week(control, challenger):
    cumulative = 0.0
    for week, (base, alternative) in enumerate(zip(control, challenger), start=1):
        cumulative += float(alternative) - float(base)
        if cumulative < 0:
            return week
    return None


def main() -> None:
    data, _ = lens.load_or_build_prepared_history()
    data = data.reset_index(drop=True)
    scores, plan_scores, captain_scores = championship_forecasts(data)
    seasons = list(dict.fromkeys(data["season"].tolist()))
    championship = json.loads(
        (lens.ROOT / "analysis" / "data" / "championship_stack_validation.json").read_text(
            encoding="utf-8"
        )
    )["variants"]["hybridCaptain50"]
    control_by_season = {
        row["season"]: row for row in championship["evaluation"]
    }
    models = {}
    for name, changes in {
        "exactNoRules": {"exact_initial_optimiser": True},
        "exactHardRules": {
            "exact_initial_optimiser": True,
            "initial_spend_gap": 5,
            "bench_premium_limit": 20,
            "bench_premium_penalty": 0.022,
            "transfer_bench_premium_penalty": 0.022,
        },
    }.items():
        print(f"Running {name}", flush=True)
        strategy = replace(STRATEGY, name=name, **changes)
        totals, stats = lens.simulate_candidate(
            data,
            scores,
            strategy,
            plan_scores=plan_scores,
            captain_scores=captain_scores,
            tracked_player_name="Salah",
        )
        models[name] = variant_summary(totals, stats, seasons)

    comparisons = []
    for exact_row, hard_row in zip(
        models["exactNoRules"]["seasons"], models["exactHardRules"]["seasons"]
    ):
        control = control_by_season[exact_row["season"]]
        control_weekly = control.get("weeklyPoints", [])
        exact_weekly = exact_row["weeklyPoints"]
        hard_weekly = hard_row["weeklyPoints"]
        comparisons.append(
            {
                "season": exact_row["season"],
                "controlPoints": control["points"],
                "exactPoints": exact_row["points"],
                "hardPoints": hard_row["points"],
                "exactDelta": exact_row["points"] - control["points"],
                "hardVsExactDelta": hard_row["points"] - exact_row["points"],
                "controlTransfers": control["transfers"],
                "exactTransfers": exact_row["transfers"],
                "hardTransfers": hard_row["transfers"],
                "exactAverageBank": exact_row["averageBank"],
                "hardAverageBank": hard_row["averageBank"],
                "firstNegativeExactWeek": first_negative_week(
                    control_weekly, exact_weekly
                )
                if control_weekly
                else None,
                "firstNegativeHardVsExactWeek": first_negative_week(
                    exact_weekly, hard_weekly
                ),
                "exactFirstSix": round(float(np.sum(exact_weekly[:6])), 1),
                "hardFirstSix": round(float(np.sum(hard_weekly[:6])), 1),
            }
        )
    result = {
        "status": "research-only; diagnostic",
        "finding": "Opening allocation and recursive transfer-path effects are reported separately.",
        "models": models,
        "comparisons": comparisons,
    }
    output = lens.ROOT / "analysis" / "data" / "recursive_path_diagnostic.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
