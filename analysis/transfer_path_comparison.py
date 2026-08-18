"""Compare transfer paths where a small bench penalty caused large regressions."""

from __future__ import annotations

import json
from dataclasses import replace

import calibrate_model as lens
from bench_efficiency_validation import championship_forecasts
from frontier_ranker_validation import STRATEGY


def main() -> None:
    data, _ = lens.load_or_build_prepared_history()
    data = data.reset_index(drop=True)
    scores, plan_scores, captain_scores = championship_forecasts(data)
    mask = data["season"].isin(["2018-19", "2024-25", "2025-26"]).to_numpy(bool)
    subset = data.loc[mask].reset_index(drop=True)
    subset_scores = scores[mask]
    subset_plan = plan_scores[mask]
    subset_captain = captain_scores[mask]
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
        totals, stats = lens.simulate_candidate(
            subset,
            subset_scores,
            replace(STRATEGY, name=name, **changes),
            plan_scores=subset_plan,
            captain_scores=subset_captain,
        )
        models[name] = {
            stat["season"]: {
                "points": round(float(totals[index])),
                "transfers": stat["transferLog"],
                "weeklyPoints": stat["weeklyPoints"],
            }
            for index, stat in enumerate(stats)
        }
    result = {
        "status": "research-only; diagnostic",
        "models": models,
    }
    output = lens.ROOT / "analysis" / "data" / "transfer_path_comparison.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for season in models["exactNoRules"]:
        base = models["exactNoRules"][season]
        hard = models["exactHardRules"][season]
        print(
            season,
            base["points"],
            hard["points"],
            "\nBASE",
            json.dumps(base["transfers"], ensure_ascii=False),
            "\nHARD",
            json.dumps(hard["transfers"], ensure_ascii=False),
            flush=True,
        )


if __name__ == "__main__":
    main()
