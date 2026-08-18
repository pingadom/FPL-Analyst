"""Faithful audit: apply allocation rules only to fresh squad construction."""

from __future__ import annotations

import json
from dataclasses import replace

import calibrate_model as lens
from bench_efficiency_validation import championship_forecasts, variant_summary
from frontier_ranker_validation import STRATEGY


def main() -> None:
    data, _ = lens.load_or_build_prepared_history()
    data = data.reset_index(drop=True)
    scores, plan_scores, captain_scores = championship_forecasts(data)
    seasons = list(dict.fromkeys(data["season"].tolist()))
    exact_strategy = replace(
        STRATEGY,
        name="Aligned exact solver, no allocation rules",
        exact_initial_optimiser=True,
        initial_spend_gap=None,
        bench_premium_limit=None,
        bench_premium_penalty=0.0,
        transfer_bench_premium_penalty=0.0,
    )
    strategy = replace(
        STRATEGY,
        name="Fresh-squad allocation only",
        exact_initial_optimiser=True,
        initial_spend_gap=5,
        bench_premium_limit=20,
        bench_premium_penalty=0.022,
        transfer_bench_premium_penalty=0.0,
    )
    print("Running aligned exact solver without allocation rules", flush=True)
    exact_totals, exact_stats = lens.simulate_candidate(
        data,
        scores,
        exact_strategy,
        plan_scores=plan_scores,
        captain_scores=captain_scores,
        tracked_player_name="Salah",
    )
    exact = variant_summary(exact_totals, exact_stats, seasons)
    print("Running aligned exact solver with fresh-squad allocation rules", flush=True)
    totals, stats = lens.simulate_candidate(
        data,
        scores,
        strategy,
        plan_scores=plan_scores,
        captain_scores=captain_scores,
        tracked_player_name="Salah",
    )
    faithful = variant_summary(totals, stats, seasons)
    control = json.loads(
        (lens.ROOT / "analysis" / "data" / "bench_efficiency_validation.json").read_text(
            encoding="utf-8"
        )
    )["before"]
    paired = [
        {
            "season": row["season"],
            "control": control["seasons"][index]["points"],
            "exactNoRules": exact["seasons"][index]["points"],
            "freshRules": row["points"],
            "freshVsExact": row["points"] - exact["seasons"][index]["points"],
            "initialSpend": row["initialSpend"],
            "initialBenchSpend": row["initialBenchSpend"],
        }
        for index, row in enumerate(faithful["seasons"])
    ]
    result = {
        "status": "research-only; corrected audit",
        "finding": "Allocation constraints apply only at fresh squad construction. The weekly transfer utility is unchanged.",
        "heuristicControlAverage": control["average"],
        "exactNoRulesAverage": exact["average"],
        "freshRulesAverage": faithful["average"],
        "freshRulesVsExact": round(faithful["average"] - exact["average"], 1),
        "freshRulesVsHeuristic": round(faithful["average"] - control["average"], 1),
        "faithful": faithful,
        "paired": paired,
    }
    output = lens.ROOT / "analysis" / "data" / "faithful_allocation_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("heuristicControlAverage", "exactNoRulesAverage", "freshRulesAverage", "freshRulesVsExact", "freshRulesVsHeuristic", "paired")}, indent=2))


if __name__ == "__main__":
    main()
