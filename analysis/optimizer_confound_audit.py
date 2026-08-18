"""Measure the exact-solver confound in the first bench-efficiency audit."""

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
    print("Running exact solver without spend or bench rules", flush=True)
    exact_strategy = replace(
        STRATEGY,
        name="Exact solver, no allocation rules",
        exact_initial_optimiser=True,
    )
    totals, stats = lens.simulate_candidate(
        data,
        scores,
        exact_strategy,
        plan_scores=plan_scores,
        captain_scores=captain_scores,
        tracked_player_name="Salah",
    )
    exact = variant_summary(totals, stats, seasons)
    first = json.loads(
        (lens.ROOT / "analysis" / "data" / "bench_efficiency_validation.json").read_text(
            encoding="utf-8"
        )
    )
    control = first["before"]
    hard = first["after"]
    result = {
        "status": "research-only; audit correction",
        "finding": "The first hard-rule audit also changed the opening-squad algorithm. This comparison separates the exact-solver effect from the constraints.",
        "heuristicControl": control,
        "exactNoRules": exact,
        "exactHardRules": hard,
        "decomposition": {
            "solverDelta": round(exact["average"] - control["average"], 1),
            "rulesConditionalOnExactSolver": round(
                hard["average"] - exact["average"], 1
            ),
            "combinedDelta": round(hard["average"] - control["average"], 1),
        },
    }
    output = lens.ROOT / "analysis" / "data" / "optimizer_confound_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"exact": exact, "decomposition": result["decomposition"]}, indent=2))


if __name__ == "__main__":
    main()
