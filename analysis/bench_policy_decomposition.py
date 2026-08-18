"""Decompose hard spend/bench rules after their paired historical failure."""

from __future__ import annotations

import json
from dataclasses import replace

import calibrate_model as lens
from bench_efficiency_validation import championship_forecasts, variant_summary
from frontier_ranker_validation import STRATEGY


VARIANTS = {
    "softPenalty005": {
        "bench_premium_penalty": 0.005,
        "transfer_bench_premium_penalty": 0.005,
    },
    "softPenalty011": {
        "bench_premium_penalty": 0.011,
        "transfer_bench_premium_penalty": 0.011,
    },
    "spendFloorOnly": {
        "initial_spend_gap": 5,
    },
    "benchCapOnly": {
        "bench_premium_limit": 20,
        "bench_premium_penalty": 0.022,
        "transfer_bench_premium_penalty": 0.022,
    },
    "softFloorAndPenalty": {
        "initial_spend_gap": 15,
        "bench_premium_penalty": 0.005,
        "transfer_bench_premium_penalty": 0.005,
    },
}


def main() -> None:
    data, _ = lens.load_or_build_prepared_history()
    data = data.reset_index(drop=True)
    scores, plan_scores, captain_scores = championship_forecasts(data)
    seasons = list(dict.fromkeys(data["season"].tolist()))
    strict_audit = json.loads(
        (lens.ROOT / "analysis" / "data" / "bench_efficiency_validation.json").read_text(
            encoding="utf-8"
        )
    )
    control = strict_audit["before"]
    models = {}
    for name, changes in VARIANTS.items():
        print(f"Starting {name}", flush=True)
        strategy = replace(STRATEGY, name=name, **changes)
        totals, stats = lens.simulate_candidate(
            data,
            scores,
            strategy,
            plan_scores=plan_scores,
            captain_scores=captain_scores,
            tracked_player_name="Salah",
        )
        summary = variant_summary(totals, stats, seasons)
        summary["deltaVsControl"] = round(summary["average"] - control["average"], 1)
        models[name] = summary
        print(
            name,
            summary["average"],
            summary["deltaVsControl"],
            summary["salah"],
            flush=True,
        )
    best = max(models, key=lambda name: models[name]["average"])
    result = {
        "status": "research-only; historically exposed",
        "controlAverage": control["average"],
        "strictPolicyAverage": strict_audit["after"]["average"],
        "models": models,
        "bestVariant": best,
        "bestAverage": models[best]["average"],
        "bestDeltaVsControl": models[best]["deltaVsControl"],
        "decision": "Keep the historical control unless a softer rule beats it; recognisable squad structure is not sufficient evidence.",
    }
    output = lens.ROOT / "analysis" / "data" / "bench_policy_decomposition.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best": best, "models": {key: {"average": value["average"], "delta": value["deltaVsControl"], "salah": value["salah"]} for key, value in models.items()}}, indent=2))


if __name__ == "__main__":
    main()
