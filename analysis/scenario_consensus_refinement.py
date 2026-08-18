"""Soft scenario-consensus refinements after hard vetoes proved too inert."""

from __future__ import annotations

import json

import numpy as np

import calibrate_model as lens
from bench_efficiency_validation import variant_summary
from frontier_ranker_validation import STRATEGY
from multiscale_horizon_validation import add_targets
from scenario_transfer_validation import (
    CorrelatedScenarioGenerator,
    ScenarioTransferPolicy,
    development_stability,
)
from wildcard_freehit_ablation import champion_forecasts


CONFIGS = (
    ("tail005", 0.005, 0.0),
    ("tail010", 0.010, 0.0),
    ("tail020", 0.020, 0.0),
    ("tail040", 0.040, 0.0),
    ("consensus025", 0.0, 0.25),
    ("consensus050", 0.0, 0.50),
    ("consensus100", 0.0, 1.00),
    ("tail010-consensus050", 0.010, 0.50),
)


def main() -> None:
    original, _ = lens.load_or_build_prepared_history()
    data = add_targets(original.reset_index(drop=True))
    immediate, plan, captain = champion_forecasts(data)
    seasons = list(dict.fromkeys(data["season"].tolist()))
    generator = CorrelatedScenarioGenerator(data, immediate, plan)
    base_totals, base_stats = lens.simulate_candidate(
        data, immediate, STRATEGY, plan_scores=plan, captain_scores=captain,
        tracked_player_name="Salah",
    )
    baseline = variant_summary(base_totals, base_stats, seasons)
    rows = []
    for name, tail_share, clear_penalty in CONFIGS:
        print(f"Soft scenario consensus {name}", flush=True)
        policy = ScenarioTransferPolicy(
            data, plan, generator, clear_threshold=0.0, tail_share=tail_share,
            clear_penalty=clear_penalty,
        )
        totals, stats = lens.simulate_candidate(
            data, immediate, STRATEGY, plan_scores=plan, captain_scores=captain,
            tracked_player_name="Salah", package_action_adjustment=policy,
        )
        summary = variant_summary(totals, stats, seasons)
        deltas = [new["points"] - old["points"] for new, old in zip(summary["seasons"], baseline["seasons"])]
        rows.append({
            "name": name,
            "tailShare": tail_share,
            "clearPenalty": clear_penalty,
            "developmentStability": round(development_stability(totals), 3),
            "holdoutAverage": round(float(totals[8:].mean()), 1),
            "summary": summary,
            "averageDelta": round(summary["average"] - baseline["average"], 1),
            "minimumDelta": summary["minimum"] - baseline["minimum"],
            "improvedSeasons": int(sum(delta > 0 for delta in deltas)),
            "unchangedSeasons": int(sum(delta == 0 for delta in deltas)),
            "worseSeasons": int(sum(delta < 0 for delta in deltas)),
            "seasonDeltas": deltas,
        })
        print(name, rows[-1]["averageDelta"], deltas, flush=True)
    selected = max(rows, key=lambda row: row["developmentStability"])
    baseline_stability = development_stability(base_totals)
    baseline_holdout = float(base_totals[8:].mean())
    robust = bool(
        selected["developmentStability"] > baseline_stability
        and selected["holdoutAverage"] >= baseline_holdout + 5
        and selected["minimumDelta"] >= 0
        and selected["improvedSeasons"] >= 5
    )
    result = {
        "status": "promoted" if robust else "research-only; robust promotion gate failed",
        "method": "Soft lower-tail and scenario-consensus penalties; no package is hard-vetoed and the callback remains default-off.",
        "baselineDevelopmentStability": round(baseline_stability, 3),
        "baselineHoldoutAverage": round(baseline_holdout, 1),
        "baseline": baseline,
        "variants": rows,
        "selectedByDevelopment": selected,
        "robustPromotion": robust,
    }
    output = lens.ROOT / "analysis" / "data" / "scenario_consensus_refinement.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "selected": selected["name"],
        "average": selected["summary"]["average"],
        "minimum": selected["summary"]["minimum"],
        "holdout": selected["holdoutAverage"],
        "robustPromotion": robust,
    }, indent=2))


if __name__ == "__main__":
    main()
