"""Refine the only positive challenger: a conservative captain ceiling blend."""

from __future__ import annotations

import json

import numpy as np

import calibrate_model as lens
from frontier_ranker_validation import STRATEGY
from multiscale_horizon_validation import add_targets
from premium_captain_validation import captain_variants
from probabilistic_minutes_validation import season_summary
from wildcard_freehit_ablation import champion_forecasts


def main() -> None:
    original, _ = lens.load_or_build_prepared_history()
    data = add_targets(original.reset_index(drop=True))
    immediate, plan, captain = champion_forecasts(data)
    components = captain_variants(data, immediate, captain)
    frozen = components["frozen"]
    ceiling = (components["ceiling25"] - 0.75 * frozen) / 0.25
    seasons = list(dict.fromkeys(data["season"].tolist()))
    base_totals, _ = lens.simulate_candidate(data, immediate, STRATEGY, plan_scores=plan, captain_scores=captain)
    baseline = season_summary(base_totals, seasons)
    rows = []
    for share in [0.05, 0.10, 0.15, 0.20, 0.25]:
        captain_score = (1 - share) * frozen + share * ceiling
        totals, _ = lens.simulate_candidate(data, immediate, STRATEGY, plan_scores=plan, captain_scores=captain_score)
        summary = season_summary(totals, seasons)
        deltas = [row["points"] - old["points"] for row, old in zip(summary["seasons"], baseline["seasons"])]
        rows.append({
            "name": f"ceiling-{share:.2f}", "share": share, **summary,
            "averageDelta": round(summary["average"] - baseline["average"], 1),
            "developmentDelta": round(summary["developmentAverage"] - baseline["developmentAverage"], 1),
            "holdoutDelta": round(summary["holdoutAverage"] - baseline["holdoutAverage"], 1),
            "improvedSeasons": int(sum(delta > 0 for delta in deltas)),
            "unchangedSeasons": int(sum(delta == 0 for delta in deltas)),
            "worstSeasonDelta": int(min(deltas)),
            "seasonDeltas": deltas,
        })
        print("captain refinement", share, rows[-1]["average"], deltas, flush=True)
    eligible = [row for row in rows if row["developmentDelta"] > 0 and row["holdoutDelta"] >= 5 and row["worstSeasonDelta"] >= 0 and row["improvedSeasons"] >= 5]
    selected = max(eligible, key=lambda row: (row["holdoutDelta"], row["developmentDelta"])) if eligible else None
    result = {
        "status": "promoted" if selected else "research-only; strict downside gate failed",
        "method": "Predeclared 5-25% blends between the frozen causal captain rank and an expected-points/haul/reliability ceiling rank; squad and transfer forecasts are unchanged.",
        "baseline": baseline,
        "variants": rows,
        "selected": selected,
    }
    output = lens.ROOT / "analysis" / "data" / "captain_consensus_refinement.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"baseline": baseline, "selected": selected}, indent=2))


if __name__ == "__main__":
    main()
