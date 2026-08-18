"""Sensitivity analysis for the naturally defined GW25 non-GK action policy."""

from __future__ import annotations

import json

import numpy as np

import calibrate_model as lens
from action_segment_validation import consensus_parts, development_stability
from bench_efficiency_validation import variant_summary
from frontier_ranker_validation import STRATEGY
from multiscale_horizon_validation import add_targets
from multiscale_phase_validation import event_number
from wildcard_freehit_ablation import champion_forecasts


CONFIGS = (
    ("gw25-share025", 25, 0.025),
    ("gw25-share050", 25, 0.050),
    ("gw25-share075", 25, 0.075),
    ("gw25-share100", 25, 0.100),
    ("gw21-share050", 21, 0.050),
    ("gw29-share050", 29, 0.050),
)


def main() -> None:
    original, _ = lens.load_or_build_prepared_history()
    data = add_targets(original.reset_index(drop=True))
    immediate, champion_plan, captain = champion_forecasts(data)
    consensus, agreement, _ = consensus_parts(data, champion_plan)
    events = event_number(data)
    no_goalkeeper = data["position_id"].to_numpy(int) != 1
    seasons = list(dict.fromkeys(data["season"].tolist()))
    base_totals, base_stats = lens.simulate_candidate(
        data, immediate, STRATEGY, plan_scores=champion_plan, captain_scores=captain,
        tracked_player_name="Salah",
    )
    baseline = variant_summary(base_totals, base_stats, seasons)
    rows = []
    for name, start, share in CONFIGS:
        active = (events >= start) & agreement & no_goalkeeper
        plan = champion_plan + share * active.astype(float) * (consensus - champion_plan)
        print(f"Late action sensitivity {name}", flush=True)
        totals, stats = lens.simulate_candidate(
            data, immediate, STRATEGY, plan_scores=plan, captain_scores=captain,
            tracked_player_name="Salah",
        )
        summary = variant_summary(totals, stats, seasons)
        deltas = [new["points"] - old["points"] for new, old in zip(summary["seasons"], baseline["seasons"])]
        changed = [delta for delta in deltas if delta != 0]
        rows.append({
            "name": name,
            "startEvent": start,
            "share": share,
            "developmentStability": round(development_stability(totals), 3),
            "holdoutAverage": round(float(totals[8:].mean()), 1),
            "summary": summary,
            "averageDelta": round(summary["average"] - baseline["average"], 1),
            "minimumDelta": summary["minimum"] - baseline["minimum"],
            "positiveChangedRate": round(sum(delta > 0 for delta in changed) / len(changed), 3) if changed else 0,
            "improvedSeasons": int(sum(delta > 0 for delta in deltas)),
            "unchangedSeasons": int(sum(delta == 0 for delta in deltas)),
            "worseSeasons": int(sum(delta < 0 for delta in deltas)),
            "seasonDeltas": deltas,
        })
        print(name, rows[-1]["averageDelta"], deltas, flush=True)
    selected = max(rows, key=lambda row: row["developmentStability"])
    neighbourhood = [row for row in rows if row["startEvent"] == 25]
    sensitivity = {
        "positiveAverageSettings": int(sum(row["averageDelta"] > 0 for row in rows)),
        "positiveGW25Shares": int(sum(row["averageDelta"] > 0 for row in neighbourhood)),
        "gw25AverageDeltaRange": [
            min(row["averageDelta"] for row in neighbourhood),
            max(row["averageDelta"] for row in neighbourhood),
        ],
    }
    prospective_shadow = bool(
        selected["developmentStability"] > development_stability(base_totals)
        and selected["holdoutAverage"] >= float(base_totals[8:].mean()) + 5
        and selected["minimumDelta"] >= 0
        and selected["positiveChangedRate"] >= 2 / 3
        and sensitivity["positiveGW25Shares"] >= 3
    )
    result = {
        "status": "prospective shadow candidate" if prospective_shadow else "research-only; sensitivity gate failed",
        "method": "Predeclared local sensitivity around the GW25 checkpoint, non-goalkeeper two-band consensus and 5% blend.",
        "baseline": baseline,
        "variants": rows,
        "sensitivity": sensitivity,
        "selectedByDevelopment": selected,
        "prospectiveShadow": prospective_shadow,
        "productionPromotion": False,
    }
    output = lens.ROOT / "analysis" / "data" / "late_action_sensitivity.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "selected": selected["name"],
        "average": selected["summary"]["average"],
        "minimum": selected["summary"]["minimum"],
        "holdout": selected["holdoutAverage"],
        "sensitivity": sensitivity,
        "prospectiveShadow": prospective_shadow,
    }, indent=2))


if __name__ == "__main__":
    main()
