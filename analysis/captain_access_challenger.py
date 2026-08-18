"""Test whether squad/transfer decisions preserve access to the captain model."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

import calibrate_model as lens
from bench_efficiency_validation import championship_forecasts, variant_summary
from frontier_ranker_validation import STRATEGY


CONFIGS = (
    {"name": "captain-access-070", "captainWeight": 0.70},
    {"name": "captain-access-100", "captainWeight": 1.00},
    {"name": "captain-access-130", "captainWeight": 1.30},
    {
        "name": "captain-access-100-diverse16",
        "captainWeight": 1.00,
        "diverse": True,
    },
)


def main() -> None:
    data, _ = lens.load_or_build_prepared_history()
    data = data.reset_index(drop=True)
    scores, plan_scores, captain_scores = championship_forecasts(data)
    seasons = list(dict.fromkeys(data["season"].tolist()))
    training_count = len(lens.TRAINING_SEASONS)

    print("Running frozen captain-unaligned baseline", flush=True)
    baseline_totals, baseline_stats = lens.simulate_candidate(
        data,
        scores,
        STRATEGY,
        plan_scores=plan_scores,
        captain_scores=captain_scores,
        tracked_player_name="Salah",
    )
    baseline = variant_summary(baseline_totals, baseline_stats, seasons)
    rows = []
    for index, config in enumerate(CONFIGS, start=1):
        diverse = bool(config.get("diverse", False))
        strategy = replace(
            STRATEGY,
            name=str(config["name"]),
            align_captain_objective=True,
            squad_captain_weight=float(config["captainWeight"]),
            expand_transfer_frontier=diverse,
            transfer_candidate_limit=16 if diverse else 10,
            transfer_beam_width=10,
        )
        print(f"Running {index}/{len(CONFIGS)}: {strategy.name}", flush=True)
        totals, stats = lens.simulate_candidate(
            data,
            scores,
            strategy,
            plan_scores=plan_scores,
            captain_scores=captain_scores,
            tracked_player_name="Salah",
        )
        training = totals[:training_count]
        rows.append(
            {
                **config,
                "trainingStability": round(
                    float(training.mean() - 0.25 * training.std()), 3
                ),
                "summary": variant_summary(totals, stats, seasons),
            }
        )
    selected = max(rows, key=lambda row: row["trainingStability"])
    result = {
        "status": "training-selected causal challenger",
        "baseline": baseline,
        "selected": selected,
        "lift": {
            "average": round(selected["summary"]["average"] - baseline["average"], 1),
            "minimum": selected["summary"]["minimum"] - baseline["minimum"],
            "total": selected["summary"]["total"] - baseline["total"],
        },
        "experiments": rows,
    }
    output = lens.ROOT / "analysis" / "data" / "captain_access_challenger.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "baselineAverage": baseline["average"],
                "selected": {
                    "name": selected["name"],
                    "trainingStability": selected["trainingStability"],
                    "evaluationAverage": selected["summary"]["average"],
                    "minimum": selected["summary"]["minimum"],
                },
                "lift": result["lift"],
                "all": [
                    {
                        "name": row["name"],
                        "trainingStability": row["trainingStability"],
                        "evaluationAverage": row["summary"]["average"],
                        "minimum": row["summary"]["minimum"],
                    }
                    for row in rows
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
