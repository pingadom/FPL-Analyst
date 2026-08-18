"""Training-only audit of diversified transfer candidates and beam width."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

import calibrate_model as lens
from bench_efficiency_validation import championship_forecasts, variant_summary
from frontier_ranker_validation import STRATEGY


CONFIGS = (
    (10, 10),
    (16, 10),
    (16, 20),
    (24, 20),
    (24, 32),
)


def main() -> None:
    data, _ = lens.load_or_build_prepared_history()
    data = data.reset_index(drop=True)
    scores, plan_scores, captain_scores = championship_forecasts(data)
    seasons = list(dict.fromkeys(data["season"].tolist()))

    print("Running frozen top-plan frontier", flush=True)
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
    training_count = len(lens.TRAINING_SEASONS)
    for index, (candidate_limit, beam_width) in enumerate(CONFIGS, start=1):
        strategy = replace(
            STRATEGY,
            name=f"diverse-frontier-{candidate_limit}-beam-{beam_width}",
            expand_transfer_frontier=True,
            transfer_candidate_limit=candidate_limit,
            transfer_beam_width=beam_width,
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
                "candidateLimit": candidate_limit,
                "beamWidth": beam_width,
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
    output = lens.ROOT / "analysis" / "data" / "transfer_frontier_challenger.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "baselineAverage": baseline["average"],
                "selected": {
                    "candidateLimit": selected["candidateLimit"],
                    "beamWidth": selected["beamWidth"],
                    "trainingStability": selected["trainingStability"],
                    "evaluationAverage": selected["summary"]["average"],
                },
                "lift": result["lift"],
                "all": [
                    {
                        "candidateLimit": row["candidateLimit"],
                        "beamWidth": row["beamWidth"],
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
