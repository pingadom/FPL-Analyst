"""Causal adaptive transfer patience when the current squad becomes stale."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

import calibrate_model as lens
from bench_efficiency_validation import championship_forecasts, variant_summary
from frontier_ranker_validation import STRATEGY


CONFIGS = (
    ("baseline", None, 0.0, 0.0),
    ("stale80-soft", 80.0, 4.0, 0.10),
    ("stale100-medium", 100.0, 8.0, 0.15),
    ("stale120-medium", 120.0, 8.0, 0.15),
    ("stale140-strong", 140.0, 12.0, 0.25),
)


def main() -> None:
    data, _ = lens.load_or_build_prepared_history()
    data = data.reset_index(drop=True)
    scores, plan, captain = championship_forecasts(data)
    seasons = list(dict.fromkeys(data["season"].tolist()))
    rows = []
    raw_results = {}
    for name, trigger, hurdle_reduction, hold_reduction in CONFIGS:
        strategy = replace(
            STRATEGY,
            name=name,
            staleness_gap_trigger=trigger,
            staleness_hurdle_reduction=hurdle_reduction,
            staleness_hold_reduction=hold_reduction,
        )
        print(f"Running {name}", flush=True)
        totals, stats = lens.simulate_candidate(
            data,
            scores,
            strategy,
            plan_scores=plan,
            captain_scores=captain,
            tracked_player_name="Salah",
        )
        training = totals[: len(lens.TRAINING_SEASONS)]
        row = {
            "name": name,
            "trigger": trigger,
            "hurdleReduction": hurdle_reduction,
            "holdReduction": hold_reduction,
            "trainingStability": round(
                float(training.mean() - 0.25 * training.std()), 3
            ),
            "staleness": [stat["staleness"] for stat in stats],
            "summary": variant_summary(totals, stats, seasons),
        }
        rows.append(row)
        raw_results[name] = (totals, stats, strategy)
    selected = max(rows, key=lambda row: row["trainingStability"])
    selected_totals, _, selected_strategy = raw_results[selected["name"]]
    print(f"Running selected with audited chips: {selected['name']}", flush=True)
    chip_totals, chip_stats = lens.simulate_candidate(
        data,
        scores,
        selected_strategy,
        chip_policy=lens.AUDITED_CHAMPION_CHIP_POLICY,
        plan_scores=plan,
        captain_scores=captain,
        tracked_player_name="Salah",
    )
    result = {
        "status": "training-selected causal staleness challenger",
        "method": (
            "At each deadline compare the current 15's decision utility with a "
            "fresh legal squad affordable at current selling value. Relax transfer "
            "patience only when that model-implied gap clears a fixed threshold."
        ),
        "selected": selected,
        "selectedWithAuditedChips": variant_summary(chip_totals, chip_stats, seasons),
        "experiments": rows,
    }
    output = lens.ROOT / "analysis" / "data" / "staleness_policy_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "selected": selected["name"],
                "selectedAverage": selected["summary"]["average"],
                "selectedWithChips": result["selectedWithAuditedChips"]["average"],
                "experiments": [
                    {
                        "name": row["name"],
                        "trainingStability": row["trainingStability"],
                        "average": row["summary"]["average"],
                        "minimum": row["summary"]["minimum"],
                        "evaluationTriggers": row["staleness"][2:],
                    }
                    for row in rows
                ],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
