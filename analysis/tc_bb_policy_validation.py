"""Training-selected deployable TC/Bench Boost threshold policy."""

from __future__ import annotations

import json

import numpy as np

import calibrate_model as lens
from bench_efficiency_validation import championship_forecasts
from frontier_ranker_validation import STRATEGY


THRESHOLDS = ((11, 15), (14, 18), (16, 21), (18, 24))


def main() -> None:
    data, _ = lens.load_or_build_prepared_history()
    data = data.reset_index(drop=True)
    scores, plan_scores, captain_scores = championship_forecasts(data)
    training_count = len(lens.TRAINING_SEASONS)
    baseline, _ = lens.simulate_candidate(
        data,
        scores,
        STRATEGY,
        plan_scores=plan_scores,
        captain_scores=captain_scores,
    )
    rows = []
    for index, (bench, triple) in enumerate(THRESHOLDS, start=1):
        policy = lens.ChipPolicy(
            1e6,
            1e6,
            bench,
            triple,
            0.0,
            10,
            28,
            ("Bench Boost", "Triple Captain"),
        )
        print(f"Running {index}/{len(THRESHOLDS)}: BB{bench}/TC{triple}", flush=True)
        totals, stats = lens.simulate_candidate(
            data,
            scores,
            STRATEGY,
            chip_policy=policy,
            plan_scores=plan_scores,
            captain_scores=captain_scores,
        )
        gain = totals - baseline
        training = gain[:training_count]
        evaluation = gain[training_count:]
        rows.append(
            {
                "benchThreshold": bench,
                "tripleThreshold": triple,
                "trainingStabilityGain": round(
                    float(training.mean() - 0.25 * training.std()), 3
                ),
                "evaluationAverageGain": round(float(evaluation.mean()), 1),
                "evaluationMinimumGain": int(round(float(evaluation.min()))),
                "evaluationMaximumGain": int(round(float(evaluation.max()))),
                "seasonGain": [round(float(value)) for value in gain],
                "evaluationChips": [row["chips"] for row in stats[training_count:]],
            }
        )
    selected = max(rows, key=lambda row: row["trainingStabilityGain"])
    result = {
        "status": "training-selected; chips are structurally restricted to confirmed doubles",
        "baselineAverage": round(float(baseline[training_count:].mean()), 1),
        "selected": selected,
        "selectedAverage": round(
            float(baseline[training_count:].mean())
            + selected["evaluationAverageGain"],
            1,
        ),
        "experiments": rows,
    }
    output = lens.ROOT / "analysis" / "data" / "tc_bb_policy_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
