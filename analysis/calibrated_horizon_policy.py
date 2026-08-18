"""Retune only the transfer hurdle after causal horizon-scale calibration."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

import calibrate_model as lens
from captain_ranker_validation import rank_blend
from forecast_calibration_challenger import causal_isotonic
from frontier_ranker_validation import PLAYER_CANDIDATE, STRATEGY
from listwise_ranker_validation import quantile_map


HURDLES = (8.0, 10.0, 12.0, 14.0, 16.0)


def main() -> None:
    data, _ = lens.load_or_build_prepared_history()
    data = data.reset_index(drop=True)
    immediate, horizon, _ = lens.candidate_forecasts(
        data, PLAYER_CANDIDATE, robust_planning=False, schedule_censored=True
    )
    stable_plan = 0.75 * immediate * 4.5 + 0.25 * horizon
    frontier_raw = np.load(lens.CACHE / "frontier-causal-predictions-v2.npz")[
        "prediction"
    ]
    horizon_raw = np.load(lens.CACHE / "listwise-horizon_target-v1.npz")[
        "prediction"
    ]
    captain_raw = np.load(lens.CACHE / "captain-listwise-v1.npz")["prediction"]
    score = 0.75 * immediate + 0.25 * quantile_map(data, frontier_raw, immediate)
    plan = 0.75 * stable_plan + 0.25 * quantile_map(data, horizon_raw, stable_plan)
    calibrated_plan = causal_isotonic(data, plan, "horizon_target")
    captain = rank_blend(data, immediate, captain_raw, 0.50)
    seasons = list(dict.fromkeys(data["season"].tolist()))
    training_count = len(lens.TRAINING_SEASONS)
    rows = []
    for index, hurdle in enumerate(HURDLES, start=1):
        strategy = replace(
            STRATEGY,
            name=f"calibrated-horizon-h{hurdle:.0f}",
            transfer_hurdle=hurdle,
        )
        print(f"Running {index}/{len(HURDLES)}: {strategy.name}", flush=True)
        totals, stats = lens.simulate_candidate(
            data,
            score,
            strategy,
            plan_scores=calibrated_plan,
            captain_scores=captain,
            tracked_player_name="Salah",
        )
        training = totals[:training_count]
        evaluation = totals[training_count:]
        rows.append(
            {
                "hurdle": hurdle,
                "trainingStability": round(
                    float(training.mean() - 0.25 * training.std()), 3
                ),
                "evaluationAverage": round(float(evaluation.mean()), 1),
                "evaluationMinimum": int(round(float(evaluation.min()))),
                "seasonTotals": [round(float(value)) for value in totals],
                "averageTransfers": round(
                    float(
                        np.mean(
                            [row["transfers"] for row in stats[training_count:]]
                        )
                    ),
                    1,
                ),
            }
        )
    selected = max(rows, key=lambda row: row["trainingStability"])
    result = {
        "status": "training-selected calibrated-scale policy",
        "selected": selected,
        "experiments": rows,
        "referenceFullStackAverage": 2149.5,
        "selectedLift": round(selected["evaluationAverage"] - 2149.5, 1),
    }
    output = lens.ROOT / "analysis" / "data" / "calibrated_horizon_policy.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
