"""Audit forecast-stack selection using only information available before each season."""

from __future__ import annotations

import json

import numpy as np

import calibrate_model as lens
from captain_ranker_validation import rank_blend
from frontier_ranker_validation import PLAYER_CANDIDATE, STRATEGY
from listwise_ranker_validation import quantile_map


def main() -> None:
    data, _ = lens.load_or_build_prepared_history()
    data = data.reset_index(drop=True)
    immediate, horizon, _ = lens.candidate_forecasts(
        data,
        PLAYER_CANDIDATE,
        robust_planning=False,
        schedule_censored=True,
    )
    stable_plan = 0.75 * immediate * 4.5 + 0.25 * horizon
    frontier_raw = np.load(lens.CACHE / "frontier-causal-predictions-v2.npz")[
        "prediction"
    ]
    horizon_raw = np.load(lens.CACHE / "listwise-horizon_target-v1.npz")[
        "prediction"
    ]
    captain_raw = np.load(lens.CACHE / "captain-listwise-v1.npz")["prediction"]
    immediate_mapped = quantile_map(data, frontier_raw, immediate)
    plan_mapped = quantile_map(data, horizon_raw, stable_plan)
    frontier25 = 0.75 * immediate + 0.25 * immediate_mapped
    list_plan25 = 0.75 * stable_plan + 0.25 * plan_mapped
    captain50 = rank_blend(data, immediate, captain_raw, 0.50)

    variants = (
        ("structural", immediate, stable_plan, None),
        ("frontier25", frontier25, stable_plan, None),
        ("listPlan25", immediate, list_plan25, None),
        ("frontier25-listPlan25", frontier25, list_plan25, None),
        ("fullStack-captain50", frontier25, list_plan25, captain50),
    )
    seasons = list(dict.fromkeys(data["season"].tolist()))
    totals = []
    stats = []
    for index, (name, score, plan, captain) in enumerate(variants, start=1):
        print(f"Running {index}/{len(variants)}: {name}", flush=True)
        season_totals, season_stats = lens.simulate_candidate(
            data,
            score,
            STRATEGY,
            plan_scores=plan,
            captain_scores=captain,
            tracked_player_name="Salah",
        )
        totals.append(season_totals)
        stats.append(season_stats)
    matrix = np.vstack(totals)
    training_count = len(lens.TRAINING_SEASONS)

    training_stability = (
        matrix[:, :training_count].mean(axis=1)
        - 0.25 * matrix[:, :training_count].std(axis=1)
    )
    static_index = int(np.argmax(training_stability))
    static_points = matrix[static_index, training_count:]

    rolling = []
    for season_index in range(training_count, len(seasons)):
        history = matrix[:, :season_index]
        stability = history.mean(axis=1) - 0.25 * history.std(axis=1)
        chosen = int(np.argmax(stability))
        rolling.append(
            {
                "season": seasons[season_index].replace("-", "/"),
                "selected": variants[chosen][0],
                "points": int(round(float(matrix[chosen, season_index]))),
            }
        )

    oracle_points = matrix[:, training_count:].max(axis=0)
    result = {
        "status": "causal stack-selection audit",
        "variants": [
            {
                "name": variants[index][0],
                "trainingStability": round(float(training_stability[index]), 3),
                "evaluationAverage": round(
                    float(matrix[index, training_count:].mean()), 1
                ),
                "evaluationMinimum": int(
                    round(float(matrix[index, training_count:].min()))
                ),
                "seasonTotals": [round(float(value)) for value in matrix[index]],
            }
            for index in range(len(variants))
        ],
        "staticTrainingOnly": {
            "selected": variants[static_index][0],
            "average": round(float(static_points.mean()), 1),
            "minimum": int(round(float(static_points.min()))),
        },
        "rollingPriorSeasonsOnly": {
            "average": round(float(np.mean([row["points"] for row in rolling])), 1),
            "minimum": min(row["points"] for row in rolling),
            "seasons": rolling,
        },
        "postExposureFullStack": {
            "average": round(float(matrix[-1, training_count:].mean()), 1),
            "minimum": int(round(float(matrix[-1, training_count:].min()))),
        },
        "variantOracleDiagnostic": {
            "average": round(float(oracle_points.mean()), 1),
            "minimum": int(round(float(oracle_points.min()))),
        },
    }
    output = lens.ROOT / "analysis" / "data" / "walk_forward_stack_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
