"""Training-only search for a consistent multi-horizon FPL decision objective.

The search is deliberately compact. 2016/17 and 2017/18 select one policy;
2018/19 onward are read once as evaluation. No website code is involved.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace

import numpy as np

import calibrate_model as lens
from bench_efficiency_validation import championship_forecasts, variant_summary
from frontier_ranker_validation import STRATEGY


EXPERIMENTS = [
    {
        "name": "horizon-core",
        "decision_immediate_share": 0.00,
        "decision_uncertainty_penalty": 0.00,
        "bench_reliability_weight": 0.00,
        "squad_bench_weight": 0.05,
        "bench_premium_penalty": 0.000,
    },
    {
        "name": "horizon-80-current-20",
        "decision_immediate_share": 0.20,
        "decision_uncertainty_penalty": 0.00,
        "bench_reliability_weight": 0.00,
        "squad_bench_weight": 0.05,
        "bench_premium_penalty": 0.000,
    },
    {
        "name": "horizon-65-current-35",
        "decision_immediate_share": 0.35,
        "decision_uncertainty_penalty": 0.00,
        "bench_reliability_weight": 0.00,
        "squad_bench_weight": 0.05,
        "bench_premium_penalty": 0.000,
    },
    {
        "name": "robust-horizon-current",
        "decision_immediate_share": 0.20,
        "decision_uncertainty_penalty": 0.08,
        "bench_reliability_weight": 0.00,
        "squad_bench_weight": 0.05,
        "bench_premium_penalty": 0.000,
    },
    {
        "name": "rotation-insurance-soft",
        "decision_immediate_share": 0.20,
        "decision_uncertainty_penalty": 0.00,
        "bench_reliability_weight": 0.50,
        "squad_bench_weight": 0.08,
        "bench_premium_penalty": 0.005,
    },
    {
        "name": "rotation-value-efficient",
        "decision_immediate_share": 0.20,
        "decision_uncertainty_penalty": 0.00,
        "bench_reliability_weight": 0.50,
        "squad_bench_weight": 0.05,
        "bench_premium_penalty": 0.010,
    },
    {
        "name": "exact-horizon-core-control",
        "decision_immediate_share": 0.00,
        "decision_uncertainty_penalty": 0.00,
        "bench_reliability_weight": 0.00,
        "squad_bench_weight": 0.05,
        "bench_premium_penalty": 0.000,
        "exact": True,
    },
]


def strategy_for(experiment: dict) -> lens.SimulationStrategy:
    penalty = float(experiment["bench_premium_penalty"])
    return replace(
        STRATEGY,
        name=str(experiment["name"]),
        exact_initial_optimiser=bool(experiment.get("exact", False)),
        initial_spend_gap=None,
        bench_premium_limit=None,
        bench_premium_penalty=penalty,
        transfer_bench_premium_penalty=penalty,
        decision_immediate_share=float(experiment["decision_immediate_share"]),
        decision_uncertainty_penalty=float(
            experiment["decision_uncertainty_penalty"]
        ),
        bench_reliability_weight=float(experiment["bench_reliability_weight"]),
        squad_bench_weight=float(experiment["squad_bench_weight"]),
    )


def main() -> None:
    data, _ = lens.load_or_build_prepared_history()
    data = data.reset_index(drop=True)
    scores, plan_scores, captain_scores = championship_forecasts(data)
    seasons = list(dict.fromkeys(data["season"].tolist()))
    rows = []

    print("Running frozen heuristic baseline", flush=True)
    baseline_totals, baseline_stats = lens.simulate_candidate(
        data,
        scores,
        STRATEGY,
        plan_scores=plan_scores,
        captain_scores=captain_scores,
        tracked_player_name="Salah",
    )
    baseline = variant_summary(baseline_totals, baseline_stats, seasons)

    for index, experiment in enumerate(EXPERIMENTS, start=1):
        strategy = strategy_for(experiment)
        print(f"Running challenger {index}/{len(EXPERIMENTS)}: {strategy.name}", flush=True)
        totals, stats = lens.simulate_candidate(
            data,
            scores,
            strategy,
            plan_scores=plan_scores,
            captain_scores=captain_scores,
            tracked_player_name="Salah",
        )
        summary = variant_summary(totals, stats, seasons)
        training = totals[: len(lens.TRAINING_SEASONS)]
        rows.append(
            {
                "experiment": experiment,
                "strategy": asdict(strategy),
                "trainingAverage": round(float(training.mean()), 1),
                "trainingMinimum": round(float(training.min()), 1),
                "trainingStability": round(
                    float(training.mean() - 0.25 * training.std()), 3
                ),
                "evaluation": summary,
                "allSeasonTotals": [round(float(value), 1) for value in totals],
            }
        )

    selected = max(rows, key=lambda row: row["trainingStability"])
    result = {
        "status": "training-selected causal challenger; evaluation seasons not used for selection",
        "selectionSeasons": [season.replace("-", "/") for season in lens.TRAINING_SEASONS],
        "baseline": baseline,
        "selected": selected,
        "selectedLift": {
            "average": round(
                selected["evaluation"]["average"] - baseline["average"], 1
            ),
            "minimum": selected["evaluation"]["minimum"] - baseline["minimum"],
            "total": selected["evaluation"]["total"] - baseline["total"],
        },
        "experiments": rows,
    }
    output = lens.ROOT / "analysis" / "data" / "model_engine_challenger.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "baselineAverage": baseline["average"],
                "selected": selected["experiment"],
                "selectedTrainingStability": selected["trainingStability"],
                "selectedEvaluationAverage": selected["evaluation"]["average"],
                "selectedLift": result["selectedLift"],
                "all": [
                    {
                        "name": row["experiment"]["name"],
                        "trainingStability": row["trainingStability"],
                        "evaluationAverage": row["evaluation"]["average"],
                        "evaluationMinimum": row["evaluation"]["minimum"],
                    }
                    for row in rows
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
