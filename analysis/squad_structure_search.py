"""Test captain concentration and bench insurance in legal squad optimisation."""

from __future__ import annotations

import json

import numpy as np

import calibrate_model as lens
from elite_policy_search import nested_choices


def main() -> None:
    data, _ = lens.load_or_build_prepared_history()
    seasons = list(dict.fromkeys(data["season"].tolist()))
    candidate = lens.Candidate(
        0.30, 0.06, 0.00, 0.13, 0.17, 0.03, 0.18, 0.13, 0.78
    )
    scores, horizon, _ = lens.candidate_forecasts(
        data, candidate, robust_planning=False
    )
    plan = 0.75 * scores * 4.5 + 0.25 * horizon
    experiments = []
    totals = []
    stats = []
    for captain_weight in (0.70, 1.00, 1.30):
        for bench_weight in (0.05, 0.18, 0.35):
            strategy = lens.SimulationStrategy(
                name=f"cap{captain_weight:.2f}-bench{bench_weight:.2f}",
                transfer_hurdle=16.0,
                bank_limit=5,
                force_weekly_review=False,
                safe_captain=False,
                max_hits=0,
                hit_immediate_hurdle=99.0,
                joint_chip_preflight=True,
                hold_option_value=0.25,
                captain_mode="expected",
                phase_banking=False,
                early_price_weight=0.6,
                joint_squad_optimiser=True,
                squad_captain_weight=captain_weight,
                squad_bench_weight=bench_weight,
            )
            season_totals, season_stats = lens.simulate_candidate(
                data, scores, strategy, plan_scores=plan
            )
            experiments.append(
                {"captainWeight": captain_weight, "benchWeight": bench_weight}
            )
            totals.append(season_totals)
            stats.append(season_stats)
            print(f"Tested {strategy.name}")

    matrix = np.vstack(totals)
    nested = nested_choices(matrix, seasons)
    benchmarks = json.loads(
        (lens.ROOT / "analysis" / "data" / "historical_rank_benchmarks.json").read_text(
            encoding="utf-8"
        )
    )
    targets = {
        item["season"].replace("/", "-"): int(item["points"])
        for item in benchmarks["seasons"]
    }
    for season_index, row in enumerate(nested, start=len(lens.TRAINING_SEASONS)):
        experiment_index = row.pop("selected")
        row.update(experiments[experiment_index])
        row["target"] = targets[seasons[season_index]]
        row["margin"] = row["points"] - row["target"]
        row["transfers"] = stats[experiment_index][season_index]["transfers"]
        row["rolled"] = stats[experiment_index][season_index]["rolled"]
    result = {
        "experiments": len(experiments),
        "nestedSeasons": nested,
        "nestedAverage": round(float(np.mean([row["points"] for row in nested])), 1),
        "nestedTargetHits": sum(row["margin"] >= 0 for row in nested),
        "nestedAverageMargin": round(float(np.mean([row["margin"] for row in nested])), 1),
        "experimentAverages": [
            {
                **experiment,
                "trainingAverage": round(float(matrix[index, :2].mean()), 1),
                "evaluationAverage": round(float(matrix[index, 2:].mean()), 1),
                "evaluationMinimum": round(float(matrix[index, 2:].min())),
                "averageTransfers": round(
                    float(np.mean([item["transfers"] for item in stats[index][2:]])), 1
                ),
            }
            for index, experiment in enumerate(experiments)
        ],
    }
    output = lens.ROOT / "analysis" / "data" / "squad_structure_search.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
