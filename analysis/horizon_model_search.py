"""Strict nested comparison of heuristic and directly supervised squad horizons."""

from __future__ import annotations

import json

import numpy as np

import calibrate_model as lens
from elite_policy_search import candidate_from_weights, nested_choices


def main() -> None:
    data, _ = lens.load_or_build_prepared_history()
    seasons = list(dict.fromkeys(data["season"].tolist()))

    target = data["horizon_target"].to_numpy(float)
    baseline_horizon = data["component_horizon_censored"].to_numpy(float)
    ridge_horizon = data["causal_horizon_ridge"].to_numpy(float)
    diagnostic_rows = []
    for season in seasons:
        mask = data["season"].to_numpy() == season
        diagnostic_rows.append(
            {
                "season": season.replace("-", "/"),
                "baselineMae": round(float(np.mean(np.abs(baseline_horizon[mask] - target[mask]))), 3),
                "ridgeMae": round(float(np.mean(np.abs(ridge_horizon[mask] - target[mask]))), 3),
                "baselineCorrelation": round(float(np.corrcoef(baseline_horizon[mask], target[mask])[0, 1]), 3),
                "ridgeCorrelation": round(float(np.corrcoef(ridge_horizon[mask], target[mask])[0, 1]), 3),
            }
        )
    print("Horizon diagnostics", diagnostic_rows)

    artifact = json.loads(lens.OUTPUT.read_text(encoding="utf-8-sig"))
    candidates = {
        "winner_principles": lens.Candidate(
            0.30, 0.06, 0.00, 0.13, 0.17, 0.03, 0.18, 0.13, 0.78
        ),
        "lens6_calibrated": candidate_from_weights(
            {
                "performance": 29,
                "value": 18,
                "age": 2,
                "fixture": 4,
                "team": 9,
                "crowd": 15,
                "minutes": 14,
                "underlying": 9,
                "recent": 83,
            }
        ),
        "lens7_calibrated": candidate_from_weights(artifact["model"]["weights"]),
    }
    plan_specs = {
        "old-now75": lambda current, old, ridge: 0.75 * current * 4.5 + 0.25 * old,
        "ridge-now75": lambda current, old, ridge: 0.75 * current * 4.5 + 0.25 * ridge,
        "ridge-now50": lambda current, old, ridge: 0.50 * current * 4.5 + 0.50 * ridge,
        "ridge-now25": lambda current, old, ridge: 0.25 * current * 4.5 + 0.75 * ridge,
        "dual-now50": lambda current, old, ridge: 0.50 * current * 4.5 + 0.25 * old + 0.25 * ridge,
    }
    experiments: list[dict] = []
    totals: list[np.ndarray] = []
    stats: list[list[dict]] = []
    for candidate_name, candidate in candidates.items():
        current, candidate_horizon, _ = lens.candidate_forecasts(
            data, candidate, robust_planning=False
        )
        calibration = np.divide(
            candidate_horizon,
            baseline_horizon,
            out=np.ones_like(candidate_horizon),
            where=baseline_horizon > 0.05,
        )
        candidate_ridge = ridge_horizon * calibration
        for plan_name, builder in plan_specs.items():
            plan = builder(current, candidate_horizon, candidate_ridge)
            strategy = lens.SimulationStrategy(
                name=f"{candidate_name}:{plan_name}",
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
            )
            season_totals, season_stats = lens.simulate_candidate(
                data, current, strategy, plan_scores=plan
            )
            experiments.append(
                {
                    "candidate": candidate_name,
                    "plan": plan_name,
                    "weights": candidate.as_dict(),
                }
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
        "horizonDiagnostics": diagnostic_rows,
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
                "evaluationTargetHits": int(
                    sum(
                        matrix[index, season_index] >= targets[season]
                        for season_index, season in enumerate(seasons[2:], start=2)
                    )
                ),
                "averageTransfers": round(
                    float(np.mean([item["transfers"] for item in stats[index][2:]])), 1
                ),
            }
            for index, experiment in enumerate(experiments)
        ],
    }
    output = lens.ROOT / "analysis" / "data" / "horizon_model_search.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
