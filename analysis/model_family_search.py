"""Pair fixed player-weight families with the corrected XI-weighted optimiser."""

from __future__ import annotations

import json

import numpy as np

import calibrate_model as lens
from elite_policy_search import candidate_from_weights, nested_choices


def fixed_candidates(artifact: dict) -> dict[str, lens.Candidate]:
    return {
        "lens7_calibrated": candidate_from_weights(artifact["model"]["weights"]),
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
        "winner_principles": lens.Candidate(
            0.30, 0.06, 0.00, 0.13, 0.17, 0.03, 0.18, 0.13, 0.78
        ),
        "team_fixture_minutes": lens.Candidate(
            0.32, 0.05, 0.00, 0.13, 0.19, 0.03, 0.18, 0.10, 0.76
        ),
        "market_safety": lens.Candidate(
            0.32, 0.08, 0.00, 0.11, 0.15, 0.13, 0.15, 0.06, 0.78
        ),
        "underlying_team": lens.Candidate(
            0.28, 0.05, 0.00, 0.11, 0.18, 0.03, 0.15, 0.20, 0.75
        ),
    }


def main() -> None:
    data, _ = lens.load_or_build_prepared_history()
    artifact = json.loads(lens.OUTPUT.read_text(encoding="utf-8-sig"))
    seasons = list(dict.fromkeys(data["season"].tolist()))
    experiments = []
    totals = []
    stats = []
    for model_name, candidate in fixed_candidates(artifact).items():
        scores, horizon, calibration_score = lens.candidate_forecasts(
            data, candidate, robust_planning=False
        )
        calibration = 0.72 + 0.56 * calibration_score
        non_role_weight = (1 - data["ensemble_role_weight"]).clip(lower=0.05)
        no_role_component = (
            data["component_xpts"]
            - data["ensemble_role_weight"] * data["role_ridge_xpts"]
        ) / non_role_weight
        layers = {
            "role": (scores, horizon),
            "no_role": (
                no_role_component.to_numpy(float) * calibration,
                data["component_horizon_censored"].to_numpy(float)
                * (
                    no_role_component / data["component_xpts"].clip(lower=0.2)
                ).to_numpy(float)
                * calibration,
            ),
        }
        for layer_name, (layer_scores, layer_horizon) in layers.items():
            plan = 0.25 * layer_horizon + 0.75 * layer_scores * 4.5
            strategy = lens.SimulationStrategy(
                name=f"{model_name}:{layer_name}",
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
                data, layer_scores, strategy, plan_scores=plan
            )
            experiments.append(
                {
                    "model": model_name,
                    "layer": layer_name,
                    "weights": candidate.as_dict(),
                }
            )
            totals.append(season_totals)
            stats.append(season_stats)
            print(f"Tested {model_name}:{layer_name}")
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
        "nestedTargetHitsBeforeChips": sum(row["margin"] >= 0 for row in nested),
        "nestedAverageMarginBeforeChips": round(
            float(np.mean([row["margin"] for row in nested])), 1
        ),
        "experimentAverages": [
            {
                **experiment,
                "evaluationAverage": round(float(matrix[index, 2:].mean()), 1),
                "evaluationMinimum": round(float(matrix[index, 2:].min())),
                "averageTransfers": round(
                    float(np.mean([item["transfers"] for item in stats[index][2:]])),
                    1,
                ),
            }
            for index, experiment in enumerate(experiments)
        ],
    }
    output = lens.ROOT / "analysis" / "data" / "model_family_search.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
