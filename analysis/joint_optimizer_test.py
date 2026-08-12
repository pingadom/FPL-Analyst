"""Test the legal beam-search transfer optimiser with nested season selection."""

from __future__ import annotations

import json

import numpy as np

import calibrate_model as lens
from elite_policy_search import candidate_from_weights, nested_choices


def main() -> None:
    data, _ = lens.load_or_build_prepared_history()
    artifact = json.loads(lens.OUTPUT.read_text(encoding="utf-8-sig"))
    candidate = candidate_from_weights(artifact["model"]["weights"])
    scores, plan, _ = lens.candidate_forecasts(data, candidate, robust_planning=False)
    seasons = list(dict.fromkeys(data["season"].tolist()))

    experiments = []
    for immediate_share in (0.50, 0.75, 1.0):
        persistent_plan = (
            (1 - immediate_share) * plan + immediate_share * scores * 4.5
        )
        for hurdle in (8.0, 12.0, 16.0, 20.0):
            strategy = lens.SimulationStrategy(
                    name=f"joint-h{hurdle:.1f}-now{immediate_share:.2f}",
                    transfer_hurdle=hurdle,
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
            experiments.append((strategy, persistent_plan, immediate_share))
    totals = []
    stats = []
    fixed_preseason_chip_policy = lens.ChipPolicy(3.0, 1.10, 3.70, 1.70, 0.40)
    for index, (strategy, experiment_plan, _) in enumerate(experiments, start=1):
        season_totals, season_stats = lens.simulate_candidate(
            data, scores, strategy, plan_scores=experiment_plan
        )
        totals.append(season_totals)
        stats.append(season_stats)
        print(f"Joint policy {index}/{len(experiments)}: {strategy.name}")
    matrix = np.vstack(totals)
    nested = nested_choices(matrix, seasons)
    selected_policy_indices = [int(row["selected"]) for row in nested]
    selected_chip_totals: dict[int, np.ndarray] = {}
    selected_chip_stats: dict[int, list[dict]] = {}
    for strategy_index in sorted(set(selected_policy_indices)):
        strategy, experiment_plan, _ = experiments[strategy_index]
        fresh = lens.precompute_fresh_squads(data, experiment_plan)
        free_hits = lens.precompute_fresh_squads(data, scores)
        season_chip_totals, season_chip_stats = lens.simulate_candidate(
            data,
            scores,
            strategy,
            chip_policy=fixed_preseason_chip_policy,
            fresh_squads=fresh,
            free_hit_squads=free_hits,
            plan_scores=experiment_plan,
        )
        selected_chip_totals[strategy_index] = season_chip_totals
        selected_chip_stats[strategy_index] = season_chip_stats
        print(f"Chip validation: {strategy.name}")
    benchmark = json.loads(
        (lens.ROOT / "analysis" / "data" / "historical_rank_benchmarks.json").read_text(
            encoding="utf-8"
        )
    )
    targets = {
        item["season"].replace("/", "-"): int(item["points"])
        for item in benchmark["seasons"]
    }
    for season_index, row in enumerate(nested, start=len(lens.TRAINING_SEASONS)):
        strategy_index = row.pop("selected")
        row["strategy"] = experiments[strategy_index][0].name
        row["target"] = targets[seasons[season_index]]
        row["margin"] = row["points"] - row["target"]
        row["transfers"] = stats[strategy_index][season_index]["transfers"]
        row["rolled"] = stats[strategy_index][season_index]["rolled"]
    nested_with_chips = []
    for season_index, strategy_index in enumerate(
        selected_policy_indices, start=len(lens.TRAINING_SEASONS)
    ):
        season_stat = selected_chip_stats[strategy_index][season_index]
        row = {
            "season": seasons[season_index].replace("-", "/"),
            "points": round(float(selected_chip_totals[strategy_index][season_index])),
            "strategy": experiments[strategy_index][0].name,
        }
        row["target"] = targets[seasons[season_index]]
        row["margin"] = row["points"] - row["target"]
        row["transfers"] = season_stat["transfers"]
        row["rolled"] = season_stat["rolled"]
        row["immediateChipGain"] = season_stat["immediateChipGain"]
        row["chips"] = season_stat["chips"]
        nested_with_chips.append(row)
    result = {
        "policies": len(experiments),
        "nestedSeasons": nested,
        "nestedAverage": round(float(np.mean([row["points"] for row in nested])), 1),
        "nestedTargetHitsBeforeChips": sum(row["margin"] >= 0 for row in nested),
        "nestedAverageMarginBeforeChips": round(
            float(np.mean([row["margin"] for row in nested])), 1
        ),
        "fixedPreseasonChipPolicy": fixed_preseason_chip_policy.as_dict(),
        "nestedWithChips": nested_with_chips,
        "nestedAverageWithChips": round(
            float(np.mean([row["points"] for row in nested_with_chips])), 1
        ),
        "nestedTargetHitsWithChips": sum(
            row["margin"] >= 0 for row in nested_with_chips
        ),
        "nestedAverageMarginWithChips": round(
            float(np.mean([row["margin"] for row in nested_with_chips])), 1
        ),
        "policyAverages": [
            {
                "strategy": experiment[0].name,
                "immediateShare": experiment[2],
                "evaluationAverage": round(float(matrix[index, 2:].mean()), 1),
                "evaluationMinimum": round(float(matrix[index, 2:].min())),
                "averageTransfers": round(
                    float(np.mean([item["transfers"] for item in stats[index][2:]])), 1
                ),
            }
            for index, experiment in enumerate(experiments)
        ],
    }
    output = lens.ROOT / "analysis" / "data" / "joint_optimizer_test.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
