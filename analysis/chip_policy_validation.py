"""Select chip thresholds on past seasons and validate them recursively."""

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
    strategy = lens.SimulationStrategy(
        name="fixed winner-principles decision layer",
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
        squad_captain_weight=0.70,
        squad_bench_weight=0.18,
    )
    baseline, baseline_stats = lens.simulate_candidate(
        data, scores, strategy, plan_scores=plan
    )
    fresh = lens.precompute_fresh_squads(data, plan)
    free_hits = lens.precompute_fresh_squads(data, scores)
    policies = [
        lens.ChipPolicy(wildcard_gap, free_hit_gap, bench_score, triple_score, 0.55, first_start, second_start)
        for wildcard_gap in (45.0, 60.0, 75.0)
        for free_hit_gap in (10.0, 20.0)
        for bench_score, triple_score in ((11.0, 15.0), (16.0, 21.0))
        for first_start, second_start in ((6, 20), (8, 24), (10, 28))
    ]
    totals = []
    stats = []
    for index, policy in enumerate(policies, start=1):
        season_totals, season_stats = lens.simulate_candidate(
            data,
            scores,
            strategy,
            chip_policy=policy,
            fresh_squads=fresh,
            free_hit_squads=free_hits,
            plan_scores=plan,
        )
        totals.append(season_totals)
        stats.append(season_stats)
        print(f"Chip policy {index}/{len(policies)}")
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
        policy_index = row.pop("selected")
        row["policy"] = policies[policy_index].as_dict()
        row["noChipPoints"] = round(float(baseline[season_index]))
        row["realisedChipDelta"] = row["points"] - row["noChipPoints"]
        row["target"] = targets[seasons[season_index]]
        row["margin"] = row["points"] - row["target"]
        row["chips"] = stats[policy_index][season_index]["chips"]
        row["transfers"] = stats[policy_index][season_index]["transfers"]
        row["rolled"] = stats[policy_index][season_index]["rolled"]
    result = {
        "policies": len(policies),
        "selection": "expanding-season nested; every threshold is chosen only from prior seasons",
        "nestedSeasons": nested,
        "nestedAverage": round(float(np.mean([row["points"] for row in nested])), 1),
        "nestedAverageNoChips": round(
            float(np.mean([row["noChipPoints"] for row in nested])), 1
        ),
        "nestedAverageChipDelta": round(
            float(np.mean([row["realisedChipDelta"] for row in nested])), 1
        ),
        "nestedTargetHits": sum(row["margin"] >= 0 for row in nested),
        "nestedAverageMargin": round(float(np.mean([row["margin"] for row in nested])), 1),
        "policyAverages": [
            {
                "policy": policy.as_dict(),
                "trainingAverage": round(float(matrix[index, :2].mean()), 1),
                "evaluationAverage": round(float(matrix[index, 2:].mean()), 1),
                "evaluationChipDelta": round(
                    float((matrix[index, 2:] - baseline[2:]).mean()), 1
                ),
                "evaluationMinimum": round(float(matrix[index, 2:].min())),
            }
            for index, policy in enumerate(policies)
        ],
        "baselineStats": baseline_stats,
    }
    output = lens.ROOT / "analysis" / "data" / "chip_policy_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
