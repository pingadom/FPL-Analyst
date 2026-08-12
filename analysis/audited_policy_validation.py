"""Final compact validation on the corrected, schedule-censored simulator.

Every decision-layer choice is frozen using only 2016/17 and 2017/18. The
eight seasons from 2018/19 onward are reported once as untouched evaluation.
"""

from __future__ import annotations

import json

import numpy as np

import calibrate_model as lens


def main() -> None:
    data, _ = lens.load_or_build_prepared_history()
    seasons = list(dict.fromkeys(data["season"].tolist()))
    candidate = lens.Candidate(
        0.32, 0.05, 0.00, 0.13, 0.19, 0.03, 0.18, 0.10, 0.76
    )
    scores, horizon, _ = lens.candidate_forecasts(
        data, candidate, robust_planning=False, schedule_censored=True
    )

    decision_options: list[tuple[lens.SimulationStrategy, np.ndarray]] = []
    for immediate_share in (0.65, 0.80):
        plan = immediate_share * scores * 4.5 + (1 - immediate_share) * horizon
        for hurdle in (12.0, 16.0, 20.0):
            for captain_mode in ("expected", "attacking_tail"):
                decision_options.append(
                    (
                        lens.SimulationStrategy(
                            name=(
                                f"audited-team-h{hurdle:.0f}-"
                                f"now{immediate_share:.2f}-{captain_mode}"
                            ),
                            transfer_hurdle=hurdle,
                            bank_limit=5,
                            force_weekly_review=False,
                            safe_captain=False,
                            max_hits=0,
                            hit_immediate_hurdle=99.0,
                            joint_chip_preflight=True,
                            hold_option_value=0.25,
                            captain_mode=captain_mode,
                            phase_banking=False,
                            early_price_weight=0.6,
                            joint_squad_optimiser=True,
                            squad_captain_weight=0.70,
                            squad_bench_weight=0.05,
                        ),
                        plan,
                    )
                )

    decision_totals = []
    decision_stats = []
    for index, (strategy, plan) in enumerate(decision_options, start=1):
        totals, stats = lens.simulate_candidate(
            data, scores, strategy, plan_scores=plan
        )
        decision_totals.append(totals)
        decision_stats.append(stats)
        print(f"Decision option {index}/{len(decision_options)}: {strategy.name}")
    decision_matrix = np.vstack(decision_totals)
    training_stability = (
        decision_matrix[:, :2].mean(axis=1)
        - 0.25 * decision_matrix[:, :2].std(axis=1)
    )
    selected_decision = int(np.argmax(training_stability))
    strategy, plan = decision_options[selected_decision]
    no_chip = decision_matrix[selected_decision]

    policies = [
        lens.ChipPolicy(60, 20, 11, 15, 0.55, 10, 28),
        lens.ChipPolicy(60, 20, 16, 21, 0.55, 10, 28),
        lens.ChipPolicy(75, 20, 16, 21, 0.55, 10, 28),
        lens.ChipPolicy(75, 25, 16, 21, 0.55, 10, 28),
    ]
    fresh = lens.precompute_fresh_squads(data, plan)
    free_hits = lens.precompute_fresh_squads(data, scores)
    chip_totals = []
    chip_stats = []
    for index, policy in enumerate(policies, start=1):
        totals, stats = lens.simulate_candidate(
            data,
            scores,
            strategy,
            chip_policy=policy,
            fresh_squads=fresh,
            free_hit_squads=free_hits,
            plan_scores=plan,
        )
        chip_totals.append(totals)
        chip_stats.append(stats)
        print(f"Chip option {index}/{len(policies)}")
    chip_matrix = np.vstack(chip_totals)
    training_chip_gain = chip_matrix[:, :2] - no_chip[:2]
    chip_stability = (
        training_chip_gain.mean(axis=1) - 0.25 * training_chip_gain.std(axis=1)
    )
    selected_chip = int(np.argmax(chip_stability))
    final = chip_matrix[selected_chip]

    benchmark = json.loads(
        (lens.ROOT / "analysis" / "data" / "historical_rank_benchmarks.json").read_text(
            encoding="utf-8"
        )
    )
    targets = {
        row["season"].replace("/", "-"): int(row["points"])
        for row in benchmark["seasons"]
    }
    evaluation = []
    for season_index in range(2, len(seasons)):
        points = round(float(final[season_index]))
        target = targets[seasons[season_index]]
        evaluation.append(
            {
                "season": seasons[season_index].replace("-", "/"),
                "points": points,
                "noChipPoints": round(float(no_chip[season_index])),
                "chipDelta": round(float(final[season_index] - no_chip[season_index])),
                "target": target,
                "margin": points - target,
                "transfers": chip_stats[selected_chip][season_index]["transfers"],
                "rolled": chip_stats[selected_chip][season_index]["rolled"],
                "chips": chip_stats[selected_chip][season_index]["chips"],
            }
        )

    result = {
        "selection": "Frozen on 2016/17 and 2017/18 only",
        "playerFamily": "team_fixture_minutes with role ensemble",
        "weights": candidate.as_dict(),
        "decisionOptions": len(decision_options),
        "selectedDecision": strategy.name,
        "selectedDecisionTrainingScore": round(float(training_stability[selected_decision]), 1),
        "chipOptions": len(policies),
        "selectedChipPolicy": policies[selected_chip].as_dict(),
        "selectedChipTrainingGain": round(float(training_chip_gain[selected_chip].mean()), 1),
        "evaluation": evaluation,
        "average": round(float(np.mean([row["points"] for row in evaluation])), 1),
        "averageNoChips": round(
            float(np.mean([row["noChipPoints"] for row in evaluation])), 1
        ),
        "averageChipDelta": round(
            float(np.mean([row["chipDelta"] for row in evaluation])), 1
        ),
        "targetHits": sum(row["margin"] >= 0 for row in evaluation),
        "averageMargin": round(float(np.mean([row["margin"] for row in evaluation])), 1),
        "minimum": min(row["points"] for row in evaluation),
    }
    output = lens.ROOT / "analysis" / "data" / "audited_policy_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
