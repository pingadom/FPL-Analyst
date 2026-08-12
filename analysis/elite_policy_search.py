"""Nested, season-level search for elite FPL decision policies.

This is intentionally separate from the 2,400 player-weight calibration. It
tests whether transfer/captain policy generalises when each evaluation season
is chosen using only earlier seasons, and compares the Lens 7 player layer with
the incumbent no-role blend.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np

import calibrate_model as lens


def candidate_from_weights(weights: dict) -> lens.Candidate:
    keys = [
        "performance",
        "value",
        "age",
        "fixture",
        "team",
        "crowd",
        "minutes",
        "underlying",
    ]
    raw = np.array([float(weights[key]) for key in keys], dtype=float)
    raw /= raw.sum()
    return lens.Candidate(*raw.tolist(), float(weights["recent"]) / 100)


def strategy_pool() -> list[lens.SimulationStrategy]:
    strategies: list[lens.SimulationStrategy] = []
    for hurdle in (4.2, 5.0, 5.8, 6.8, 8.0):
        for hold in (0.0, 0.35, 0.8):
            for phase in (False, True):
                for captain in ("expected", "attacking_tail"):
                    name = (
                        f"h{hurdle:.1f}-hold{hold:.2f}-"
                        f"{'phase' if phase else 'flat'}-{captain}"
                    )
                    strategies.append(
                        lens.SimulationStrategy(
                            name=name,
                            transfer_hurdle=hurdle,
                            bank_limit=5,
                            force_weekly_review=False,
                            safe_captain=False,
                            max_hits=0,
                            hit_immediate_hurdle=99.0,
                            joint_chip_preflight=True,
                            hold_option_value=hold,
                            captain_mode=captain,
                            phase_banking=phase,
                            early_price_weight=1.10 if phase else 0.0,
                        )
                    )
    return strategies


def nested_choices(scores: np.ndarray, season_names: list[str]) -> list[dict]:
    # Standardise within season before aggregation so high-scoring FPL seasons
    # do not dominate the policy choice.
    standardised = (scores - scores.mean(axis=0)) / scores.std(axis=0).clip(1e-6)
    rows: list[dict] = []
    for season_index in range(len(lens.TRAINING_SEASONS), len(season_names)):
        prior = standardised[:, :season_index]
        stability = prior.mean(axis=1) - 0.25 * prior.std(axis=1)
        selected = int(np.argmax(stability))
        rows.append(
            {
                "season": season_names[season_index].replace("-", "/"),
                "selected": selected,
                "points": round(float(scores[selected, season_index])),
            }
        )
    return rows


def main() -> None:
    data, _ = lens.load_or_build_prepared_history()
    artifact = json.loads(lens.OUTPUT.read_text(encoding="utf-8-sig"))
    candidates = {
        "lens7_role": candidate_from_weights(artifact["model"]["weights"]),
        "lens6_weights": candidate_from_weights(
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
    }
    strategies = strategy_pool()
    seasons = list(dict.fromkeys(data["season"].tolist()))
    combinations: list[dict] = []
    totals: list[np.ndarray] = []
    transfer_stats: list[list[dict]] = []
    for model_name, candidate in candidates.items():
        candidate_scores, candidate_plan, model_score = lens.candidate_forecasts(
            data, candidate, robust_planning=False
        )
        calibration = 0.72 + 0.56 * model_score
        variants = {"role": (candidate_scores, candidate_plan)}
        non_role_weight = (1 - data["ensemble_role_weight"]).clip(lower=0.05)
        no_role_component = (
            data["component_xpts"]
            - data["ensemble_role_weight"] * data["role_ridge_xpts"]
        ) / non_role_weight
        variants["no_role"] = (
            no_role_component.to_numpy(float) * calibration,
            data["component_horizon_censored"].to_numpy(float)
            * (no_role_component / data["component_xpts"].clip(lower=0.2)).to_numpy(float)
            * calibration,
        )
        for variant_name, (scores, plan) in variants.items():
            for strategy in strategies:
                season_totals, stats = lens.simulate_candidate(
                    data, scores, strategy, plan_scores=plan
                )
                combinations.append(
                    {
                        "model": model_name,
                        "playerLayer": variant_name,
                        "strategy": strategy.name,
                        "strategyConfig": asdict(strategy),
                    }
                )
                totals.append(season_totals)
                transfer_stats.append(stats)
            print(f"Tested {model_name}:{variant_name} ({len(strategies)} policies)")

    score_matrix = np.vstack(totals)
    nested = nested_choices(score_matrix, seasons)
    evaluation_indices = range(len(lens.TRAINING_SEASONS), len(seasons))
    target_payload = json.loads(
        (lens.ROOT / "analysis" / "data" / "historical_rank_benchmarks.json").read_text(
            encoding="utf-8"
        )
    )
    targets = {
        item["season"].replace("/", "-"): int(item["points"])
        for item in target_payload["seasons"]
    }
    for row, season_index in zip(nested, evaluation_indices):
        selected = row["selected"]
        row.update(combinations[selected])
        row["target"] = targets[seasons[season_index]]
        row["margin"] = row["points"] - row["target"]
        stats = transfer_stats[selected][season_index]
        row["transfers"] = stats["transfers"]
        row["rolled"] = stats["rolled"]
        del row["selected"]

    fixed_standardised = (
        score_matrix[:, : len(lens.TRAINING_SEASONS)]
        - score_matrix[:, : len(lens.TRAINING_SEASONS)].mean(axis=0)
    ) / score_matrix[:, : len(lens.TRAINING_SEASONS)].std(axis=0).clip(1e-6)
    fixed_score = fixed_standardised.mean(axis=1) - 0.25 * fixed_standardised.std(axis=1)
    fixed_index = int(np.argmax(fixed_score))
    fixed_eval = score_matrix[fixed_index, len(lens.TRAINING_SEASONS) :]
    result = {
        "combinations": len(combinations),
        "fixedTrainingOnlyWinner": combinations[fixed_index],
        "fixedEvaluationPoints": [round(float(value)) for value in fixed_eval],
        "fixedAverage": round(float(fixed_eval.mean()), 1),
        "nestedSeasons": nested,
        "nestedAverage": round(float(np.mean([row["points"] for row in nested])), 1),
        "nestedTargetHits": sum(row["margin"] >= 0 for row in nested),
        "nestedAverageMargin": round(float(np.mean([row["margin"] for row in nested])), 1),
    }
    output = lens.ROOT / "analysis" / "data" / "elite_policy_search.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
