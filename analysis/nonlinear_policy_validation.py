"""Run cached nonlinear forecasts through the full recursive FPL simulator."""

from __future__ import annotations

import json

import numpy as np

import calibrate_model as lens


PLAYER_CANDIDATE = lens.Candidate(
    0.32, 0.05, 0.00, 0.13, 0.19, 0.03, 0.18, 0.10, 0.76
)
STRATEGY = lens.SimulationStrategy(
    name="Audited stable joint planner",
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
    squad_bench_weight=0.05,
)
CHIP_POLICY = lens.ChipPolicy(60, 20, 11, 15, 0.55, 10, 28)


def plan_from_immediate(data, scores: np.ndarray) -> np.ndarray:
    baseline = data["component_xpts"].to_numpy(float)
    ratio = np.divide(
        scores,
        baseline,
        out=np.ones_like(scores),
        where=baseline > 0.20,
    )
    ratio = np.clip(ratio, 0.55, 1.55)
    horizon = data["component_horizon_censored"].to_numpy(float) * ratio
    return 0.75 * scores * 4.5 + 0.25 * horizon


def main() -> None:
    data, _ = lens.load_or_build_prepared_history()
    cached = np.load(lens.CACHE / "nonlinear-causal-predictions-v1.npz")
    baseline_scores, baseline_horizon, _ = lens.candidate_forecasts(
        data, PLAYER_CANDIDATE, robust_planning=False, schedule_censored=True
    )
    models = {
        "stableLinear": (
            baseline_scores,
            0.75 * baseline_scores * 4.5 + 0.25 * baseline_horizon,
        ),
        "randomForest": (
            cached["random_forest"],
            plan_from_immediate(data, cached["random_forest"]),
        ),
        "xgboost": (
            cached["xgboost"],
            plan_from_immediate(data, cached["xgboost"]),
        ),
        "causalTreeEnsemble": (
            cached["causal_ensemble"],
            plan_from_immediate(data, cached["causal_ensemble"]),
        ),
    }
    seasons = list(dict.fromkeys(data["season"].tolist()))
    model_totals: dict[str, np.ndarray] = {}
    model_stats: dict[str, list[dict]] = {}
    for name, (scores, plan) in models.items():
        totals, stats = lens.simulate_candidate(
            data, scores, STRATEGY, plan_scores=plan
        )
        model_totals[name] = totals
        model_stats[name] = stats
        print(f"Simulated {name}: {totals[2:].mean():.1f} evaluation average")

    ensemble_scores, ensemble_plan = models["causalTreeEnsemble"]
    fresh = lens.precompute_fresh_squads(data, ensemble_plan)
    free_hits = lens.precompute_fresh_squads(data, ensemble_scores)
    chip_totals, chip_stats = lens.simulate_candidate(
        data,
        ensemble_scores,
        STRATEGY,
        chip_policy=CHIP_POLICY,
        fresh_squads=fresh,
        free_hit_squads=free_hits,
        plan_scores=ensemble_plan,
    )
    print(f"Simulated tree ensemble with fixed chips: {chip_totals[2:].mean():.1f}")

    benchmarks = json.loads(
        (lens.ROOT / "analysis" / "data" / "historical_rank_benchmarks.json").read_text(
            encoding="utf-8"
        )
    )
    targets = {
        item["season"].replace("/", "-"): int(item["points"])
        for item in benchmarks["seasons"]
    }
    evaluations: dict[str, list[dict]] = {}
    all_totals = {**model_totals, "causalTreeEnsembleWithChips": chip_totals}
    for name, totals in all_totals.items():
        stats = chip_stats if name.endswith("WithChips") else model_stats[name]
        rows = []
        for season_index in range(2, len(seasons)):
            season = seasons[season_index]
            points = int(round(float(totals[season_index])))
            target = targets[season]
            rows.append(
                {
                    "season": season.replace("-", "/"),
                    "points": points,
                    "target": target,
                    "margin": points - target,
                    "transfers": stats[season_index]["transfers"],
                    "rolled": stats[season_index]["rolled"],
                    "chips": stats[season_index]["chips"],
                }
            )
        evaluations[name] = rows

    result = {
        "selection": "Forecast architecture and policy fixed before recursive evaluation",
        "strategy": STRATEGY.name,
        "chipPolicy": CHIP_POLICY.as_dict(),
        "models": {
            name: {
                "average": round(float(totals[2:].mean()), 1),
                "minimum": int(round(float(totals[2:].min()))),
                "targetHits": sum(row["margin"] >= 0 for row in evaluations[name]),
                "averageMargin": round(
                    float(np.mean([row["margin"] for row in evaluations[name]])), 1
                ),
                "evaluation": evaluations[name],
            }
            for name, totals in all_totals.items()
        },
    }
    output = lens.ROOT / "analysis" / "data" / "nonlinear_policy_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: value | {"evaluation": "omitted"} for name, value in result["models"].items()}, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
