"""Test orthogonal immediate and transfer-horizon challengers together."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

import calibrate_model as lens
from frontier_ranker_validation import PLAYER_CANDIDATE, STRATEGY
from listwise_ranker_validation import quantile_map


def summarize(totals: np.ndarray, seasons: list[str], targets: dict[str, int], stats: list[dict]) -> dict:
    evaluation = []
    for index in range(2, len(seasons)):
        points = int(round(float(totals[index])))
        evaluation.append(
            {
                "season": seasons[index].replace("-", "/"),
                "points": points,
                "target": targets[seasons[index]],
                "margin": points - targets[seasons[index]],
                "transfers": stats[index]["transfers"],
                "rolled": stats[index]["rolled"],
            }
        )
    return {
        "average": round(float(totals[2:].mean()), 1),
        "minimum": int(round(float(totals[2:].min()))),
        "targetHits": sum(row["margin"] >= 0 for row in evaluation),
        "averageMargin": round(float(np.mean([row["margin"] for row in evaluation])), 1),
        "averageTransfers": round(float(np.mean([row["transfers"] for row in evaluation])), 1),
        "evaluation": evaluation,
    }


def main() -> None:
    data, _ = lens.load_or_build_prepared_history()
    data = data.reset_index(drop=True)
    immediate, horizon, _ = lens.candidate_forecasts(
        data, PLAYER_CANDIDATE, robust_planning=False, schedule_censored=True
    )
    stable_plan = 0.75 * immediate * 4.5 + 0.25 * horizon
    frontier_raw = np.load(lens.CACHE / "frontier-causal-predictions-v2.npz")["prediction"]
    list_immediate_raw = np.load(lens.CACHE / "listwise-points-v1.npz")["prediction"]
    list_plan_raw = np.load(lens.CACHE / "listwise-horizon_target-v1.npz")["prediction"]
    frontier_immediate = quantile_map(data, frontier_raw, immediate)
    list_immediate = quantile_map(data, list_immediate_raw, immediate)
    list_plan = quantile_map(data, list_plan_raw, stable_plan)
    score_frontier25 = 0.75 * immediate + 0.25 * frontier_immediate
    score_list40 = 0.60 * immediate + 0.40 * list_immediate
    plan_list25 = 0.75 * stable_plan + 0.25 * list_plan
    variants = {
        "stable": (immediate, stable_plan),
        "frontierImmediate25": (score_frontier25, stable_plan),
        "listImmediate40": (score_list40, stable_plan),
        "listPlan25": (immediate, plan_list25),
        "frontier25_listPlan25": (score_frontier25, plan_list25),
        "list40_listPlan25": (score_list40, plan_list25),
    }
    seasons = list(dict.fromkeys(data["season"].tolist()))
    benchmark = json.loads(
        (lens.ROOT / "analysis" / "data" / "historical_rank_benchmarks.json").read_text(encoding="utf-8")
    )
    targets = {row["season"].replace("/", "-"): int(row["points"]) for row in benchmark["seasons"]}
    models = {}
    for name, (score, plan) in variants.items():
        totals, stats = lens.simulate_candidate(data, score, STRATEGY, plan_scores=plan)
        models[name] = summarize(totals, seasons, targets, stats)
        print(name, models[name]["average"])
    best_name = max(models, key=lambda name: models[name]["average"])
    best_score, best_plan = variants[best_name]
    policies = {}
    for hurdle in (12.0, 16.0, 20.0):
        for captain_weight in (0.70, 1.00):
            strategy = replace(
                STRATEGY,
                name=f"hybrid-h{hurdle:.0f}-cap{captain_weight:.2f}",
                transfer_hurdle=hurdle,
                squad_captain_weight=captain_weight,
            )
            totals, stats = lens.simulate_candidate(data, best_score, strategy, plan_scores=best_plan)
            policies[strategy.name] = summarize(totals, seasons, targets, stats)
            print(strategy.name, policies[strategy.name]["average"])
    result = {
        "status": "challenger-only",
        "method": "Orthogonal fusion: causal frontier regression for the next deadline, causal LambdaMART for six-week transfer ordering, then a small predeclared policy interaction grid.",
        "bestForecastCombination": best_name,
        "forecastModels": models,
        "policyInteractions": policies,
    }
    output = lens.ROOT / "analysis" / "data" / "hybrid_decision_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best": best_name, "models": {name: {key: value for key, value in row.items() if key != "evaluation"} for name, row in models.items()}, "policies": {name: {key: value for key, value in row.items() if key != "evaluation"} for name, row in policies.items()}}, indent=2))


if __name__ == "__main__":
    main()
