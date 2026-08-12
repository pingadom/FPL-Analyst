"""Decision validation for direct six-GW forecasts with stable one-GW scores."""

from __future__ import annotations

import json

import numpy as np

import calibrate_model as lens
from nonlinear_policy_validation import PLAYER_CANDIDATE, STRATEGY


def main() -> None:
    data, _ = lens.load_or_build_prepared_history()
    current, horizon, _ = lens.candidate_forecasts(
        data, PLAYER_CANDIDATE, robust_planning=False, schedule_censored=True
    )
    stable_plan = 0.75 * current * 4.5 + 0.25 * horizon
    cached = np.load(lens.CACHE / "nonlinear-horizon-predictions-v1.npz")
    direct = {
        "ridge": cached["ridge"],
        "randomForest": cached["random_forest"],
        "xgboost": cached["xgboost"],
        "treeEnsemble": cached["causal_ensemble"],
    }
    plans: dict[str, np.ndarray] = {"stablePlan": stable_plan}
    for name, values in direct.items():
        plans[f"direct-{name}"] = values
        plans[f"halfStable-half{name}"] = 0.5 * stable_plan + 0.5 * values

    seasons = list(dict.fromkeys(data["season"].tolist()))
    totals_by_name: dict[str, np.ndarray] = {}
    stats_by_name: dict[str, list[dict]] = {}
    for name, plan in plans.items():
        totals, stats = lens.simulate_candidate(
            data, current, STRATEGY, plan_scores=plan
        )
        totals_by_name[name] = totals
        stats_by_name[name] = stats
        print(f"{name}: {totals[2:].mean():.1f}")

    benchmark = json.loads(
        (lens.ROOT / "analysis" / "data" / "historical_rank_benchmarks.json").read_text(
            encoding="utf-8"
        )
    )
    targets = {
        row["season"].replace("/", "-"): int(row["points"])
        for row in benchmark["seasons"]
    }
    models = {}
    for name, totals in totals_by_name.items():
        evaluation = []
        for season_index in range(2, len(seasons)):
            season = seasons[season_index]
            points = int(round(float(totals[season_index])))
            evaluation.append(
                {
                    "season": season.replace("-", "/"),
                    "points": points,
                    "target": targets[season],
                    "margin": points - targets[season],
                    "transfers": stats_by_name[name][season_index]["transfers"],
                    "rolled": stats_by_name[name][season_index]["rolled"],
                }
            )
        models[name] = {
            "average": round(float(totals[2:].mean()), 1),
            "minimum": int(round(float(totals[2:].min()))),
            "targetHits": sum(row["margin"] >= 0 for row in evaluation),
            "averageMargin": round(float(np.mean([row["margin"] for row in evaluation])), 1),
            "evaluation": evaluation,
        }
    result = {
        "selection": "All architectures specified before recursive policy evaluation",
        "immediateForecast": "stableLinear",
        "strategy": STRATEGY.name,
        "models": models,
    }
    output = lens.ROOT / "analysis" / "data" / "horizon_policy_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: {k: v for k, v in item.items() if k != "evaluation"} for name, item in models.items()}, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
