"""Test tree ordering while retaining the structural model's FPL utility scale."""

from __future__ import annotations

import json

import numpy as np

import calibrate_model as lens
from nonlinear_policy_validation import PLAYER_CANDIDATE, STRATEGY


def quantile_map_by_deadline(
    data, challenger: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    """Give challenger ranks the reference distribution within position/GW."""
    mapped = reference.copy()
    for indices in data.groupby(["season", "GW", "position_id"], sort=False).groups.values():
        local = np.asarray(indices, dtype=int)
        challenger_order = local[np.argsort(challenger[local], kind="stable")]
        reference_values = np.sort(reference[local])
        mapped[challenger_order] = reference_values
    return mapped


def main() -> None:
    data, _ = lens.load_or_build_prepared_history()
    current, horizon, _ = lens.candidate_forecasts(
        data, PLAYER_CANDIDATE, robust_planning=False, schedule_censored=True
    )
    stable_plan = 0.75 * current * 4.5 + 0.25 * horizon
    one_week = np.load(lens.CACHE / "nonlinear-causal-predictions-v1.npz")[
        "causal_ensemble"
    ]
    six_week = np.load(lens.CACHE / "nonlinear-horizon-predictions-v1.npz")[
        "causal_ensemble"
    ]
    mapped_current = quantile_map_by_deadline(data, one_week, current)
    mapped_plan = quantile_map_by_deadline(data, six_week, stable_plan)
    variants: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "stable": (current, stable_plan),
    }
    for share in (0.25, 0.50, 0.75, 1.00):
        label = f"treeImmediate{int(share * 100)}"
        variants[label] = (
            (1 - share) * current + share * mapped_current,
            stable_plan,
        )
        label = f"treePlan{int(share * 100)}"
        variants[label] = (
            current,
            (1 - share) * stable_plan + share * mapped_plan,
        )

    seasons = list(dict.fromkeys(data["season"].tolist()))
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
    for name, (scores, plan) in variants.items():
        totals, stats = lens.simulate_candidate(
            data, scores, STRATEGY, plan_scores=plan
        )
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
                    "transfers": stats[season_index]["transfers"],
                    "rolled": stats[season_index]["rolled"],
                }
            )
        models[name] = {
            "average": round(float(totals[2:].mean()), 1),
            "minimum": int(round(float(totals[2:].min()))),
            "targetHits": sum(row["margin"] >= 0 for row in evaluation),
            "averageMargin": round(float(np.mean([row["margin"] for row in evaluation])), 1),
            "evaluation": evaluation,
        }
        print(f"{name}: {models[name]['average']:.1f}")

    result = {
        "method": "Tree ranks quantile-mapped to the stable model inside each deadline and position",
        "selection": "Blend grid specified before recursive policy evaluation",
        "models": models,
    }
    output = lens.ROOT / "analysis" / "data" / "rank_calibrated_tree_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: {k: v for k, v in item.items() if k != "evaluation"} for name, item in models.items()}, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
