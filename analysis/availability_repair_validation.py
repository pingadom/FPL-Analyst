"""Like-for-like ablation of the held-player/availability repairs."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

import calibrate_model as lens
from elite_policy_search import candidate_from_weights
from held_player_audit import chip_policy_from_artifact, strategy_from_artifact


OUTPUT = lens.ROOT / "analysis" / "data" / "availability_repair_validation.json"


def main() -> None:
    data, _ = lens.load_or_build_prepared_history()
    artifact = json.loads(lens.OUTPUT.read_text(encoding="utf-8-sig"))
    candidate = candidate_from_weights(artifact["model"]["weights"])
    immediate, plan, _ = lens.candidate_forecasts(
        data,
        candidate,
        robust_planning=bool(artifact["model"].get("robustPlanningEnabled", False)),
    )
    chip_policy = chip_policy_from_artifact(artifact["chipStrategy"]["policy"])
    active = strategy_from_artifact(str(artifact["model"]["strategy"]))
    print("Precomputing exact persistent squads...", flush=True)
    fresh = lens.precompute_fresh_squads(data, plan)
    print("Precomputing exact one-week squads...", flush=True)
    free_hit = lens.precompute_fresh_squads(data, immediate, one_week_only=True)
    variants = [
        (
            "selected high-precision availability repair",
            replace(
                active,
                enforce_weekly_xi_floor=False,
                consistent_transfer_objective=False,
            ),
        ),
        (
            "rejected high-precision + penalised transfer hurdle",
            replace(
                active,
                enforce_weekly_xi_floor=False,
                consistent_transfer_objective=True,
            ),
        ),
    ]
    seasons = list(dict.fromkeys(data["season"].tolist()))
    evaluation_indices = [seasons.index(season) for season in lens.EVALUATION_SEASONS]
    rows = []
    for label, strategy in variants:
        totals, stats = lens.simulate_candidate(
            data,
            immediate,
            strategy,
            chip_policy=chip_policy,
            fresh_squads=fresh,
            free_hit_squads=free_hit,
            plan_scores=plan,
        )
        season_points = [int(totals[index]) for index in evaluation_indices]
        rows.append(
            {
                "variant": label,
                "averagePoints": round(float(np.mean(season_points)), 1),
                "seasonPoints": season_points,
                "averageTransfers": round(
                    float(
                        np.mean(
                            [
                                stat["transfers"]
                                for stat in stats
                                if stat["season"] in lens.EVALUATION_SEASONS
                            ]
                        )
                    ),
                    2,
                ),
            }
        )
        print(rows[-1], flush=True)
    result = {
        "status": "fixed-final-policy causal repair ablation",
        "preRepairAveragePoints": 2096.6,
        "warning": "This fixed-policy diagnostic is not the published season-by-season walk-forward score.",
        "rejectedBroadSignalRun": [
            {"variant": "broad xP-zero availability evidence", "averagePoints": 2066.8},
            {"variant": "broad evidence + 78% weekly XI floor", "averagePoints": 2050.5},
            {"variant": "broad evidence + penalised transfer hurdle", "averagePoints": 2073.6},
            {"variant": "broad evidence + both overcorrections", "averagePoints": 2057.2},
        ],
        "variants": rows,
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
