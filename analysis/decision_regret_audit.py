"""Decompose FPL Lens decision regret into forecast and optimisation layers."""

from __future__ import annotations

import json

import numpy as np

import calibrate_model as lens
from elite_policy_search import candidate_from_weights


def realised(
    indices: list[int],
    frame,
    selection_scores: np.ndarray,
    captain_scores: np.ndarray,
    actual: np.ndarray,
    minutes: np.ndarray,
) -> tuple[float, float, float]:
    state = {
        int(frame.loc[index, "element"]): {
            "position": int(frame.loc[index, "position_id"]),
            "team": int(frame.loc[index, "team_id"]),
            "purchase": int(frame.loc[index, "price"]),
            "last_price": int(frame.loc[index, "price"]),
        }
        for index in indices
    }
    row_by_element = {
        int(frame.loc[index, "element"]): int(index) for index in frame.index
    }
    xi, bench = lens.choose_xi(state, row_by_element, selection_scores)
    captain_order = sorted(
        xi, key=lambda element: captain_scores[row_by_element[element]], reverse=True
    )
    breakdown = lens.realised_week_breakdown(
        xi,
        bench,
        captain_order[0],
        captain_order[1],
        state,
        row_by_element,
        actual,
        minutes,
    )
    actual_xi, actual_bench = lens.choose_xi(state, row_by_element, actual)
    actual_captains = sorted(
        actual_xi, key=lambda element: actual[row_by_element[element]], reverse=True
    )
    xi_oracle = lens.realised_week_breakdown(
        actual_xi,
        actual_bench,
        actual_captains[0],
        actual_captains[1],
        state,
        row_by_element,
        actual,
        minutes,
    )["normal"]
    captain_oracle = lens.realised_week_breakdown(
        xi,
        bench,
        max(xi, key=lambda element: actual[row_by_element[element]]),
        captain_order[1],
        state,
        row_by_element,
        actual,
        minutes,
    )["normal"]
    return breakdown["normal"], xi_oracle, captain_oracle


def main() -> None:
    data, _ = lens.load_or_build_prepared_history()
    artifact = json.loads(lens.OUTPUT.read_text(encoding="utf-8-sig"))
    candidate = candidate_from_weights(artifact["model"]["weights"])
    scores, plan, model_score = lens.candidate_forecasts(
        data, candidate, robust_planning=False
    )
    calibration = 0.72 + 0.56 * model_score
    non_role_weight = (1 - data["ensemble_role_weight"]).clip(lower=0.05)
    no_role_component = (
        data["component_xpts"]
        - data["ensemble_role_weight"] * data["role_ridge_xpts"]
    ) / non_role_weight
    variants = {
        "lens7_role": (scores, plan),
        "lens7_no_role": (
            no_role_component.to_numpy(float) * calibration,
            data["component_horizon_censored"].to_numpy(float)
            * (no_role_component / data["component_xpts"].clip(lower=0.2)).to_numpy(float)
            * calibration,
        ),
    }
    actual = data["points"].to_numpy(float)
    minutes = data["minutes"].to_numpy(float)
    benchmark_payload = json.loads(
        (lens.ROOT / "analysis" / "data" / "historical_rank_benchmarks.json").read_text(
            encoding="utf-8"
        )
    )
    targets = {
        item["season"].replace("/", "-"): int(item["points"])
        for item in benchmark_payload["seasons"]
    }
    output: dict[str, list[dict]] = {}
    for variant_name, (variant_scores, variant_plan) in variants.items():
        managed_strategy = lens.SimulationStrategy(
            "No-transfer set-and-forget", 999.0, 5, False, False, 0, 999.0
        )
        set_forget, _ = lens.simulate_candidate(
            data, variant_scores, managed_strategy, plan_scores=variant_plan
        )
        rows = []
        for season_index, (season, season_frame) in enumerate(
            data.groupby("season", sort=False)
        ):
            weekly_fresh = 0.0
            weekly_xi_oracle = 0.0
            weekly_captain_oracle = 0.0
            hindsight = 0.0
            for _, frame in season_frame.groupby("GW", sort=False):
                predicted = lens.initial_squad(frame, variant_scores)
                score, xi_oracle, captain_oracle = realised(
                    predicted,
                    frame,
                    variant_scores,
                    variant_scores,
                    actual,
                    minutes,
                )
                weekly_fresh += score
                weekly_xi_oracle += xi_oracle
                weekly_captain_oracle += captain_oracle
                actual_squad = lens.initial_squad(frame, actual)
                hindsight_score, _, _ = realised(
                    actual_squad, frame, actual, actual, actual, minutes
                )
                hindsight += hindsight_score
            if season not in lens.EVALUATION_SEASONS:
                continue
            target = targets[str(season)]
            rows.append(
                {
                    "season": str(season).replace("-", "/"),
                    "target": target,
                    "setAndForget": round(float(set_forget[season_index])),
                    "unlimitedWeeklyRebuild": round(weekly_fresh),
                    "lineupCaptainOracleOnModel15": round(weekly_xi_oracle),
                    "captainOracleOnModelXI": round(weekly_captain_oracle),
                    "fullHindsightUpperBound": round(hindsight),
                    "forecastCeilingMargin": round(weekly_fresh - target),
                    "lineupRegret": round(weekly_xi_oracle - weekly_fresh),
                    "captainRegret": round(weekly_captain_oracle - weekly_fresh),
                }
            )
        output[variant_name] = rows
        print(variant_name, json.dumps(rows, indent=2))
    result = {
        "method": "No chips. Unlimited weekly rebuild isolates the player-ranking ceiling; oracle XI/captain retains the model-selected 15; full hindsight is an intentionally unattainable upper bound.",
        "variants": output,
    }
    target = lens.ROOT / "analysis" / "data" / "decision_regret_audit.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {target.relative_to(lens.ROOT)}")


if __name__ == "__main__":
    main()
