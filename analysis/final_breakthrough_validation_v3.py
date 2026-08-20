"""Reproduce the fully integrated captain + chip breakthrough candidate.

This is the final audit boundary: it compares every promoted stage on the same
recursive replay and reconstructs the top-500k margin without using that target
to select any policy.
"""

from __future__ import annotations

import json

import numpy as np

import calibrate_model as lens
from captain_fixture_history_validation import add_fixture_history
from captain_route_consensus_validation import selected_consensus_metric, weekly_percentile
from dynamic_match_model_v2 import build_dynamic_history
from forecast_champion_v2 import (
    BENCH_BOOST_THRESHOLD,
    FREE_HIT_RISK_DISCOUNT,
    FREE_HIT_THRESHOLD,
    MODEL_ID,
    TRIPLE_CAPTAIN_THRESHOLD,
    selected_forecasts,
)
from forecast_layer_v2 import dynamic_route_score
from freehit_value_validation import causal_predictions, opportunity_frame
from frontier_ranker_validation import STRATEGY
from multiscale_horizon_validation import add_targets
from probabilistic_minutes_validation import season_summary
from wildcard_freehit_ablation import champion_forecasts


def top500_targets() -> dict[str, float]:
    path = lens.ROOT / "analysis" / "data" / "breakthrough_benchmark.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(row["season"]).replace("/", "-"): float(row["top500Target"])
        for row in payload["control"]["seasons"]
    }


def stage_summary(totals: np.ndarray, seasons: list[str], targets: dict[str, float]) -> dict:
    result = season_summary(totals, seasons)
    evaluation = np.asarray(totals[2:], float)
    target = np.asarray([targets[season] for season in lens.EVALUATION_SEASONS], float)
    margins = evaluation - target
    result.update(
        {
            "top500Pace": round(float(target.mean()), 1),
            "averageTop500Margin": round(float(margins.mean()), 1),
            "top500Hits": int((margins >= 0).sum()),
            "top500SeasonMargins": margins.astype(int).tolist(),
        }
    )
    return result


def selected_chip_inputs(data, immediate, plan, captain):
    free_hit_squads = lens.precompute_fresh_squads(data, immediate)
    collector = lens.ChipPolicy(
        1e6,
        1e6,
        1e6,
        1e6,
        0.0,
        10,
        28,
        ("Free Hit", "Bench Boost", "Triple Captain"),
    )
    no_chip, stats = lens.simulate_candidate(
        data,
        immediate,
        STRATEGY,
        chip_policy=collector,
        free_hit_squads=free_hit_squads,
        plan_scores=plan,
        captain_scores=captain,
    )
    seasons = list(dict.fromkeys(data["season"].tolist()))
    frame = opportunity_frame(stats, seasons)
    prediction, residual_scale, fit_audit = causal_predictions(frame)
    frame["fhSignal"] = (
        prediction
        - frame["permanentTransferValueForegone"].to_numpy(float)
        - FREE_HIT_RISK_DISCOUNT * residual_scale
    )
    overrides = {
        (str(row.season), int(row.gw), "Free Hit"): float(row.fhSignal)
        for row in frame.itertuples()
    }
    policy = lens.ChipPolicy(
        1e6,
        FREE_HIT_THRESHOLD,
        BENCH_BOOST_THRESHOLD,
        TRIPLE_CAPTAIN_THRESHOLD,
        0.0,
        10,
        28,
        ("Free Hit", "Bench Boost", "Triple Captain"),
    )
    return no_chip, free_hit_squads, overrides, policy, fit_audit


def main() -> None:
    dynamic, _ = build_dynamic_history()
    data = add_fixture_history(add_targets(dynamic.reset_index(drop=True)))
    immediate, plan, frozen_captain = champion_forecasts(data)
    route = selected_consensus_metric(data, immediate, frozen_captain)
    seasons = list(dict.fromkeys(data["season"].tolist()))
    targets = top500_targets()

    original, _ = lens.simulate_candidate(
        data,
        immediate,
        STRATEGY,
        chip_policy=lens.AUDITED_CHAMPION_CHIP_POLICY,
        plan_scores=plan,
        captain_scores=route,
    )
    previous_dynamic, _ = dynamic_route_score(data, immediate, 0.30)
    previous_captain = 0.90 * route + 0.10 * weekly_percentile(data, previous_dynamic)
    previous_v2, _ = lens.simulate_candidate(
        data,
        immediate,
        STRATEGY,
        chip_policy=lens.AUDITED_CHAMPION_CHIP_POLICY,
        plan_scores=plan,
        captain_scores=previous_captain,
    )
    selected_immediate, selected_plan, selected_captain = selected_forecasts(data)
    captain_only, _ = lens.simulate_candidate(
        data,
        selected_immediate,
        STRATEGY,
        chip_policy=lens.AUDITED_CHAMPION_CHIP_POLICY,
        plan_scores=selected_plan,
        captain_scores=selected_captain,
    )
    no_chip, free_hit_squads, overrides, chip_policy, fit_audit = selected_chip_inputs(
        data, selected_immediate, selected_plan, selected_captain
    )
    integrated, stats = lens.simulate_candidate(
        data,
        selected_immediate,
        STRATEGY,
        chip_policy=chip_policy,
        free_hit_squads=free_hit_squads,
        plan_scores=selected_plan,
        captain_scores=selected_captain,
        chip_value_overrides=overrides,
    )
    stages = {
        "originalRouteControl": stage_summary(original, seasons, targets),
        "previousForecastV2": stage_summary(previous_v2, seasons, targets),
        "newCaptainOldChips": stage_summary(captain_only, seasons, targets),
        "newCaptainNoChips": stage_summary(no_chip, seasons, targets),
        "fullyIntegrated": stage_summary(integrated, seasons, targets),
    }
    evaluation_delta = integrated[2:] - original[2:]
    prior_delta = integrated[2:] - previous_v2[2:]
    gate = {
        "beatsOriginalAverage": float(evaluation_delta.mean()) > 0,
        "beatsPreviousV2Average": float(prior_delta.mean()) > 0,
        "developmentPositive": float(evaluation_delta[:-2].mean()) > 0,
        "holdoutPositive": float(evaluation_delta[-2:].mean()) > 0,
        "noNegativeSeasonVsOriginal": bool((evaluation_delta >= 0).all()),
        "chipPolicyImprovesEverySeasonVsNoChip": bool(
            ((integrated[2:] - no_chip[2:]) > 0).all()
        ),
    }
    result = {
        "schemaVersion": 1,
        "modelId": MODEL_ID,
        "status": "fully integrated recursive research winner",
        "selectionProtocol": (
            "Captain and chip hyperparameters were selected by first-six-season "
            "development stability; 2024/25 and 2025/26 were used only for gates."
        ),
        "automaticWildcard": "rejected after all six exact recursive variants failed",
        "stages": stages,
        "integratedVsOriginal": {
            "averageDelta": round(float(evaluation_delta.mean()), 1),
            "developmentDelta": round(float(evaluation_delta[:-2].mean()), 1),
            "holdoutDelta": round(float(evaluation_delta[-2:].mean()), 1),
            "minimumDelta": int(evaluation_delta.min()),
            "seasonDeltas": evaluation_delta.astype(int).tolist(),
        },
        "integratedVsPreviousV2": {
            "averageDelta": round(float(prior_delta.mean()), 1),
            "seasonDeltas": prior_delta.astype(int).tolist(),
        },
        "chipChoices": [row["chips"] for row in stats[2:]],
        "freeHitFitAudit": fit_audit,
        "gate": gate,
        "passed": all(gate.values()),
        "promotionStatus": (
            "research shadow only: historical odds vintage is not promotion eligible "
            "and the model clears only 2/8 reconstructed top-500k cutoffs"
        ),
    }
    output = lens.ROOT / "analysis" / "data" / "final_breakthrough_validation_v3.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stages": stages, "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
