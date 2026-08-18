"""Combine causal near-price action ranks with scenario-aware package utility."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import calibrate_model as lens
from bench_efficiency_validation import variant_summary
from frontier_ranker_validation import STRATEGY
from listwise_ranker_validation import quantile_map
from multiscale_horizon_validation import add_targets
from multiscale_phase_validation import event_number
from probabilistic_component_challenger import causal_route_predictions
from scenario_transfer_validation import CorrelatedScenarioGenerator, ScenarioTransferPolicy
from transfer_action_ranker_validation import (
    CACHE_VERSION,
    SHIFTS,
    agreed_action_plan,
    causal_action_predictions,
)
from wildcard_freehit_ablation import champion_forecasts


POLICIES = (
    ("none", 0.0, 0.0, 0.0),
    ("tail005", 0.0, 0.005, 0.0),
    ("consensus025", 0.0, 0.0, 0.25),
    ("win45", 0.45, 0.0, 0.0),
    ("win50", 0.50, 0.0, 0.0),
)


def action_plans(data: pd.DataFrame, champion_plan: np.ndarray, immediate: np.ndarray) -> dict[str, np.ndarray]:
    component, _ = causal_route_predictions(data, immediate)
    causal_action_predictions(data, champion_plan, component)
    cached = np.load(lens.CACHE / f"transfer-action-ranker-v{CACHE_VERSION}.npz")
    mapped_shifts = np.column_stack(
        [quantile_map(data, cached["shifts"][:, index], champion_plan) for index in range(len(SHIFTS))]
    )
    consensus = mapped_shifts.mean(axis=1)
    shift_delta = mapped_shifts - champion_plan[:, None]
    agreement = np.sign(shift_delta[:, 0]) == np.sign(shift_delta[:, 1])
    strength = np.min(np.abs(shift_delta), axis=1)
    strength_rank = pd.Series(strength).groupby(
        [data["season"], data["GW"], data["position_id"]], sort=False
    ).rank(pct=True).to_numpy(float)
    events = event_number(data)
    strong_active = 0.10 * (events >= 13) * agreement * (strength_rank >= 0.80)
    return {
        "consensus050": agreed_action_plan(data, champion_plan, 0.05),
        "strong80": champion_plan + strong_active * (consensus - champion_plan),
    }


def development_stability(totals: np.ndarray) -> float:
    values = totals[2:8]
    return float(values.mean() - 0.25 * values.std())


def main() -> None:
    original, _ = lens.load_or_build_prepared_history()
    data = add_targets(original.reset_index(drop=True))
    immediate, champion_plan, captain = champion_forecasts(data)
    plans = action_plans(data, champion_plan, immediate)
    seasons = list(dict.fromkeys(data["season"].tolist()))
    base_totals, base_stats = lens.simulate_candidate(
        data, immediate, STRATEGY, plan_scores=champion_plan, captain_scores=captain,
        tracked_player_name="Salah",
    )
    baseline = variant_summary(base_totals, base_stats, seasons)
    rows = []
    for plan_name, plan in plans.items():
        generator = CorrelatedScenarioGenerator(data, immediate, plan)
        for policy_name, threshold, tail_share, clear_penalty in POLICIES:
            name = f"{plan_name}-{policy_name}"
            print(f"Action/scenario combination {name}", flush=True)
            policy = None
            if policy_name != "none":
                policy = ScenarioTransferPolicy(
                    data, plan, generator, threshold, tail_share,
                    clear_penalty=clear_penalty,
                )
            totals, stats = lens.simulate_candidate(
                data, immediate, STRATEGY, plan_scores=plan, captain_scores=captain,
                tracked_player_name="Salah", package_action_adjustment=policy,
            )
            summary = variant_summary(totals, stats, seasons)
            deltas = [new["points"] - old["points"] for new, old in zip(summary["seasons"], baseline["seasons"])]
            rows.append({
                "name": name,
                "plan": plan_name,
                "scenarioPolicy": policy_name,
                "developmentStability": round(development_stability(totals), 3),
                "holdoutAverage": round(float(totals[8:].mean()), 1),
                "summary": summary,
                "averageDelta": round(summary["average"] - baseline["average"], 1),
                "minimumDelta": summary["minimum"] - baseline["minimum"],
                "improvedSeasons": int(sum(delta > 0 for delta in deltas)),
                "unchangedSeasons": int(sum(delta == 0 for delta in deltas)),
                "worseSeasons": int(sum(delta < 0 for delta in deltas)),
                "seasonDeltas": deltas,
            })
            print(name, rows[-1]["averageDelta"], deltas, flush=True)
    selected = max(rows, key=lambda row: row["developmentStability"])
    base_stability = development_stability(base_totals)
    base_holdout = float(base_totals[8:].mean())
    robust = bool(
        selected["developmentStability"] > base_stability
        and selected["holdoutAverage"] >= base_holdout + 5
        and selected["minimumDelta"] >= 0
        and selected["improvedSeasons"] >= 5
    )
    result = {
        "status": "promoted" if robust else "research-only; robust promotion gate failed",
        "method": "Two independent near-price action ranks must agree; correlated scenarios then softly rank or veto only the resulting legal transfer packages.",
        "baselineDevelopmentStability": round(base_stability, 3),
        "baselineHoldoutAverage": round(base_holdout, 1),
        "baseline": baseline,
        "variants": rows,
        "selectedByDevelopment": selected,
        "robustPromotion": robust,
    }
    output = lens.ROOT / "analysis" / "data" / "action_scenario_combination_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "selected": selected["name"],
        "average": selected["summary"]["average"],
        "minimum": selected["summary"]["minimum"],
        "holdout": selected["holdoutAverage"],
        "robustPromotion": robust,
    }, indent=2))


if __name__ == "__main__":
    main()
