"""Decompose the causal action signal by position, phase and confidence."""

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
from probabilistic_minutes_validation import causal_predictions as minute_predictions
from transfer_action_ranker_validation import CACHE_VERSION, SHIFTS
from wildcard_freehit_ablation import champion_forecasts


def consensus_parts(data: pd.DataFrame, champion_plan: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cached = np.load(lens.CACHE / f"transfer-action-ranker-v{CACHE_VERSION}.npz")
    mapped = np.column_stack(
        [quantile_map(data, cached["shifts"][:, index], champion_plan) for index in range(len(SHIFTS))]
    )
    delta = mapped - champion_plan[:, None]
    agreement = np.sign(delta[:, 0]) == np.sign(delta[:, 1])
    consensus = mapped.mean(axis=1)
    strength = np.min(np.abs(delta), axis=1)
    strength_rank = pd.Series(strength).groupby(
        [data["season"], data["GW"], data["position_id"]], sort=False
    ).rank(pct=True).to_numpy(float)
    return consensus, agreement, strength_rank


def development_stability(totals: np.ndarray) -> float:
    values = totals[2:8]
    return float(values.mean() - 0.25 * values.std())


def main() -> None:
    original, _ = lens.load_or_build_prepared_history()
    data = add_targets(original.reset_index(drop=True))
    immediate, champion_plan, captain = champion_forecasts(data)
    consensus, agreement, strength_rank = consensus_parts(data, champion_plan)
    events = event_number(data)
    position = data["position_id"].to_numpy(int)
    minute = minute_predictions(data)
    minute_agreement = (
        np.abs(minute["minutes"] - data["expected_minutes"].to_numpy(float)) <= 18
    )
    disagreement_rank = pd.Series(data["ensemble_disagreement"].to_numpy(float)).groupby(
        [data["season"], data["GW"], data["position_id"]], sort=False
    ).rank(pct=True).to_numpy(float)
    base = (events >= 13) & agreement
    masks = {
        "all": base,
        "noGK": base & (position != 1),
        "DEF": base & (position == 2),
        "MID": base & (position == 3),
        "FWD": base & (position == 4),
        "attackers": base & np.isin(position, [3, 4]),
        "lateNoGK": base & (events >= 25) & (position != 1),
        "earlyNoGK": base & (events < 25) & (position != 1),
        "confidentNoGK": (
            base
            & (position != 1)
            & minute_agreement
            & (disagreement_rank <= 0.75)
            & (strength_rank >= 0.50)
        ),
    }
    plans = {
        name: champion_plan + 0.05 * mask.astype(float) * (consensus - champion_plan)
        for name, mask in masks.items()
    }
    seasons = list(dict.fromkeys(data["season"].tolist()))
    base_totals, base_stats = lens.simulate_candidate(
        data, immediate, STRATEGY, plan_scores=champion_plan, captain_scores=captain,
        tracked_player_name="Salah",
    )
    baseline = variant_summary(base_totals, base_stats, seasons)
    rows = []
    for name, plan in plans.items():
        print(f"Action segment {name}", flush=True)
        totals, stats = lens.simulate_candidate(
            data, immediate, STRATEGY, plan_scores=plan, captain_scores=captain,
            tracked_player_name="Salah",
        )
        summary = variant_summary(totals, stats, seasons)
        deltas = [new["points"] - old["points"] for new, old in zip(summary["seasons"], baseline["seasons"])]
        rows.append({
            "name": name,
            "activePlayerWeeks": int(masks[name].sum()),
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
        "method": "Fixed 5% two-band action consensus decomposed by position, GW13/GW25 phase and independent minutes/ensemble confidence.",
        "baselineDevelopmentStability": round(base_stability, 3),
        "baselineHoldoutAverage": round(base_holdout, 1),
        "baseline": baseline,
        "variants": rows,
        "selectedByDevelopment": selected,
        "robustPromotion": robust,
    }
    output = lens.ROOT / "analysis" / "data" / "action_segment_validation.json"
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
