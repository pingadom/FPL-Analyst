"""Validate premium access and captain ceiling without name-based forcing."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd

import calibrate_model as lens
from frontier_ranker_validation import STRATEGY
from multiscale_horizon_validation import add_targets
from probabilistic_minutes_validation import causal_predictions as minute_predictions
from probabilistic_minutes_validation import season_summary
from wildcard_freehit_ablation import champion_forecasts


def percentile(data: pd.DataFrame, values: np.ndarray, keys: list[str]) -> np.ndarray:
    return pd.Series(values, index=data.index).groupby([data[key] for key in keys]).rank(pct=True).to_numpy(float)


def captain_variants(data: pd.DataFrame, score: np.ndarray, frozen: np.ndarray) -> dict[str, np.ndarray]:
    minute = minute_predictions(data)
    immediate_rank = percentile(data, score, ["season", "GW"])
    frozen_rank = percentile(data, frozen, ["season", "GW"])
    haul_rank = percentile(data, data["haul8_probability"].to_numpy(float), ["season", "GW"])
    return_rank = percentile(data, data["return5_probability"].to_numpy(float), ["season", "GW"])
    goal_rank = percentile(data, data["goal_rate"].to_numpy(float), ["season", "GW"])
    attack_team_rank = percentile(data, data["team_attack_rating"].to_numpy(float), ["season", "GW"])
    price_rank = percentile(data, data["price"].to_numpy(float), ["season", "GW", "position_id"])
    ownership_rank = percentile(data, np.log1p(data["selected"].clip(lower=0).to_numpy(float)), ["season", "GW"])
    reliable = 0.45 * minute["play"] + 0.55 * minute["sixty"]
    reliable_rank = percentile(data, reliable, ["season", "GW"])
    ceiling = 0.44 * immediate_rank + 0.25 * haul_rank + 0.10 * return_rank + 0.10 * goal_rank + 0.06 * attack_team_rank + 0.05 * reliable_rank
    premium_ceiling = 0.78 * ceiling + 0.14 * price_rank + 0.08 * ownership_rank
    reliable_ceiling = 0.72 * ceiling + 0.28 * reliable_rank
    return {
        "frozen": frozen_rank,
        "structural": immediate_rank,
        "ceiling25": 0.75 * frozen_rank + 0.25 * ceiling,
        "ceiling50": 0.50 * frozen_rank + 0.50 * ceiling,
        "premiumCeiling25": 0.75 * frozen_rank + 0.25 * premium_ceiling,
        "premiumCeiling50": 0.50 * frozen_rank + 0.50 * premium_ceiling,
        "reliableCeiling25": 0.75 * frozen_rank + 0.25 * reliable_ceiling,
        "reliableCeiling50": 0.50 * frozen_rank + 0.50 * reliable_ceiling,
    }


def main() -> None:
    original, _ = lens.load_or_build_prepared_history()
    data = add_targets(original.reset_index(drop=True))
    immediate, plan, captain = champion_forecasts(data)
    variants = captain_variants(data, immediate, captain)
    seasons = list(dict.fromkeys(data["season"].tolist()))
    base_totals, _ = lens.simulate_candidate(data, immediate, STRATEGY, plan_scores=plan, captain_scores=captain)
    baseline = season_summary(base_totals, seasons)
    rows = []
    for name, captain_score in variants.items():
        totals, _ = lens.simulate_candidate(data, immediate, STRATEGY, plan_scores=plan, captain_scores=captain_score)
        summary = season_summary(totals, seasons)
        deltas = [row["points"] - old["points"] for row, old in zip(summary["seasons"], baseline["seasons"])]
        rows.append({
            "name": name, **summary,
            "averageDelta": round(summary["average"] - baseline["average"], 1),
            "developmentDelta": round(summary["developmentAverage"] - baseline["developmentAverage"], 1),
            "holdoutDelta": round(summary["holdoutAverage"] - baseline["holdoutAverage"], 1),
            "improvedSeasons": int(sum(delta > 0 for delta in deltas)),
            "worstSeasonDelta": int(min(deltas)),
        })
        print("captain", name, rows[-1]["average"], deltas, flush=True)

    # Separately test whether small, generic premium-access utility helps the
    # transfer path.  This depends on price, forecast rank and armband ceiling;
    # it never names or forces Salah, Haaland, Fernandes or any other player.
    price_m = data["price"].to_numpy(float) / 10
    premium = ((data["position_id"].to_numpy(int) == 3) & (price_m >= 9.0)) | ((data["position_id"].to_numpy(int) == 4) & (price_m >= 9.0))
    forecast_rank = percentile(data, immediate, ["season", "GW", "position_id"])
    armband_rank = variants["premiumCeiling25"]
    access_signal = premium.astype(float) * np.clip((forecast_rank - 0.75) / 0.25, 0, 1) * np.clip((armband_rank - 0.70) / 0.30, 0, 1)
    access_rows = []
    for strength in [0.25, 0.50, 0.75, 1.00]:
        plan_score = plan + strength * access_signal
        strategy = replace(STRATEGY, name=f"premium-access-{strength:.2f}", align_captain_objective=True)
        totals, _ = lens.simulate_candidate(data, immediate, strategy, plan_scores=plan_score, captain_scores=variants["premiumCeiling25"])
        summary = season_summary(totals, seasons)
        deltas = [row["points"] - old["points"] for row, old in zip(summary["seasons"], baseline["seasons"])]
        access_rows.append({
            "name": strategy.name, **summary,
            "averageDelta": round(summary["average"] - baseline["average"], 1),
            "developmentDelta": round(summary["developmentAverage"] - baseline["developmentAverage"], 1),
            "holdoutDelta": round(summary["holdoutAverage"] - baseline["holdoutAverage"], 1),
            "improvedSeasons": int(sum(delta > 0 for delta in deltas)),
            "worstSeasonDelta": int(min(deltas)),
        })
        print("premium access", strength, access_rows[-1]["average"], deltas, flush=True)

    all_rows = rows + access_rows
    eligible = [row for row in all_rows if row["name"] != "frozen" and row["developmentDelta"] > 0 and row["holdoutDelta"] >= 5 and row["worstSeasonDelta"] >= 0 and row["improvedSeasons"] >= 5]
    selected = max(eligible, key=lambda row: (row["holdoutDelta"], row["developmentDelta"])) if eligible else None
    result = {
        "status": "promoted" if selected else "research-only; robust promotion gate failed",
        "method": "Generic price/forecast/ceiling premium access plus probabilistic-minutes captain ranking; no named-player constraints.",
        "baseline": baseline,
        "captainVariants": rows,
        "premiumAccessVariants": access_rows,
        "selected": selected,
    }
    output = lens.ROOT / "analysis" / "data" / "premium_captain_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"baseline": baseline, "selected": selected}, indent=2))


if __name__ == "__main__":
    main()
