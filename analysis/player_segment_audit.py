"""Expose forecast weaknesses hidden by aggregate player-week metrics."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import calibrate_model as lens
from bench_efficiency_validation import championship_forecasts


def metrics(frame: pd.DataFrame) -> dict:
    error = frame["forecast"] - frame["points"]
    return {
        "rows": len(frame),
        "forecast": round(float(frame["forecast"].mean()), 3),
        "actual": round(float(frame["points"].mean()), 3),
        "bias": round(float(error.mean()), 3),
        "mae": round(float(error.abs().mean()), 3),
        "returnRate": round(float((frame["points"] >= 5).mean()), 3),
        "blankRate": round(float((frame["points"] <= 2).mean()), 3),
    }


def grouped(work: pd.DataFrame, column: str) -> list[dict]:
    return [
        {column: str(value), **metrics(frame)}
        for value, frame in work.groupby(column, observed=True, sort=False)
    ]


def main() -> None:
    data, _ = lens.load_or_build_prepared_history()
    data = data.reset_index(drop=True)
    score, _, _ = championship_forecasts(data)
    work = data[data["season"].isin(lens.EVALUATION_SEASONS)].copy()
    work = work[work["fixture_count"] > 0].copy()
    work["forecast"] = score[work.index]
    work["position"] = work["position_id"].map(
        {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    )
    work["seasonPhase"] = pd.cut(
        work["GW"], [0, 6, 19, 30, 50], labels=["GW1-6", "GW7-19", "GW20-30", "GW31+"]
    )
    work["minutesTier"] = pd.cut(
        work["start_probability"],
        [-np.inf, 0.35, 0.65, 0.82, np.inf],
        labels=["rotation", "uncertain", "likely", "nailed"],
    )
    price_m = work["price"] / 10.0
    premium_threshold = work["position_id"].map({1: 5.5, 2: 5.5, 3: 9.0, 4: 9.0})
    work["priceTier"] = np.where(price_m >= premium_threshold, "premium", "non-premium")
    work["popularityTier"] = pd.qcut(
        work["selected"].rank(method="first"), 4, labels=["low", "medium", "popular", "elite-owned"]
    )
    work["teamAttackTier"] = pd.qcut(
        work["team_attack_rating"].rank(method="first"),
        5,
        labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
    )
    work["teamDefenceTier"] = pd.qcut(
        work["team_clean_probability"].rank(method="first"),
        5,
        labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
    )
    defenders = work[work["position_id"] == 2]
    result = {
        "method": "Evaluation seasons only; positive bias means the causal forecast overpredicts realised FPL points.",
        "overall": metrics(work),
        "position": grouped(work, "position"),
        "seasonPhase": grouped(work, "seasonPhase"),
        "minutesTier": grouped(work, "minutesTier"),
        "priceTier": grouped(work, "priceTier"),
        "popularityTier": grouped(work, "popularityTier"),
        "defenderTeamStrength": grouped(defenders, "teamDefenceTier"),
        "attackerTeamStrength": grouped(
            work[work["position_id"].isin([3, 4])], "teamAttackTier"
        ),
    }
    output = lens.ROOT / "analysis" / "data" / "player_segment_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
