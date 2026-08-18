"""Validate when a causal multi-timescale planner should become influential."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

import calibrate_model as lens
from bench_efficiency_validation import championship_forecasts, variant_summary
from frontier_ranker_validation import STRATEGY
from listwise_ranker_validation import quantile_map
from multiscale_horizon_validation import (
    adaptive_value,
    add_targets,
    causal_online_ridge_horizons,
    causal_ridge_horizons,
    structural_horizons,
)


def event_number(data) -> np.ndarray:
    result = np.zeros(len(data), dtype=int)
    for _, season_frame in data.groupby("season", sort=False):
        weeks = list(dict.fromkeys(season_frame["GW"].astype(int).tolist()))
        number = {gw: index + 1 for index, gw in enumerate(weeks)}
        result[season_frame.index.to_numpy(int)] = (
            season_frame["GW"].astype(int).map(number).to_numpy(int)
        )
    return result


def phased_plan(
    baseline: np.ndarray,
    challenger: np.ndarray,
    events: np.ndarray,
    start: int,
    share: float,
) -> np.ndarray:
    active_share = np.where(events >= start, share, 0.0)
    return (1 - active_share) * baseline + active_share * challenger


def main() -> None:
    original, _ = lens.load_or_build_prepared_history()
    data = add_targets(original.reset_index(drop=True))
    scores, baseline_plan, captain = championship_forecasts(data)
    structural = structural_horizons(data, scores)
    learned, _ = causal_ridge_horizons(data, structural)
    online, online_audit = causal_online_ridge_horizons(data, structural, learned)
    learned_value = quantile_map(
        data, adaptive_value(data, learned, 3.0), baseline_plan
    )
    online_value = quantile_map(
        data, adaptive_value(data, online, 3.0), baseline_plan
    )
    events = event_number(data)

    plans = {
        "baseline": baseline_plan,
        "learned5AfterGW5": phased_plan(
            baseline_plan, learned_value, events, 5, 0.05
        ),
        "online5AfterGW13": phased_plan(
            baseline_plan, online_value, events, 13, 0.05
        ),
        "online10AfterGW13": phased_plan(
            baseline_plan, online_value, events, 13, 0.10
        ),
        "online10AfterGW25": phased_plan(
            baseline_plan, online_value, events, 25, 0.10
        ),
    }
    ramp_share = np.where(events < 5, 0.0, np.where(events < 13, 0.05, 0.10))
    plans["causalRamp0to5to10"] = (
        (1 - ramp_share) * baseline_plan + ramp_share * online_value
    )

    seasons = list(dict.fromkeys(data["season"].tolist()))
    rows = []
    for name, plan in plans.items():
        for hurdle in ((16.0, 18.0, 20.0) if name == "causalRamp0to5to10" else (16.0,)):
            experiment_name = name if hurdle == 16 else f"{name}H{int(hurdle)}"
            print(f"Running {experiment_name}", flush=True)
            strategy = replace(
                STRATEGY,
                name=f"Multi-timescale phase gate {experiment_name}",
                transfer_hurdle=hurdle,
            )
            totals, stats = lens.simulate_candidate(
                data,
                scores,
                strategy,
                plan_scores=plan,
                captain_scores=captain,
                tracked_player_name="Salah",
            )
            training = totals[: len(lens.TRAINING_SEASONS)]
            rows.append(
                {
                    "name": experiment_name,
                    "hurdle": hurdle,
                    "trainingStability": round(
                        float(training.mean() - 0.25 * training.std()), 3
                    ),
                    "summary": variant_summary(totals, stats, seasons),
                }
            )

    baseline = next(row for row in rows if row["name"] == "baseline")
    selected = max(rows, key=lambda row: row["trainingStability"])
    paired = []
    for old, new in zip(
        baseline["summary"]["seasons"], selected["summary"]["seasons"]
    ):
        paired.append(
            {
                "season": old["season"],
                "baseline": old["points"],
                "challenger": new["points"],
                "delta": new["points"] - old["points"],
            }
        )
    robust = bool(
        selected["name"] != "baseline"
        and selected["summary"]["average"] > baseline["summary"]["average"]
        and selected["summary"]["minimum"] >= baseline["summary"]["minimum"]
        and sum(row["delta"] > 0 for row in paired) >= 5
    )
    result = {
        "status": "phase-gated multi-timescale challenger",
        "method": (
            "The opening squad remains on the established structural/listwise "
            "planner. Multi-horizon influence begins only after causal evidence "
            "from the current season is available; one ramp and two conservative "
            "transfer hurdles are tested."
        ),
        "onlineFits": online_audit,
        "selected": selected,
        "pairedVsBaseline": paired,
        "robustPromotion": robust,
        "experiments": rows,
    }
    output = lens.ROOT / "analysis" / "data" / "multiscale_phase_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "selected": selected["name"],
                "robustPromotion": robust,
                "paired": paired,
                "experiments": [
                    {
                        "name": row["name"],
                        "trainingStability": row["trainingStability"],
                        "average": row["summary"]["average"],
                        "minimum": row["summary"]["minimum"],
                    }
                    for row in rows
                ],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
