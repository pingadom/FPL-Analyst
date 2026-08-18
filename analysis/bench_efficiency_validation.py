"""Paired historical audit of the pre-fix and bench-efficient squad selectors."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

import calibrate_model as lens
from captain_ranker_validation import rank_blend
from frontier_ranker_validation import PLAYER_CANDIDATE, STRATEGY
from listwise_ranker_validation import quantile_map


def championship_forecasts(data):
    immediate, horizon, _ = lens.candidate_forecasts(
        data,
        PLAYER_CANDIDATE,
        robust_planning=False,
        schedule_censored=True,
    )
    stable_plan = 0.75 * immediate * 4.5 + 0.25 * horizon
    frontier_raw = np.load(lens.CACHE / "frontier-causal-predictions-v2.npz")[
        "prediction"
    ]
    horizon_raw = np.load(lens.CACHE / "listwise-horizon_target-v1.npz")[
        "prediction"
    ]
    captain_raw = np.load(lens.CACHE / "captain-listwise-v1.npz")["prediction"]
    immediate_mapped = quantile_map(data, frontier_raw, immediate)
    plan_mapped = quantile_map(data, horizon_raw, stable_plan)
    score = 0.75 * immediate + 0.25 * immediate_mapped
    plan = 0.75 * stable_plan + 0.25 * plan_mapped
    captain_score = rank_blend(data, immediate, captain_raw, 0.50)
    return score, plan, captain_score


def variant_summary(totals, stats, seasons):
    rows = []
    for index, season in enumerate(seasons):
        if season not in lens.EVALUATION_SEASONS:
            continue
        tracked = stats[index]["trackedPlayer"]
        eligible = int(tracked["eligibleWeeks"])
        allocation = stats[index]["allocation"]
        rows.append(
            {
                "season": season.replace("-", "/"),
                "points": int(round(float(totals[index]))),
                "initialSpend": allocation["initialSpend"],
                "initialBank": allocation["initialBank"],
                "initialBenchSpend": allocation["initialBenchSpend"],
                "initialBenchPremium": allocation["initialBenchPremium"],
                "averageBenchSpend": allocation["averageBenchSpend"],
                "averageBenchPremium": allocation["averageBenchPremium"],
                "averageBank": allocation["averageBank"],
                "transfers": stats[index]["transfers"],
                "rolled": stats[index]["rolled"],
                "weeksChanged": stats[index]["weeksChanged"],
                "weeklyPoints": stats[index]["weeklyPoints"],
                "initialSelection": stats[index]["initialSelection"],
                "salah": {
                    **tracked,
                    "squadRate": round(100 * tracked["squadWeeks"] / eligible, 1)
                    if eligible
                    else None,
                    "xiRate": round(100 * tracked["xiWeeks"] / eligible, 1)
                    if eligible
                    else None,
                    "captainRate": round(
                        100 * tracked["captainWeeks"] / eligible, 1
                    )
                    if eligible
                    else None,
                },
            }
        )
    eligible_total = sum(row["salah"]["eligibleWeeks"] for row in rows)
    return {
        "average": round(float(np.mean([row["points"] for row in rows])), 1),
        "minimum": min(row["points"] for row in rows),
        "total": sum(row["points"] for row in rows),
        "averageInitialSpend": round(
            float(np.mean([row["initialSpend"] for row in rows])), 2
        ),
        "averageInitialBenchSpend": round(
            float(np.mean([row["initialBenchSpend"] for row in rows])), 2
        ),
        "averageBenchSpend": round(
            float(np.mean([row["averageBenchSpend"] for row in rows])), 2
        ),
        "salah": {
            "initialSquadSeasons": sum(row["salah"]["initialSquad"] for row in rows),
            "majoritySquadSeasons": sum(
                (row["salah"]["squadRate"] or 0) >= 50 for row in rows
            ),
            "majorityXiSeasons": sum(
                (row["salah"]["xiRate"] or 0) >= 50 for row in rows
            ),
            "overallSquadRate": round(
                100
                * sum(row["salah"]["squadWeeks"] for row in rows)
                / eligible_total,
                1,
            ),
            "overallXiRate": round(
                100 * sum(row["salah"]["xiWeeks"] for row in rows) / eligible_total,
                1,
            ),
            "overallCaptainRate": round(
                100
                * sum(row["salah"]["captainWeeks"] for row in rows)
                / eligible_total,
                1,
            ),
        },
        "seasons": rows,
    }


def main() -> None:
    data, _ = lens.load_or_build_prepared_history()
    data = data.reset_index(drop=True)
    scores, plan_scores, captain_scores = championship_forecasts(data)
    seasons = list(dict.fromkeys(data["season"].tolist()))

    before_totals, before_stats = lens.simulate_candidate(
        data,
        scores,
        STRATEGY,
        plan_scores=plan_scores,
        captain_scores=captain_scores,
        tracked_player_name="Salah",
    )
    corrected_strategy = replace(
        STRATEGY,
        name="Audited stable joint planner + bench efficiency",
        initial_spend_gap=5,
        bench_premium_limit=20,
        bench_premium_penalty=0.022,
    )
    after_totals, after_stats = lens.simulate_candidate(
        data,
        scores,
        corrected_strategy,
        plan_scores=plan_scores,
        captain_scores=captain_scores,
        tracked_player_name="Salah",
    )
    before = variant_summary(before_totals, before_stats, seasons)
    after = variant_summary(after_totals, after_stats, seasons)
    paired = []
    for old, new in zip(before["seasons"], after["seasons"]):
        paired.append(
            {
                "season": old["season"],
                "before": old["points"],
                "after": new["points"],
                "delta": new["points"] - old["points"],
                "salahSquadRateBefore": old["salah"]["squadRate"],
                "salahSquadRateAfter": new["salah"]["squadRate"],
                "salahXiRateAfter": new["salah"]["xiRate"],
                "salahCaptainRateAfter": new["salah"]["captainRate"],
            }
        )

    result = {
        "status": "research-only; historically exposed",
        "comparison": "Same causal forecasts, captain model, prices, transfer rules and realised points; only initial/wildcard spend discipline and bench-premium valuation change.",
        "before": before,
        "after": after,
        "lift": {
            "average": round(after["average"] - before["average"], 1),
            "totalAcrossEightSeasons": after["total"] - before["total"],
            "improvedSeasons": sum(row["delta"] > 0 for row in paired),
            "unchangedSeasons": sum(row["delta"] == 0 for row in paired),
            "worseSeasons": sum(row["delta"] < 0 for row in paired),
        },
        "pairedSeasons": paired,
        "policy": {
            "minimumFreshSpend": "available budget minus £0.5m",
            "maximumBenchPremium": "£2.0m above positional minimums",
            "benchPremiumPenalty": "0.022 planning-score units per £0.1m",
            "captain": "Separate causal captain rank; captain must be in the XI",
        },
    }
    for row in after["seasons"]:
        if row["initialSpend"] < 99.5:
            raise AssertionError(
                f"{row['season']} corrected initial spend is only £{row['initialSpend']}m"
            )
        if row["initialBenchPremium"] > 2.0:
            raise AssertionError(
                f"{row['season']} corrected initial bench premium is £{row['initialBenchPremium']}m"
            )
    output = lens.ROOT / "analysis" / "data" / "bench_efficiency_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"before": before, "after": after, "lift": result["lift"]}, indent=2))


if __name__ == "__main__":
    main()
