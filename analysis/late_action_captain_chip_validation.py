"""Paired final interaction: late non-GK action consensus, captain and chips."""

from __future__ import annotations

import json

import numpy as np

import calibrate_model as lens
from action_segment_validation import consensus_parts
from frontier_ranker_validation import STRATEGY
from multiscale_horizon_validation import add_targets
from multiscale_phase_validation import event_number
from premium_captain_validation import captain_variants
from probabilistic_minutes_validation import season_summary
from wildcard_freehit_ablation import champion_forecasts


def evaluate(totals: np.ndarray, seasons: list[str], targets: dict[str, int]) -> dict:
    summary = season_summary(totals, seasons)
    rows = []
    for row in summary["seasons"]:
        season_key = row["season"].replace("/", "-")
        target = targets[season_key]
        rows.append({**row, "target": target, "margin": row["points"] - target})
    return {
        **summary,
        "targetHits": int(sum(row["margin"] >= 0 for row in rows)),
        "averageTargetGap": round(float(np.mean([row["margin"] for row in rows])), 1),
        "seasons": rows,
    }


def main() -> None:
    original, _ = lens.load_or_build_prepared_history()
    data = add_targets(original.reset_index(drop=True))
    immediate, champion_plan, captain = champion_forecasts(data)
    consensus, agreement, _ = consensus_parts(data, champion_plan)
    late_non_gk = (
        (event_number(data) >= 25)
        & agreement
        & (data["position_id"].to_numpy(int) != 1)
    )
    action_plan = champion_plan + 0.05 * late_non_gk.astype(float) * (consensus - champion_plan)
    captain_components = captain_variants(data, immediate, captain)
    frozen_rank = captain_components["frozen"]
    ceiling = (captain_components["ceiling25"] - 0.75 * frozen_rank) / 0.25
    captain15 = 0.85 * frozen_rank + 0.15 * ceiling
    seasons = list(dict.fromkeys(data["season"].tolist()))
    benchmark = json.loads(
        (lens.ROOT / "analysis" / "data" / "historical_rank_benchmarks.json").read_text(encoding="utf-8")
    )
    targets = {row["season"].replace("/", "-"): int(row["points"]) for row in benchmark["seasons"]}
    configs = (
        ("frozen-no-chips", champion_plan, captain, None),
        ("late-action-captain15-no-chips", action_plan, captain15, None),
        ("frozen-audited-chips", champion_plan, captain, lens.AUDITED_CHAMPION_CHIP_POLICY),
        ("late-action-audited-chips", action_plan, captain, lens.AUDITED_CHAMPION_CHIP_POLICY),
        ("late-action-captain15-audited-chips", action_plan, captain15, lens.AUDITED_CHAMPION_CHIP_POLICY),
    )
    results = {}
    stats_by_name = {}
    for name, plan, captain_score, chip_policy in configs:
        print(f"Final paired interaction {name}", flush=True)
        totals, stats = lens.simulate_candidate(
            data, immediate, STRATEGY, chip_policy=chip_policy,
            plan_scores=plan, captain_scores=captain_score,
        )
        results[name] = evaluate(totals, seasons, targets)
        stats_by_name[name] = stats
    frozen_no = results["frozen-no-chips"]
    combined_no = results["late-action-captain15-no-chips"]
    frozen_chip = results["frozen-audited-chips"]
    action_chip = results["late-action-audited-chips"]
    combined_chip = results["late-action-captain15-audited-chips"]

    def paired(new: dict, old: dict) -> list[dict]:
        return [
            {
                "season": before["season"],
                "before": before["points"],
                "after": after["points"],
                "delta": after["points"] - before["points"],
            }
            for before, after in zip(old["seasons"], new["seasons"])
        ]

    result = {
        "status": "prospective shadow; not production-promoted from historical interaction",
        "method": "Fully paired recursive replay. The action policy is fixed at GW25+, non-GK, 5% two-band agreement; captain is the fixed 15% ceiling blend; chip policy is unchanged.",
        "models": results,
        "comparisons": {
            "combinedNoChipVsFrozen": {
                "averageDelta": round(combined_no["average"] - frozen_no["average"], 1),
                "minimumDelta": combined_no["minimum"] - frozen_no["minimum"],
                "paired": paired(combined_no, frozen_no),
            },
            "actionChipVsFrozenChip": {
                "averageDelta": round(action_chip["average"] - frozen_chip["average"], 1),
                "minimumDelta": action_chip["minimum"] - frozen_chip["minimum"],
                "paired": paired(action_chip, frozen_chip),
            },
            "combinedChipVsFrozenChip": {
                "averageDelta": round(combined_chip["average"] - frozen_chip["average"], 1),
                "minimumDelta": combined_chip["minimum"] - frozen_chip["minimum"],
                "paired": paired(combined_chip, frozen_chip),
            },
        },
        "chipUsage": {
            name: [
                {"season": row["season"].replace("-", "/"), "chips": row["chips"]}
                for row in stats
                if row["season"] in lens.EVALUATION_SEASONS
            ]
            for name, stats in stats_by_name.items()
            if "chips" in name
        },
        "productionPromotion": False,
    }
    output = lens.ROOT / "analysis" / "data" / "late_action_captain_chip_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        name: {
            "average": row["average"], "minimum": row["minimum"],
            "holdout": row["holdoutAverage"], "targetHits": row["targetHits"],
            "averageTargetGap": row["averageTargetGap"],
        }
        for name, row in results.items()
    }, indent=2))
    print(json.dumps(result["comparisons"], indent=2))


if __name__ == "__main__":
    main()
