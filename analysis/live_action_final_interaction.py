"""Final paired replay for the live-compatible action consensus policy."""

from __future__ import annotations

import json

import numpy as np

import calibrate_model as lens
from frontier_ranker_validation import STRATEGY
from late_action_captain_chip_validation import evaluate
from live_action_ensemble_validation import mapped_seed_predictions, policy
from multiscale_horizon_validation import add_targets
from multiscale_phase_validation import event_number
from premium_captain_validation import captain_variants
from wildcard_freehit_ablation import champion_forecasts


def main() -> None:
    original, _ = lens.load_or_build_prepared_history()
    data = add_targets(original.reset_index(drop=True))
    immediate, champion_plan, captain = champion_forecasts(data)
    mapped = mapped_seed_predictions(data, champion_plan)
    raw_action, consensus, _ = policy(mapped, champion_plan, "vote80", 0.05)
    active = (
        (event_number(data) >= 25)
        & (data["position_id"].to_numpy(int) != 1)
        & consensus
    )
    action_plan = np.where(active, raw_action, champion_plan)
    captain_components = captain_variants(data, immediate, captain)
    frozen_rank = captain_components["frozen"]
    ceiling = (captain_components["ceiling25"] - 0.75 * frozen_rank) / 0.25
    captain15 = 0.85 * frozen_rank + 0.15 * ceiling
    seasons = list(dict.fromkeys(data["season"].tolist()))
    benchmark = json.loads(
        (lens.ROOT / "analysis" / "data" / "historical_rank_benchmarks.json").read_text(
            encoding="utf-8"
        )
    )
    targets = {
        row["season"].replace("/", "-"): int(row["points"])
        for row in benchmark["seasons"]
    }
    configs = (
        ("frozen-no-chips", champion_plan, captain, None),
        ("live-action-no-chips", action_plan, captain, None),
        (
            "frozen-audited-chips",
            champion_plan,
            captain,
            lens.AUDITED_CHAMPION_CHIP_POLICY,
        ),
        (
            "live-action-audited-chips",
            action_plan,
            captain,
            lens.AUDITED_CHAMPION_CHIP_POLICY,
        ),
        (
            "live-action-captain15-audited-chips",
            action_plan,
            captain15,
            lens.AUDITED_CHAMPION_CHIP_POLICY,
        ),
    )
    results = {}
    for name, plan, captain_score, chips in configs:
        print(f"Live action final interaction: {name}", flush=True)
        totals, _ = lens.simulate_candidate(
            data,
            immediate,
            STRATEGY,
            chip_policy=chips,
            plan_scores=plan,
            captain_scores=captain_score,
        )
        results[name] = evaluate(totals, seasons, targets)

    def comparison(new_name: str, old_name: str) -> dict:
        new = results[new_name]
        old = results[old_name]
        return {
            "averageDelta": round(new["average"] - old["average"], 1),
            "minimumDelta": new["minimum"] - old["minimum"],
            "paired": [
                {
                    "season": before["season"],
                    "before": before["points"],
                    "after": after["points"],
                    "delta": after["points"] - before["points"],
                }
                for before, after in zip(old["seasons"], new["seasons"])
            ],
        }

    result = {
        "status": "exploratory prospective shadow; strict historical stability gate did not pass and production requires frozen forward results",
        "method": "Exact live-compatible five-seed/two-band 80% consensus at 5%, active GW25+ for non-GKs; fully paired recursive replay with unchanged audited chips and fixed captain blend.",
        "models": results,
        "comparisons": {
            "actionNoChipVsFrozen": comparison(
                "live-action-no-chips", "frozen-no-chips"
            ),
            "actionChipVsFrozenChip": comparison(
                "live-action-audited-chips", "frozen-audited-chips"
            ),
            "combinedVsFrozenChip": comparison(
                "live-action-captain15-audited-chips",
                "frozen-audited-chips",
            ),
        },
        "productionPromotion": False,
    }
    output = lens.ROOT / "analysis" / "data" / "live_action_final_interaction.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "averages": {
                    name: model["average"] for name, model in results.items()
                },
                "comparisons": {
                    name: {
                        "averageDelta": row["averageDelta"],
                        "minimumDelta": row["minimumDelta"],
                    }
                    for name, row in result["comparisons"].items()
                },
            },
            indent=2,
        )
    )
    print(f"Wrote {output.relative_to(lens.ROOT)}")


if __name__ == "__main__":
    main()
