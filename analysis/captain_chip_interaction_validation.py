"""Paired chip interaction for the best research-only captain challenger."""

from __future__ import annotations

import json

import numpy as np

import calibrate_model as lens
from frontier_ranker_validation import STRATEGY
from multiscale_horizon_validation import add_targets
from premium_captain_validation import captain_variants
from probabilistic_minutes_validation import season_summary
from wildcard_freehit_ablation import champion_forecasts


def main() -> None:
    original, _ = lens.load_or_build_prepared_history()
    data = add_targets(original.reset_index(drop=True))
    immediate, plan, captain = champion_forecasts(data)
    variants = captain_variants(data, immediate, captain)
    frozen = variants["frozen"]
    ceiling = (variants["ceiling25"] - 0.75 * frozen) / 0.25
    challenger = 0.85 * frozen + 0.15 * ceiling
    seasons = list(dict.fromkeys(data["season"].tolist()))
    rows = {}
    raw = {}
    for name, captain_score in [("frozen", captain), ("ceiling15", challenger)]:
        print(f"Audited chips with {name} captain", flush=True)
        totals, stats = lens.simulate_candidate(
            data,
            immediate,
            STRATEGY,
            chip_policy=lens.AUDITED_CHAMPION_CHIP_POLICY,
            plan_scores=plan,
            captain_scores=captain_score,
        )
        rows[name] = season_summary(totals, seasons)
        raw[name] = (totals, stats)
    paired = []
    for challenger_row, frozen_row in zip(rows["ceiling15"]["seasons"], rows["frozen"]["seasons"]):
        paired.append({
            "season": frozen_row["season"],
            "frozen": frozen_row["points"],
            "ceiling15": challenger_row["points"],
            "delta": challenger_row["points"] - frozen_row["points"],
        })
    result = {
        "status": "research-only; captain no-chip downside gate already failed",
        "method": "Paired recursive replay with the identical audited chip policy; only the captain ranking differs.",
        "frozen": rows["frozen"],
        "ceiling15": rows["ceiling15"],
        "averageDelta": round(rows["ceiling15"]["average"] - rows["frozen"]["average"], 1),
        "paired": paired,
        "chipUsage": {
            name: [
                {"season": stats[season_index]["season"].replace("-", "/"), "chips": stats[season_index]["chips"]}
                for season_index in range(len(lens.TRAINING_SEASONS), len(stats))
            ]
            for name, (_, stats) in raw.items()
        },
    }
    output = lens.ROOT / "analysis" / "data" / "captain_chip_interaction_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"frozen": rows["frozen"], "ceiling15": rows["ceiling15"], "paired": paired}, indent=2))


if __name__ == "__main__":
    main()
