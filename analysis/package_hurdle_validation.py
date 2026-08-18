"""Predeclared isolation of the opportunity cost for extra same-GW moves."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

import calibrate_model as lens
from bench_efficiency_validation import variant_summary
from frontier_ranker_validation import STRATEGY
from multiscale_horizon_validation import add_targets
from wildcard_freehit_ablation import champion_forecasts


HURDLES = (1.15, 3.0, 5.0, 8.0, 12.0)


def main() -> None:
    original, _ = lens.load_or_build_prepared_history()
    data = add_targets(original.reset_index(drop=True))
    scores, plan, captain = champion_forecasts(data)
    seasons = list(dict.fromkeys(data["season"].tolist()))
    rows = []
    for hurdle in HURDLES:
        name = "champion" if hurdle == 1.15 else f"extra-move-hurdle-{hurdle:g}"
        strategy = replace(
            STRATEGY,
            name=name,
            additional_move_hurdle=hurdle,
        )
        print(f"Recursive {name}", flush=True)
        totals, stats = lens.simulate_candidate(
            data,
            scores,
            strategy,
            plan_scores=plan,
            captain_scores=captain,
            tracked_player_name="Salah",
        )
        development = totals[2:6]
        rows.append(
            {
                "name": name,
                "additionalMoveHurdle": hurdle,
                "developmentStability": round(
                    float(development.mean() - 0.25 * development.std()), 3
                ),
                "holdoutAverage": round(float(totals[6:].mean()), 1),
                "summary": variant_summary(totals, stats, seasons),
            }
        )

    baseline = rows[0]
    best_challenger = max(rows[1:], key=lambda row: row["developmentStability"])
    selected = max(rows, key=lambda row: row["developmentStability"])
    paired = [
        {
            "season": old["season"],
            "champion": old["points"],
            "challenger": new["points"],
            "delta": new["points"] - old["points"],
        }
        for old, new in zip(
            baseline["summary"]["seasons"],
            best_challenger["summary"]["seasons"],
        )
    ]
    robust = bool(
        best_challenger["developmentStability"] > baseline["developmentStability"]
        and best_challenger["holdoutAverage"] >= baseline["holdoutAverage"]
        and best_challenger["summary"]["minimum"] >= baseline["summary"]["minimum"] - 10
        and sum(row["delta"] > 0 for row in paired) >= 5
    )
    result = {
        "status": "paired extra-transfer opportunity-cost ablation",
        "method": "Only the extra hurdle for the second and later same-GW free transfers changes; forecasts, first-transfer hurdle, squad optimiser and captain are frozen.",
        "selectionRule": "Select on 2018/19-2021/22 stability, then require non-negative 2022/23-2025/26 holdout, downside and five-season breadth.",
        "selected": selected,
        "bestChallenger": best_challenger,
        "pairedBestChallengerVsChampion": paired,
        "robustPromotion": robust,
        "experiments": rows,
    }
    output = lens.ROOT / "analysis" / "data" / "package_hurdle_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "selected": selected["name"],
                "bestChallenger": best_challenger["name"],
                "robustPromotion": robust,
                "paired": paired,
                "experiments": [
                    {
                        "name": row["name"],
                        "developmentStability": row["developmentStability"],
                        "average": row["summary"]["average"],
                        "minimum": row["summary"]["minimum"],
                        "holdoutAverage": row["holdoutAverage"],
                        "transfers": [
                            season["transfers"]
                            for season in row["summary"]["seasons"]
                        ],
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
