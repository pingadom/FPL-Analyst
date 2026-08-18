"""Paired interaction test for audited TC/BB plus corrected Free Hit logic."""

from __future__ import annotations

import json

import numpy as np

import calibrate_model as lens
from freehit_value_validation import (
    causal_predictions,
    opportunity_frame,
)
from frontier_ranker_validation import STRATEGY
from multiscale_horizon_validation import add_targets
from wildcard_freehit_ablation import champion_forecasts


def row_summary(
    name: str,
    totals: np.ndarray,
    baseline: np.ndarray,
    stats: list[dict],
    seasons: list[str],
) -> dict:
    gain = totals - baseline
    evaluation = gain[2:]
    development = gain[2:6]
    holdout = gain[6:]
    return {
        "name": name,
        "evaluationAveragePoints": round(float(totals[2:].mean()), 1),
        "evaluationAverageGain": round(float(evaluation.mean()), 1),
        "development2018to2021AverageGain": round(float(development.mean()), 1),
        "holdout2022to2025AverageGain": round(float(holdout.mean()), 1),
        "holdoutMinimumGain": round(float(holdout.min()), 1),
        "seasonGain": [
            {
                "season": seasons[index].replace("-", "/"),
                "gain": round(float(gain[index]), 1),
                "chips": stats[index]["chips"],
            }
            for index in range(2, len(seasons))
        ],
    }


def main() -> None:
    original, _ = lens.load_or_build_prepared_history()
    data = add_targets(original.reset_index(drop=True))
    scores, plan_scores, captain_scores = champion_forecasts(data)
    seasons = list(dict.fromkeys(data["season"].tolist()))
    free_hit_squads = lens.precompute_fresh_squads(data, scores)

    collector_policy = lens.ChipPolicy(
        1e6, 1e6, 1e6, 1e6, 0.0, 10, 28, ("Free Hit",)
    )
    print("Collecting causal FH values", flush=True)
    _, collector_stats = lens.simulate_candidate(
        data,
        scores,
        STRATEGY,
        chip_policy=collector_policy,
        free_hit_squads=free_hit_squads,
        plan_scores=plan_scores,
        captain_scores=captain_scores,
    )
    frame = opportunity_frame(collector_stats, seasons)
    prediction, _, fit_audit = causal_predictions(frame)
    adjusted = prediction - frame["permanentTransferValueForegone"].to_numpy(float)
    overrides = {
        (str(row.season), int(row.gw), "Free Hit"): float(adjusted[index])
        for index, row in frame.iterrows()
    }

    print("Running no-chip control", flush=True)
    baseline, baseline_stats = lens.simulate_candidate(
        data,
        scores,
        STRATEGY,
        plan_scores=plan_scores,
        captain_scores=captain_scores,
    )
    print("Running audited TC/BB control", flush=True)
    tc_bb, tc_bb_stats = lens.simulate_candidate(
        data,
        scores,
        STRATEGY,
        chip_policy=lens.AUDITED_CHAMPION_CHIP_POLICY,
        plan_scores=plan_scores,
        captain_scores=captain_scores,
    )
    print("Running TC/BB + corrected FH challenger", flush=True)
    combined_policy = lens.ChipPolicy(
        1e6,
        3.0,
        lens.AUDITED_CHAMPION_CHIP_POLICY.bench_score,
        lens.AUDITED_CHAMPION_CHIP_POLICY.triple_score,
        0.0,
        10,
        28,
        ("Free Hit", "Bench Boost", "Triple Captain"),
    )
    combined, combined_stats = lens.simulate_candidate(
        data,
        scores,
        STRATEGY,
        chip_policy=combined_policy,
        free_hit_squads=free_hit_squads,
        plan_scores=plan_scores,
        captain_scores=captain_scores,
        chip_value_overrides=overrides,
    )
    tc_bb_row = row_summary(
        "audited-tc-bb", tc_bb, baseline, tc_bb_stats, seasons
    )
    combined_row = row_summary(
        "audited-tc-bb-plus-corrected-fh",
        combined,
        baseline,
        combined_stats,
        seasons,
    )
    paired = combined - tc_bb
    holdout_paired = paired[6:]
    promoted = bool(
        holdout_paired.mean() > 0
        and holdout_paired.min() >= -15
        and combined_row["development2018to2021AverageGain"] > 0
    )
    result = {
        "status": (
            "prospective shadow chip challenger"
            if promoted
            else "rejected combined chip challenger"
        ),
        "method": (
            "Free Hit uses prior-season-only learned one-week value, is decided "
            "before permanent transfers, and competes with TC/BB under the one-"
            "chip-per-Gameweek constraint in a complete recursive rerun."
        ),
        "fitAudit": fit_audit,
        "controls": {
            "noChipEvaluationAverage": round(float(baseline[2:].mean()), 1),
            "tcBb": tc_bb_row,
        },
        "challenger": combined_row,
        "pairedVsTcBb": {
            "evaluationAverage": round(float(paired[2:].mean()), 1),
            "development2018to2021Average": round(float(paired[2:6].mean()), 1),
            "holdout2022to2025Average": round(float(holdout_paired.mean()), 1),
            "holdoutMinimum": round(float(holdout_paired.min()), 1),
            "seasonDelta": paired[2:].round().astype(int).tolist(),
        },
        "promotedToProspectiveShadow": promoted,
        "governance": (
            "Shadow promotion does not change the historical champion or the "
            "public site. Live activation still requires the Monte Carlo downside "
            "gate and announced blank/double structure."
        ),
    }
    output = lens.ROOT / "analysis" / "data" / "combined_chip_policy_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "noChip": result["controls"]["noChipEvaluationAverage"],
                "tcBb": {
                    key: tc_bb_row[key]
                    for key in (
                        "evaluationAveragePoints",
                        "evaluationAverageGain",
                        "holdout2022to2025AverageGain",
                    )
                },
                "combined": {
                    key: combined_row[key]
                    for key in (
                        "evaluationAveragePoints",
                        "evaluationAverageGain",
                        "holdout2022to2025AverageGain",
                        "holdoutMinimumGain",
                    )
                },
                "pairedVsTcBb": result["pairedVsTcBb"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
