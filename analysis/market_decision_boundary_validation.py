"""Use the market anchor without allowing it to redirect the transfer path."""

from __future__ import annotations

import json

import numpy as np

import calibrate_model as lens
from captain_fixture_history_validation import add_fixture_history
from captain_route_consensus_validation import selected_consensus_metric, weekly_percentile
from frontier_ranker_validation import STRATEGY
from market_lineup_challenger import (
    adjusted_forecasts,
    attach_market_predictions,
    causal_market_predictions,
    load_market_matches,
)
from multiscale_horizon_validation import add_targets
from probabilistic_minutes_validation import causal_predictions as minute_predictions
from probabilistic_minutes_validation import season_summary
from wildcard_freehit_ablation import champion_forecasts


def with_delta(summary: dict, baseline: dict) -> dict:
    deltas = [
        row["points"] - old["points"]
        for row, old in zip(summary["seasons"], baseline["seasons"])
    ]
    development = np.asarray(deltas[:-2], dtype=float)
    holdout = np.asarray(deltas[-2:], dtype=float)
    return {
        **summary,
        "averageDelta": round(float(np.mean(deltas)), 1),
        "developmentDelta": round(float(development.mean()), 1),
        "developmentStability": round(
            float(development.mean() - 0.20 * development.std()), 3
        ),
        "holdoutDelta": round(float(holdout.mean()), 1),
        "worstSeasonDelta": int(min(deltas)),
        "positiveSeasons": int(sum(delta > 0 for delta in deltas)),
        "negativeSeasons": int(sum(delta < 0 for delta in deltas)),
        "seasonDeltas": deltas,
    }


def main() -> None:
    matches, _ = load_market_matches()
    market = causal_market_predictions(matches)
    original, _ = lens.load_or_build_prepared_history()
    data = add_fixture_history(add_targets(original.reset_index(drop=True)))
    data, coverage = attach_market_predictions(data, market)
    immediate, plan, frozen_captain = champion_forecasts(data)
    captain = selected_consensus_metric(data, immediate, frozen_captain)
    minute = minute_predictions(data)
    seasons = list(dict.fromkeys(data["season"].tolist()))
    baseline_totals, _ = lens.simulate_candidate(
        data,
        immediate,
        STRATEGY,
        plan_scores=plan,
        captain_scores=captain,
    )
    baseline = season_summary(baseline_totals, seasons)

    variants = []
    positions = data["position_id"].to_numpy(int)
    for strength in (0.15, 0.30, 0.50, 1.00):
        updated, market_score, _ = adjusted_forecasts(
            data,
            immediate.copy(),
            plan.copy(),
            minute,
            strength,
            0.0,
        )
        configs = {
            f"lineupAll{strength:.2f}": (market_score, captain),
            f"lineupDefence{strength:.2f}": (
                np.where(positions <= 2, market_score, immediate),
                captain,
            ),
            f"lineupAttack{strength:.2f}": (
                np.where(positions >= 3, market_score, immediate),
                captain,
            ),
        }
        market_rank = weekly_percentile(data, market_score)
        for share in (0.10, 0.20):
            configs[f"captain{strength:.2f}share{share:.2f}"] = (
                immediate,
                (1 - share) * captain + share * market_rank,
            )
        for name, (score, captain_score) in configs.items():
            print(f"Market boundary replay {name}", flush=True)
            totals, _ = lens.simulate_candidate(
                updated,
                score,
                STRATEGY,
                # The champion plan is immutable in this experiment.  It fixes
                # the opening squad and every transfer boundary.
                plan_scores=plan,
                captain_scores=captain_score,
            )
            variants.append(
                {
                    "name": name,
                    "marketStrength": strength,
                    "selectionSurface": (
                        "captain"
                        if name.startswith("captain")
                        else "defence"
                        if "Defence" in name
                        else "attack"
                        if "Attack" in name
                        else "all"
                    ),
                    **with_delta(season_summary(totals, seasons), baseline),
                }
            )

    selected = max(
        variants,
        key=lambda row: (
            row["developmentStability"],
            row["developmentDelta"],
            -row["marketStrength"],
        ),
    )
    selected_name = selected["name"]
    selected_strength = float(selected["marketStrength"])
    selected_data, market_score, _ = adjusted_forecasts(
        data,
        immediate.copy(),
        plan.copy(),
        minute,
        selected_strength,
        0.0,
    )
    if selected_name.startswith("captain"):
        share = float(selected_name.split("share", maxsplit=1)[1])
        selected_score = immediate
        selected_captain = (
            (1 - share) * captain
            + share * weekly_percentile(data, market_score)
        )
    else:
        selected_captain = captain
        selected_score = (
            np.where(positions <= 2, market_score, immediate)
            if "Defence" in selected_name
            else np.where(positions >= 3, market_score, immediate)
            if "Attack" in selected_name
            else market_score
        )
    baseline_chip_totals, _ = lens.simulate_candidate(
        data,
        immediate,
        STRATEGY,
        chip_policy=lens.AUDITED_CHAMPION_CHIP_POLICY,
        plan_scores=plan,
        captain_scores=captain,
    )
    selected_chip_totals, _ = lens.simulate_candidate(
        selected_data,
        selected_score,
        STRATEGY,
        chip_policy=lens.AUDITED_CHAMPION_CHIP_POLICY,
        plan_scores=plan,
        captain_scores=selected_captain,
    )
    baseline_chips = season_summary(baseline_chip_totals, seasons)
    selected_chips = with_delta(
        season_summary(selected_chip_totals, seasons), baseline_chips
    )
    evidence_gate = {
        "developmentPositive": selected["developmentDelta"] > 0,
        "holdoutNonNegative": selected["holdoutDelta"] >= 0,
        "worstSeasonAtLeastMinusFive": selected["worstSeasonDelta"] >= -5,
        "positiveSeasonsAtLeastFive": selected["positiveSeasons"] >= 5,
        "transferPathFrozen": True,
        "deadlineTimestampAuditable": False,
    }
    result = {
        "status": "research-only market boundary isolation",
        "method": (
            "The same champion plan selects every opening squad and transfer. "
            "Pre-closing market information is permitted to change only XI order, "
            "defence/attack order, or captain order in separate variants."
        ),
        "mapping": coverage,
        "baselineNoChips": baseline,
        "variants": variants,
        "selectedByDevelopmentOnly": selected,
        "auditedChipInteraction": {
            "baseline": baseline_chips,
            "challenger": selected_chips,
        },
        "evidenceGate": evidence_gate,
        "productionPromotion": all(evidence_gate.values()),
    }
    output = lens.ROOT / "analysis" / "data" / "market_decision_boundary_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "selected": selected,
                "auditedChips": selected_chips,
                "evidenceGate": evidence_gate,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
