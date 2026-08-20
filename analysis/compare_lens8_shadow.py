"""Like-for-like replay of the frozen Breakthrough v3 stack on Lens 8 rules."""

from __future__ import annotations

import json
from datetime import date

import numpy as np

import calibrate_model as lens
from captain_fixture_history_validation import add_fixture_history
from captain_route_consensus_validation import selected_consensus_metric
from dynamic_match_model_v2 import build_dynamic_history
from forecast_champion_v2 import (
    BENCH_BOOST_THRESHOLD,
    FREE_HIT_RISK_DISCOUNT,
    FREE_HIT_THRESHOLD,
    TRIPLE_CAPTAIN_THRESHOLD,
    selected_forecasts,
)
from freehit_value_validation import causal_predictions, opportunity_frame
from frontier_ranker_validation import STRATEGY
from multiscale_horizon_validation import add_targets
from wildcard_freehit_ablation import champion_forecasts


def repaired_chip_inputs(data, immediate, plan, captain):
    free_hit_squads = lens.precompute_fresh_squads(
        data, immediate, one_week_only=True
    )
    collector = lens.ChipPolicy(
        1e6,
        1e6,
        1e6,
        1e6,
        0.0,
        10,
        28,
        ("Free Hit", "Bench Boost", "Triple Captain"),
    )
    no_chip, stats = lens.simulate_candidate(
        data,
        immediate,
        STRATEGY,
        chip_policy=collector,
        free_hit_squads=free_hit_squads,
        plan_scores=plan,
        captain_scores=captain,
    )
    seasons = list(dict.fromkeys(data["season"].tolist()))
    frame = opportunity_frame(stats, seasons)
    prediction, residual_scale, fit_audit = causal_predictions(frame)
    frame["fhSignal"] = (
        prediction
        - frame["permanentTransferValueForegone"].to_numpy(float)
        - FREE_HIT_RISK_DISCOUNT * residual_scale
    )
    overrides = {
        (str(row.season), int(row.gw), "Free Hit"): float(row.fhSignal)
        for row in frame.itertuples()
    }
    policy = lens.ChipPolicy(
        1e6,
        FREE_HIT_THRESHOLD,
        BENCH_BOOST_THRESHOLD,
        TRIPLE_CAPTAIN_THRESHOLD,
        0.0,
        10,
        28,
        ("Free Hit", "Bench Boost", "Triple Captain"),
    )
    return no_chip, free_hit_squads, overrides, policy, fit_audit


def refresh_public_audit(result: dict, lens8_artifact: dict) -> None:
    """Keep the public benchmark hierarchy tied to reproducible output files."""
    audit_path = lens.ROOT / "app" / "data" / "model-audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    lens7_average = float(audit["lens7"]["average"])
    lens8_average = float(result["lens8"]["average"])
    causal_average = float(result["causalRepairedShadow"]["average"])
    repaired_legacy = float(result["repairedShadow"]["average"])
    recorded_legacy = float(result["recordedLegacyShadow"]["average"])
    pace = round(
        float(np.mean([row["top500Target"] for row in lens8_artifact["backtest"]])),
        1,
    )
    audit["generatedAt"] = date.today().isoformat()
    audit["lens8"].update(
        {
            "average": lens8_average,
            "deltaVsLens7": round(lens8_average - lens7_average, 1),
            "top500Hits": int(lens8_artifact["rankTarget"]["hits"]),
            "seasons": len(lens8_artifact["backtest"]),
        }
    )
    audit["causalChallenger"].update(
        {
            "average": causal_average,
            "deltaVsLens8": round(causal_average - lens8_average, 1),
        }
    )
    audit["top500Pace"] = pace
    audit["causalGapToPace"] = round(pace - causal_average, 1)
    audit["legacyBreakthrough"].update(
        {
            "average": recorded_legacy,
            "reproducedAverage": repaired_legacy,
            "overstatement": round(recorded_legacy - repaired_legacy, 1),
        }
    )
    audit["hybridDiagnostic"]["average"] = float(
        result["lens81HybridDiagnostic"]["average"]
    )
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    dynamic, dynamic_audit = build_dynamic_history()
    data = add_fixture_history(add_targets(dynamic.reset_index(drop=True)))
    immediate, plan, captain = selected_forecasts(data)
    no_chip, free_hits, overrides, policy, fit_audit = repaired_chip_inputs(
        data, immediate, plan, captain
    )
    repaired, stats = lens.simulate_candidate(
        data,
        immediate,
        STRATEGY,
        chip_policy=policy,
        free_hit_squads=free_hits,
        plan_scores=plan,
        captain_scores=captain,
        chip_value_overrides=overrides,
    )
    causal_immediate, causal_plan, frozen_captain = champion_forecasts(data)
    causal_captain = selected_consensus_metric(
        data, causal_immediate, frozen_captain
    )
    (
        causal_no_chip,
        causal_free_hits,
        causal_overrides,
        causal_policy,
        _,
    ) = repaired_chip_inputs(
        data, causal_immediate, causal_plan, causal_captain
    )
    causal_repaired, causal_stats = lens.simulate_candidate(
        data,
        causal_immediate,
        STRATEGY,
        chip_policy=causal_policy,
        free_hit_squads=causal_free_hits,
        plan_scores=causal_plan,
        captain_scores=causal_captain,
        chip_value_overrides=causal_overrides,
    )
    lens8 = json.loads(lens.OUTPUT.read_text(encoding="utf-8-sig"))
    lens8_policy_payload = lens8["chipStrategy"]["policy"]
    lens8_policy = lens.ChipPolicy(
        float(lens8_policy_payload["wildcardGap"]),
        float(lens8_policy_payload["freeHitGap"]),
        float(lens8_policy_payload["benchScore"]),
        float(lens8_policy_payload["tripleScore"]),
        float(lens8_policy_payload["afconBonus"]),
        int(lens8_policy_payload["firstWildcardMinGw"]),
        int(lens8_policy_payload["secondWildcardMinGw"]),
        (
            tuple(lens8_policy_payload["enabledChips"])
            if lens8_policy_payload.get("enabledChips")
            else None
        ),
    )
    hybrid_fresh = lens.precompute_fresh_squads(data, causal_plan)
    hybrid, hybrid_stats = lens.simulate_candidate(
        data,
        causal_immediate,
        STRATEGY,
        chip_policy=lens8_policy,
        fresh_squads=hybrid_fresh,
        free_hit_squads=causal_free_hits,
        plan_scores=causal_plan,
        captain_scores=causal_captain,
    )
    seasons = list(dict.fromkeys(data["season"].tolist()))
    evaluation = np.asarray(repaired[2:], float)
    evaluation_no_chip = np.asarray(no_chip[2:], float)
    causal_evaluation = np.asarray(causal_repaired[2:], float)
    causal_no_chip_evaluation = np.asarray(causal_no_chip[2:], float)
    hybrid_evaluation = np.asarray(hybrid[2:], float)
    lens8_points = np.asarray(
        [float(row["points"]) for row in lens8["backtest"]], dtype=float
    )
    legacy = json.loads(
        (lens.ROOT / "app" / "data" / "breakthrough-v3.json").read_text(
            encoding="utf-8-sig"
        )
    )
    legacy_points = np.asarray(
        [float(row["model"]) for row in legacy["seasons"]], dtype=float
    )
    result = {
        "status": "like-for-like diagnostic; frozen shadow forecast stack replayed through Lens 8 fixture, legality and exact-MILP rules",
        "seasons": [season.replace("-", "/") for season in seasons[2:]],
        "recordedLegacyShadow": {
            "average": round(float(legacy_points.mean()), 1),
            "points": legacy_points.astype(int).tolist(),
        },
        "repairedShadow": {
            "average": round(float(evaluation.mean()), 1),
            "points": evaluation.astype(int).tolist(),
            "noChipAverage": round(float(evaluation_no_chip.mean()), 1),
            "chipGain": round(float((evaluation - evaluation_no_chip).mean()), 1),
            "averageTransfers": round(
                float(np.mean([row["transfers"] for row in stats[2:]])), 1
            ),
            "averageHits": round(
                float(np.mean([row["hits"] for row in stats[2:]])), 1
            ),
        },
        "causalRepairedShadow": {
            "average": round(float(causal_evaluation.mean()), 1),
            "points": causal_evaluation.astype(int).tolist(),
            "noChipAverage": round(float(causal_no_chip_evaluation.mean()), 1),
            "chipGain": round(
                float((causal_evaluation - causal_no_chip_evaluation).mean()), 1
            ),
            "averageTransfers": round(
                float(np.mean([row["transfers"] for row in causal_stats[2:]])), 1
            ),
            "averageHits": round(
                float(np.mean([row["hits"] for row in causal_stats[2:]])), 1
            ),
        },
        "lens81HybridDiagnostic": {
            "average": round(float(hybrid_evaluation.mean()), 1),
            "points": hybrid_evaluation.astype(int).tolist(),
            "chipGain": round(
                float((hybrid_evaluation - causal_no_chip_evaluation).mean()), 1
            ),
            "averageTransfers": round(
                float(np.mean([row["transfers"] for row in hybrid_stats[2:]])), 1
            ),
            "averageHits": round(
                float(np.mean([row["hits"] for row in hybrid_stats[2:]])), 1
            ),
            "warning": "Diagnostic only: Lens 8 chip policy was selected on the broader historical search and must pass a frozen walk-forward gate before promotion.",
        },
        "lens8": {
            "average": round(float(lens8_points.mean()), 1),
            "points": lens8_points.astype(int).tolist(),
        },
        "deltas": {
            "repairedShadowVsRecordedLegacy": round(
                float((evaluation - legacy_points).mean()), 1
            ),
            "lens8VsRepairedShadow": round(
                float((lens8_points - evaluation).mean()), 1
            ),
            "lens8VsCausalRepairedShadow": round(
                float((lens8_points - causal_evaluation).mean()), 1
            ),
            "hybridVsLens8": round(
                float((hybrid_evaluation - lens8_points).mean()), 1
            ),
            "lens8VsRecordedLegacy": round(
                float((lens8_points - legacy_points).mean()), 1
            ),
            "lens8VsRepairedShadowBySeason": (
                lens8_points - evaluation
            ).astype(int).tolist(),
        },
        "dynamicMarketAudit": dynamic_audit,
        "freeHitFitAudit": fit_audit,
        "fixtureIntegrity": lens.fixture_integrity_audit(data),
    }
    output = lens.ROOT / "analysis" / "data" / "lens8_shadow_comparison.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    refresh_public_audit(result, lens8)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "recordedLegacyShadow",
                    "repairedShadow",
                    "causalRepairedShadow",
                    "lens81HybridDiagnostic",
                    "lens8",
                    "deltas",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
