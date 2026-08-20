"""Paired full-season tournament for the forecast-breakthrough programme.

Every candidate receives the same optimizer, transfer rules and audited chip
policy.  A forecast is useful only if it improves recursive FPL points; lower
MAE alone is not a promotion criterion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

import calibrate_model as lens
from captain_fixture_history_validation import add_fixture_history
from captain_route_consensus_validation import selected_consensus_metric, weekly_percentile
from dynamic_match_model_v2 import build_dynamic_history
from forecast_layer_v2 import captain_availability_score, dynamic_route_score
from frontier_ranker_validation import STRATEGY
from multiscale_horizon_validation import add_targets
from openfpl_position_ensemble_v2 import build_ensemble
from probabilistic_minutes_validation import causal_predictions as minute_predictions
from probabilistic_minutes_validation import season_summary
from wildcard_freehit_ablation import champion_forecasts


@dataclass(frozen=True)
class CandidateForecast:
    name: str
    score: np.ndarray
    captain: np.ndarray
    selection_surface: str
    rationale: str


def blend(base: np.ndarray, challenger: np.ndarray, share: float) -> np.ndarray:
    return (1.0 - share) * np.asarray(base, float) + share * np.asarray(challenger, float)


def summary_with_delta(
    totals: np.ndarray,
    baseline_totals: np.ndarray,
    seasons: list[str],
) -> dict:
    summary = season_summary(totals, seasons)
    baseline = season_summary(baseline_totals, seasons)
    deltas = np.asarray(
        [
            row["points"] - old["points"]
            for row, old in zip(summary["seasons"], baseline["seasons"])
        ],
        dtype=float,
    )
    development = deltas[:-2]
    holdout = deltas[-2:]
    return {
        **summary,
        "averageDelta": round(float(deltas.mean()), 1),
        "developmentDelta": round(float(development.mean()), 1),
        "developmentStability": round(
            float(development.mean() - 0.20 * development.std()), 3
        ),
        "holdoutDelta": round(float(holdout.mean()), 1),
        "worstSeasonDelta": int(deltas.min()),
        "positiveSeasons": int((deltas > 0).sum()),
        "negativeSeasons": int((deltas < 0).sum()),
        "seasonDeltas": deltas.astype(int).tolist(),
    }


def candidate_set(
    data,
    immediate: np.ndarray,
    captain: np.ndarray,
    dynamic_scores: dict[float, np.ndarray],
    minute: dict[str, np.ndarray],
    openfpl: np.ndarray,
) -> list[CandidateForecast]:
    candidates: list[CandidateForecast] = [
        CandidateForecast(
            "control",
            immediate,
            captain,
            "none",
            "Frozen recursive champion with route-consensus captaincy.",
        )
    ]
    for strength in (0.15, 0.30, 0.50):
        dynamic_rank = weekly_percentile(data, dynamic_scores[strength])
        candidates.append(
            CandidateForecast(
                f"dynamicCaptain{strength:.2f}",
                immediate,
                blend(captain, dynamic_rank, 0.10),
                "captain-only",
                "Dynamic match evidence may reorder only the armband.",
            )
        )
    dynamic_030 = dynamic_scores[0.30]
    for downside in (0.25, 0.50):
        availability, _ = captain_availability_score(
            data, dynamic_030, minute, downside
        )
        candidates.append(
            CandidateForecast(
                f"dynamicCaptainMinutes{downside:.2f}",
                immediate,
                blend(captain, weekly_percentile(data, availability), 0.10),
                "captain-only",
                "Dynamic armband evidence plus a no-show/rotation downside veto.",
            )
        )
    openfpl_rank = weekly_percentile(data, openfpl)
    for share in (0.05, 0.10, 0.15):
        candidates.append(
            CandidateForecast(
                f"openfplCaptain{share:.2f}",
                immediate,
                blend(captain, openfpl_rank, share),
                "captain-only",
                "Position ensemble may reorder only the armband.",
            )
        )
    positions = data["position_id"].to_numpy(int)
    for share in (0.025, 0.05, 0.10):
        selection = blend(immediate, openfpl, share)
        candidates.append(
            CandidateForecast(
                f"openfplSelection{share:.3f}",
                selection,
                captain,
                "recursive-selection",
                "Small position-ensemble blend may alter XI and transfer boundaries.",
            )
        )
    for share in (0.05, 0.10):
        defence_selection = np.where(
            positions <= 2, blend(immediate, openfpl, share), immediate
        )
        candidates.append(
            CandidateForecast(
                f"openfplDefence{share:.2f}",
                defence_selection,
                captain,
                "defence-selection",
                "Position ensemble is restricted to goalkeeper/defender ordering.",
            )
        )
    candidates.append(
        CandidateForecast(
            "predeclaredCombined",
            blend(immediate, openfpl, 0.05),
            blend(captain, weekly_percentile(data, dynamic_030), 0.10),
            "recursive-selection+captain",
            "Predeclared 5% player ensemble plus conservative dynamic armband boundary.",
        )
    )
    return candidates


def main() -> None:
    dynamic_raw, match_audit = build_dynamic_history()
    data = add_fixture_history(add_targets(dynamic_raw.reset_index(drop=True)))
    immediate, plan, frozen_captain = champion_forecasts(data)
    captain = selected_consensus_metric(data, immediate, frozen_captain)
    minute = minute_predictions(data)
    dynamic_scores = {
        strength: dynamic_route_score(data, immediate, strength)[0]
        for strength in (0.15, 0.30, 0.50)
    }
    openfpl, openfpl_audit = build_ensemble(data, immediate)
    seasons = list(dict.fromkeys(data["season"].tolist()))
    candidates = candidate_set(
        data, immediate, captain, dynamic_scores, minute, openfpl
    )

    print("Recursive tournament 1/%d: control" % len(candidates), flush=True)
    baseline_totals, _ = lens.simulate_candidate(
        data,
        immediate,
        STRATEGY,
        chip_policy=lens.AUDITED_CHAMPION_CHIP_POLICY,
        plan_scores=plan,
        captain_scores=captain,
    )
    rows = []
    for index, candidate in enumerate(candidates, start=1):
        if candidate.name == "control":
            totals = baseline_totals
        else:
            print(
                f"Recursive tournament {index}/{len(candidates)}: {candidate.name}",
                flush=True,
            )
            totals, _ = lens.simulate_candidate(
                data,
                candidate.score,
                STRATEGY,
                chip_policy=lens.AUDITED_CHAMPION_CHIP_POLICY,
                plan_scores=plan,
                captain_scores=candidate.captain,
            )
        rows.append(
            {
                "name": candidate.name,
                "selectionSurface": candidate.selection_surface,
                "rationale": candidate.rationale,
                **summary_with_delta(totals, baseline_totals, seasons),
            }
        )

    challengers = [row for row in rows if row["name"] != "control"]
    selected = max(
        challengers,
        key=lambda row: (
            row["developmentStability"],
            row["developmentDelta"],
            row["worstSeasonDelta"],
        ),
    )
    recursive_gate = {
        "developmentPositive": selected["developmentDelta"] > 0,
        "holdoutNonNegative": selected["holdoutDelta"] >= 0,
        "worstSeasonAtLeastMinusFive": selected["worstSeasonDelta"] >= -5,
        "positiveSeasonsAtLeastFour": selected["positiveSeasons"] >= 4,
    }
    provenance_gate = {
        "historicalDeadlineTimestampsAuditable": False,
        "prospectiveLockedDeadlineSampleComplete": False,
    }
    result = {
        "schemaVersion": 2,
        "status": "research challenger tournament",
        "selectionProtocol": (
            "Candidate choice maximises mean minus 0.20 standard deviations on "
            "the first six evaluation seasons. The final two seasons are a "
            "locked holdout and every row is a full recursive replay with the "
            "same audited chip policy."
        ),
        "baseline": rows[0],
        "selectedByDevelopmentOnly": selected,
        "recursiveGate": recursive_gate,
        "provenanceGate": provenance_gate,
        "researchGatePassed": all(recursive_gate.values()),
        "productionPromotion": all(recursive_gate.values()) and all(provenance_gate.values()),
        "matchForecast": {
            "coverage": match_audit["coverage"],
            "metrics": match_audit["metrics"],
        },
        "openfplForecast": {
            "metrics": openfpl_audit["metrics"],
            "informationBoundary": openfpl_audit["informationBoundary"],
        },
        "candidates": rows,
    }
    output = lens.ROOT / "analysis" / "data" / "forecast_breakthrough_tournament_v2.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "baseline": result["baseline"],
                "selected": selected,
                "recursiveGate": recursive_gate,
                "productionPromotion": result["productionPromotion"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
