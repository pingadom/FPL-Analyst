"""Large causal fine-margin captain search followed by exact recursive finalists."""

from __future__ import annotations

import json

import numpy as np

import calibrate_model as lens
from captain_fixture_history_validation import add_fixture_history, decision_evaluation
from captain_route_consensus_validation import selected_consensus_metric, weekly_percentile
from dynamic_match_model_v2 import build_dynamic_history
from forecast_layer_v2 import captain_availability_score, dynamic_route_score
from frontier_ranker_validation import STRATEGY
from multiscale_horizon_validation import add_targets
from openfpl_position_ensemble_v2 import build_ensemble
from probabilistic_minutes_validation import causal_predictions as minute_predictions
from probabilistic_minutes_validation import season_summary
from wildcard_freehit_ablation import champion_forecasts


DYNAMIC_STRENGTHS = tuple(round(value, 2) for value in np.linspace(0.0, 1.0, 11))
MATCH_SHARES = tuple(round(value, 3) for value in np.linspace(0.0, 0.25, 11))
PLAYER_SHARES = (0.0, 0.025, 0.05, 0.075, 0.10)
MINUTE_DOWNSIDES = (0.0, 0.25, 0.50)
FINALISTS = 12


def development_stability(row: dict) -> float:
    development = np.asarray(row["seasonDeltas"][:-2], float)
    return float(development.mean() - 0.20 * development.std())


def full_summary(totals, baseline, seasons) -> dict:
    summary = season_summary(totals, seasons)
    delta = totals[2:] - baseline[2:]
    development = delta[:-2]
    return {
        **summary,
        "averageDelta": round(float(delta.mean()), 1),
        "developmentDelta": round(float(development.mean()), 1),
        "developmentStability": round(
            float(development.mean() - 0.20 * development.std()), 3
        ),
        "holdoutDelta": round(float(delta[-2:].mean()), 1),
        "worstSeasonDelta": int(delta.min()),
        "positiveSeasons": int((delta > 0).sum()),
        "negativeSeasons": int((delta < 0).sum()),
        "seasonDeltas": delta.astype(int).tolist(),
    }


def main() -> None:
    dynamic, _ = build_dynamic_history()
    data = add_fixture_history(add_targets(dynamic.reset_index(drop=True)))
    immediate, plan, frozen_captain = champion_forecasts(data)
    route_captain = selected_consensus_metric(data, immediate, frozen_captain)
    minute = minute_predictions(data)
    openfpl, _ = build_ensemble(data, immediate)
    open_rank = weekly_percentile(data, openfpl)
    dynamic_ranks = {}
    for strength in DYNAMIC_STRENGTHS:
        dynamic_score, _ = dynamic_route_score(data, immediate, strength)
        for downside in MINUTE_DOWNSIDES:
            adjusted = dynamic_score
            if downside > 0:
                adjusted, _ = captain_availability_score(
                    data, dynamic_score, minute, downside
                )
            dynamic_ranks[(strength, downside)] = weekly_percentile(data, adjusted)

    variants = {}
    configurations = {}
    for strength in DYNAMIC_STRENGTHS:
        for match_share in MATCH_SHARES:
            for player_share in PLAYER_SHARES:
                if match_share + player_share > 0.35:
                    continue
                for downside in MINUTE_DOWNSIDES:
                    if match_share == 0 and (strength != 0 or downside != 0):
                        continue
                    name = (
                        f"s{strength:.2f}-m{match_share:.3f}-"
                        f"p{player_share:.3f}-d{downside:.2f}"
                    )
                    route_share = 1.0 - match_share - player_share
                    variants[name] = (
                        route_share * route_captain
                        + match_share * dynamic_ranks[(strength, downside)]
                        + player_share * open_rank
                    )
                    configurations[name] = {
                        "dynamicStrength": strength,
                        "matchShare": match_share,
                        "playerShare": player_share,
                        "minuteDownside": downside,
                    }
    print(f"Screening {len(variants):,} fixed-XI captain configurations", flush=True)
    baseline, stats = lens.simulate_candidate(
        data,
        immediate,
        STRATEGY,
        chip_policy=lens.AUDITED_CHAMPION_CHIP_POLICY,
        plan_scores=plan,
        captain_scores=route_captain,
        audit_selections=True,
    )
    _, screening_rows = decision_evaluation(
        data, stats, baseline, route_captain, variants
    )
    screening = []
    for row in screening_rows:
        screening.append(
            {
                "name": row["name"],
                **configurations[row["name"]],
                "fixedXiAverageDelta": row["averageDelta"],
                "fixedXiDevelopmentDelta": row["developmentDelta"],
                "fixedXiDevelopmentStability": round(development_stability(row), 3),
                "fixedXiHoldoutDelta": row["holdoutDelta"],
                "fixedXiWorstSeasonDelta": row["worstSeasonDelta"],
                "fixedXiSeasonDeltas": row["seasonDeltas"],
                "changedCaptains": row["changedCaptains"],
            }
        )
    finalist_rows = sorted(
        screening,
        key=lambda row: (
            row["fixedXiDevelopmentStability"],
            row["fixedXiDevelopmentDelta"],
            row["fixedXiWorstSeasonDelta"],
        ),
        reverse=True,
    )[:FINALISTS]
    seasons = list(dict.fromkeys(data["season"].tolist()))
    exact = []
    for index, finalist in enumerate(finalist_rows, start=1):
        print(f"Exact captain finalist {index}/{len(finalist_rows)}: {finalist['name']}", flush=True)
        totals, _ = lens.simulate_candidate(
            data,
            immediate,
            STRATEGY,
            chip_policy=lens.AUDITED_CHAMPION_CHIP_POLICY,
            plan_scores=plan,
            captain_scores=variants[finalist["name"]],
        )
        exact.append({**finalist, **full_summary(totals, baseline, seasons)})
    selected = max(
        exact,
        key=lambda row: (
            row["developmentStability"],
            row["developmentDelta"],
            row["worstSeasonDelta"],
        ),
    )
    gate = {
        "developmentPositive": selected["developmentDelta"] > 0,
        "holdoutNonNegative": selected["holdoutDelta"] >= 0,
        "worstSeasonAtLeastMinusFive": selected["worstSeasonDelta"] >= -5,
        "positiveSeasonsAtLeastFour": selected["positiveSeasons"] >= 4,
    }
    result = {
        "schemaVersion": 1,
        "status": "large causal captain surface search",
        "selectionProtocol": (
            "All configurations are screened on fixed historical XIs. The 12 "
            "best first-six-season stability rows receive exact recursive replays; "
            "the last two seasons remain holdout for the final gate."
        ),
        "screenedConfigurations": len(screening),
        "exactRecursiveFinalists": len(exact),
        "baseline": season_summary(baseline, seasons),
        "selectedByDevelopmentOnly": selected,
        "gate": gate,
        "passed": all(gate.values()),
        "exactFinalists": exact,
        "screening": screening,
    }
    output = lens.ROOT / "analysis" / "data" / "captain_surface_search_v2.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"screened": len(screening), "selected": selected, "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
