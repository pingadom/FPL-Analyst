"""Paired recursive validation of explicit action-specific value surfaces."""

from __future__ import annotations

import json

import numpy as np

import calibrate_model as lens
from action_specific_tenure_v2 import build_surfaces
from captain_fixture_history_validation import add_fixture_history
from dynamic_match_model_v2 import build_dynamic_history
from forecast_champion_v2 import selected_forecasts
from frontier_ranker_validation import STRATEGY
from listwise_ranker_validation import quantile_map
from multiscale_horizon_validation import add_targets
from multiscale_phase_validation import event_number
from probabilistic_minutes_validation import season_summary


def action_squads(
    data,
    squad_scores: np.ndarray,
    lineup_scores: np.ndarray,
    captain_scores: np.ndarray,
    bench_scores: np.ndarray,
    *,
    first_week_only: bool,
) -> dict[tuple[str, int], list[int]]:
    result = {}
    for season, season_frame in data.groupby("season", sort=False):
        weeks = list(dict.fromkeys(season_frame["GW"].astype(int).tolist()))
        selected_weeks = weeks[:1] if first_week_only else weeks
        for gw in selected_weeks:
            frame = season_frame[season_frame["GW"].eq(gw)]
            afcon_window = lens.AFCON_WINDOWS.get(str(season))
            afcon_risk = bool(
                afcon_window and afcon_window[0] - 1 <= int(gw) <= afcon_window[1]
            )
            excluded = set(
                frame.loc[
                    afcon_risk & frame["nationality"].isin(lens.AFCON_NATIONS),
                    "element",
                ].astype(int)
            )
            result[(str(season), int(gw))] = lens.initial_squad(
                frame,
                squad_scores,
                excluded_elements=excluded,
                captain_weight=STRATEGY.squad_captain_weight,
                bench_weight=STRATEGY.squad_bench_weight,
                minimum_spend_gap=STRATEGY.initial_spend_gap,
                bench_premium_limit=STRATEGY.bench_premium_limit,
                bench_premium_penalty=STRATEGY.bench_premium_penalty,
                exact_optimiser=True,
                lineup_scores=lineup_scores,
                captain_utility_scores=lineup_scores * (0.55 + 0.45 * captain_scores),
                bench_utility_scores=bench_scores,
                defence_correlation=STRATEGY.defence_residual_correlation,
            )
    return result


def paired_summary(totals, baseline, seasons) -> dict:
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
    immediate, plan, captain = selected_forecasts(data)
    surfaces, surface_audit = build_surfaces(data, immediate)
    transfer = quantile_map(data, surfaces["transfer"], plan)
    events = event_number(data)
    initial = action_squads(
        data,
        surfaces["wildcard"],
        surfaces["starting_xi"],
        captain,
        surfaces["bench"],
        first_week_only=True,
    )
    seasons = list(dict.fromkeys(data["season"].tolist()))
    print("Action-specific recursive control", flush=True)
    baseline, _ = lens.simulate_candidate(
        data,
        immediate,
        STRATEGY,
        chip_policy=lens.AUDITED_CHAMPION_CHIP_POLICY,
        plan_scores=plan,
        captain_scores=captain,
    )
    configs: list[tuple[str, np.ndarray, dict | None]] = [
        ("actionInitialSquad", plan, initial),
    ]
    for share in (0.025, 0.05, 0.10):
        configs.append((f"transferAll{share:.3f}", plan + share * (transfer - plan), None))
    for share in (0.05, 0.10):
        active = share * (events >= 13)
        configs.append((f"transferAfter12{share:.2f}", plan + active * (transfer - plan), None))
    configs.append(
        (
            "actionInitialPlusTransferAfter12",
            plan + 0.05 * (events >= 13) * (transfer - plan),
            initial,
        )
    )
    rows = []
    for name, candidate_plan, supplied_initial in configs:
        print(f"Running {name}", flush=True)
        totals, _ = lens.simulate_candidate(
            data,
            immediate,
            STRATEGY,
            chip_policy=lens.AUDITED_CHAMPION_CHIP_POLICY,
            plan_scores=candidate_plan,
            captain_scores=captain,
            initial_squads=supplied_initial,
        )
        rows.append({"name": name, **paired_summary(totals, baseline, seasons)})
    selected = max(
        rows,
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
        "status": "paired recursive action-specific horizon validation",
        "method": (
            "Captain/XI/Free-Hit use h1, bench uses h3, transfers use player-specific "
            "1/3/6/10 tenure and opening squad/Wildcard structure uses h10. This "
            "artifact tests initial-squad and transfer boundaries; chip timing is "
            "validated separately."
        ),
        "surfaceAudit": surface_audit,
        "baseline": season_summary(baseline, seasons),
        "selectedByDevelopmentOnly": selected,
        "gate": gate,
        "passed": all(gate.values()),
        "experiments": rows,
    }
    output = lens.ROOT / "analysis" / "data" / "action_specific_recursive_v2.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"baseline": result["baseline"], "selected": selected, "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
