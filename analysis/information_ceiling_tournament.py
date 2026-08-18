"""Decompose the model's remaining decision regret by information source.

This file intentionally contains hindsight oracles.  They are never causal
features and can never be promoted.  Each oracle replaces only one forecast
route while leaving the remaining champion forecast intact.  We then replay it
twice: first with transfers frozen to the champion plan, and then with the
one-week information delta allowed to affect the recursive transfer decision.

The purpose is resource allocation: determine whether better deadline minutes,
team outcome probabilities, or player involvement estimates have enough value
to justify the next causal modelling phase.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

import calibrate_model as lens
from captain_route_consensus_validation import (
    selected_consensus_metric,
    weekly_percentile,
)
from captain_fixture_history_validation import add_fixture_history
from multiscale_horizon_validation import add_targets
from probabilistic_minutes_validation import season_summary
from wildcard_freehit_ablation import champion_forecasts
from frontier_ranker_validation import STRATEGY
from forecast_routes import route_components


@dataclass(frozen=True)
class OracleSpec:
    name: str
    description: str
    perfect_minutes: bool = False
    perfect_team_outcome: bool = False
    perfect_player_involvement: bool = False
    perfect_total_points: bool = False
    attainability: str = "diagnostic"


ORACLES = (
    OracleSpec(
        "perfectMinutes",
        "Actual appearance and minutes, with all return rates still forecast.",
        perfect_minutes=True,
        attainability="partly attainable from better deadline team news",
    ),
    OracleSpec(
        "perfectTeamOutcome",
        "Actual team goals and clean sheets, with scorer involvement and minutes still forecast.",
        perfect_team_outcome=True,
        attainability="partly reducible with market-calibrated team probabilities",
    ),
    OracleSpec(
        "perfectMinutesAndTeam",
        "Actual minutes plus actual team goals and clean sheets, but no knowledge of scorer identity.",
        perfect_minutes=True,
        perfect_team_outcome=True,
        attainability="upper bound for the proposed market-plus-lineup layer",
    ),
    OracleSpec(
        "perfectPlayerInvolvement",
        "Actual player goals and assists, with all other routes still forecast.",
        perfect_player_involvement=True,
        attainability="mostly football variance; not a realistic forecast target",
    ),
    OracleSpec(
        "perfectTotalPoints",
        "Actual FPL score is known before the deadline.",
        perfect_total_points=True,
        attainability="pure hindsight ceiling only",
    ),
)


def oracle_scores(
    data: pd.DataFrame,
    baseline: np.ndarray,
    spec: OracleSpec,
) -> np.ndarray:
    """Replace only the routes named by ``spec`` with their realised source."""
    if spec.perfect_total_points:
        return data["points"].to_numpy(float).copy()

    routes = route_components(data, baseline)
    appearance = routes["appearance"].copy()
    attack = routes["attack"].copy()
    clean = routes["clean"].copy()
    bonus = routes["bonus"].copy()
    fixture = routes["fixture"]

    expected_total_minutes = (
        data["expected_minutes"].to_numpy(float) * fixture
    )
    actual_total_minutes = data["minutes"].to_numpy(float)
    minute_ratio = np.divide(
        actual_total_minutes,
        np.maximum(expected_total_minutes, 8.0),
        out=np.zeros_like(actual_total_minutes),
        where=expected_total_minutes > 0,
    )
    # Exact current-week exposure can legitimately be far below expectation,
    # but an extreme DGW total should not multiply a one-match scoring rate
    # without bound.
    minute_ratio = np.clip(minute_ratio, 0.0, 2.25)

    if spec.perfect_minutes:
        appearance = (
            data["appearances_observed"].to_numpy(float)
            + data["sixty_observed"].to_numpy(float)
        )
        attack *= minute_ratio
        clean_probability = (
            0.82 * data["team_clean_probability"] + 0.18 * data["clean_sheet_rate"]
        ).clip(0.03, 0.78).to_numpy(float)
        clean = (
            clean_probability
            * routes["cleanPoints"]
            * data["sixty_observed"].to_numpy(float)
        )
        bonus *= minute_ratio

    if spec.perfect_team_outcome:
        expected_team_goals = (
            data["team_expected_goals_for"].to_numpy(float) * fixture
        )
        team_goal_ratio = np.divide(
            data["team_goals"].to_numpy(float),
            np.maximum(expected_team_goals, 0.25),
            out=np.zeros_like(expected_team_goals),
            where=expected_team_goals > 0,
        )
        # The oracle knows the team environment, not scorer identity.  Capping
        # prevents a 4-goal upset from silently becoming a scorer oracle too.
        attack *= np.clip(team_goal_ratio, 0.0, 2.75)
        sixty_exposure = (
            data["sixty_observed"].to_numpy(float)
            if spec.perfect_minutes
            else data["sixty_probability"].to_numpy(float) * fixture
        )
        clean_share = np.divide(
            data["team_clean_sheets"].to_numpy(float),
            np.maximum(data["team_games"].to_numpy(float), 1.0),
        )
        clean = routes["cleanPoints"] * sixty_exposure * clean_share

    if spec.perfect_player_involvement:
        attack = (
            data["goals"].to_numpy(float) * routes["goalPoints"]
            + 3.0 * data["assists"].to_numpy(float)
        )

    score = appearance + attack + clean + bonus + routes["residual"]
    blank = data["fixture_count"].to_numpy(int) == 0
    score[blank] = 0.0
    return np.clip(score, -2.0, 35.0)


def summary_with_delta(
    totals: np.ndarray,
    seasons: list[str],
    baseline: dict,
) -> dict:
    summary = season_summary(totals, seasons)
    deltas = [
        row["points"] - base["points"]
        for row, base in zip(summary["seasons"], baseline["seasons"])
    ]
    return {
        **summary,
        "averageDelta": round(summary["average"] - baseline["average"], 1),
        "minimumSeasonDelta": int(min(deltas)),
        "maximumSeasonDelta": int(max(deltas)),
        "positiveSeasons": int(sum(delta > 0 for delta in deltas)),
        "negativeSeasons": int(sum(delta < 0 for delta in deltas)),
        "seasonDeltas": deltas,
    }


def main() -> None:
    original, _ = lens.load_or_build_prepared_history()
    data = add_fixture_history(add_targets(original.reset_index(drop=True)))
    immediate, plan, frozen_captain = champion_forecasts(data)
    captain = selected_consensus_metric(data, immediate, frozen_captain)
    seasons = list(dict.fromkeys(data["season"].tolist()))

    baseline_no_chips, _ = lens.simulate_candidate(
        data,
        immediate,
        STRATEGY,
        plan_scores=plan,
        captain_scores=captain,
    )
    baseline_chips, _ = lens.simulate_candidate(
        data,
        immediate,
        STRATEGY,
        chip_policy=lens.AUDITED_CHAMPION_CHIP_POLICY,
        plan_scores=plan,
        captain_scores=captain,
    )
    baseline = {
        "noChips": season_summary(baseline_no_chips, seasons),
        "auditedChips": season_summary(baseline_chips, seasons),
    }

    observed = data["fixture_count"].to_numpy(int) > 0
    evaluation = data["season"].isin(lens.EVALUATION_SEASONS).to_numpy(bool)
    metric_mask = observed & evaluation
    rows = []
    for spec in ORACLES:
        print(f"Information ceiling: {spec.name}", flush=True)
        scores = oracle_scores(data, immediate, spec)
        score_delta = scores - immediate
        # Direct percentile is appropriate here because the source is an
        # acknowledged oracle.  It must never cross into live code.
        oracle_captain = weekly_percentile(data, scores)
        variants = {}
        for path_name, path_plan in (
            ("selectionOnly", plan),
            # Only the known current-week delta enters the multiweek transfer
            # objective.  We do not pretend current outcomes repeat for 4.5 GWs.
            ("recursivePath", plan + score_delta),
        ):
            no_chip_totals, _ = lens.simulate_candidate(
                data,
                scores,
                STRATEGY,
                plan_scores=path_plan,
                captain_scores=oracle_captain,
            )
            chip_totals, _ = lens.simulate_candidate(
                data,
                scores,
                STRATEGY,
                chip_policy=lens.AUDITED_CHAMPION_CHIP_POLICY,
                plan_scores=path_plan,
                captain_scores=oracle_captain,
            )
            variants[path_name] = {
                "noChips": summary_with_delta(
                    no_chip_totals, seasons, baseline["noChips"]
                ),
                "auditedChips": summary_with_delta(
                    chip_totals, seasons, baseline["auditedChips"]
                ),
            }
        error = scores[metric_mask] - data.loc[metric_mask, "points"].to_numpy(float)
        rows.append(
            {
                "name": spec.name,
                "description": spec.description,
                "attainability": spec.attainability,
                "forecastMetrics": {
                    "mae": round(float(np.mean(np.abs(error))), 4),
                    "bias": round(float(np.mean(error)), 4),
                    "correlation": round(
                        float(
                            np.corrcoef(
                                scores[metric_mask],
                                data.loc[metric_mask, "points"].to_numpy(float),
                            )[0, 1]
                        ),
                        4,
                    ),
                },
                "variants": variants,
            }
        )

    baseline_error = (
        immediate[metric_mask] - data.loc[metric_mask, "points"].to_numpy(float)
    )
    ranked = sorted(
        rows,
        key=lambda row: row["variants"]["recursivePath"]["auditedChips"]["averageDelta"],
        reverse=True,
    )
    result = {
        "status": "diagnostic information ceiling; every challenger uses hindsight",
        "warning": (
            "No oracle score, rank, field or selected squad is eligible for live use. "
            "This tournament chooses a research direction, not a production model."
        ),
        "method": (
            "Paired recursive replay on the leak-free route-captain champion. "
            "Each source replaces only identifiable forecast routes. Selection-only "
            "keeps the transfer plan frozen; recursive-path adds only the current-GW "
            "source delta to the transfer utility."
        ),
        "baselineForecastMetrics": {
            "mae": round(float(np.mean(np.abs(baseline_error))), 4),
            "bias": round(float(np.mean(baseline_error)), 4),
            "correlation": round(
                float(
                    np.corrcoef(
                        immediate[metric_mask],
                        data.loc[metric_mask, "points"].to_numpy(float),
                    )[0, 1]
                ),
                4,
            ),
        },
        "baseline": baseline,
        "experiments": rows,
        "rankingByRecursiveAuditedChipLift": [
            {
                "name": row["name"],
                "averageDelta": row["variants"]["recursivePath"]["auditedChips"]["averageDelta"],
                "minimumSeasonDelta": row["variants"]["recursivePath"]["auditedChips"]["minimumSeasonDelta"],
                "attainability": row["attainability"],
            }
            for row in ranked
        ],
    }
    output = lens.ROOT / "analysis" / "data" / "information_ceiling_tournament.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "baseline": baseline,
                "ranking": result["rankingByRecursiveAuditedChipLift"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
