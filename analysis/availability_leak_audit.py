"""Audit starter availability separately from points forecasting quality.

The historical archive does not contain a complete deadline injury/news state.
This audit therefore reports three deliberately different quantities:

1. realised no-shows and legal autosub recovery in the champion replay;
2. a simple replacement-value estimate for slots that remained empty; and
3. a hindsight binary-appearance ceiling, which is research-only and must not
   be presented as a causal backtest.
"""

from __future__ import annotations

import json

import numpy as np

import calibrate_model as lens
from captain_fixture_history_validation import add_fixture_history
from captain_route_consensus_validation import (
    selected_consensus_metric,
)
from frontier_ranker_validation import STRATEGY
from multiscale_horizon_validation import add_targets
from probabilistic_minutes_validation import season_summary
from wildcard_freehit_ablation import champion_forecasts


def final_xi_after_autosubs(
    xi: list[int],
    bench: list[int],
    squad: dict[int, dict],
    row_by_element: dict[int, int],
    minutes: np.ndarray,
) -> tuple[list[int], int]:
    """Return the legal post-autosub XI and number of recovered starters."""

    def played(element: int) -> bool:
        return element in row_by_element and minutes[row_by_element[element]] > 0

    final_xi = list(xi)
    absent = [element for element in final_xi if not played(element)]
    original_absent = len(absent)
    for substitute in bench:
        if not absent or not played(substitute):
            continue
        for missing in list(absent):
            trial = [
                substitute if element == missing else element for element in final_xi
            ]
            if lens.legal_xi(trial, squad):
                final_xi = trial
                absent.remove(missing)
                break
    return final_xi, original_absent - len(absent)


def main() -> None:
    original, _ = lens.load_or_build_prepared_history()
    data = add_fixture_history(add_targets(original.reset_index(drop=True)))
    immediate, plan, frozen_captain = champion_forecasts(data)
    captain_metric = selected_consensus_metric(data, immediate, frozen_captain)
    seasons = list(dict.fromkeys(data["season"].tolist()))

    baseline_totals, baseline_stats = lens.simulate_candidate(
        data,
        immediate,
        STRATEGY,
        chip_policy=lens.AUDITED_CHAMPION_CHIP_POLICY,
        plan_scores=plan,
        captain_scores=captain_metric,
        audit_selections=True,
    )

    actual = data["points"].to_numpy(float)
    minutes = data["minutes"].to_numpy(float)
    positions = data["position_id"].to_numpy(int)
    teams = data["team_id"].to_numpy(int)
    names = data["display_name"].fillna("").to_numpy(str)
    by_week = {
        (str(season), int(gw)): frame.index.to_numpy(int)
        for (season, gw), frame in data.groupby(["season", "GW"], sort=False)
    }

    rows: list[dict] = []
    for stat in baseline_stats:
        season = str(stat["season"])
        if season not in lens.EVALUATION_SEASONS:
            continue
        starter_no_shows = 0
        recovered = 0
        unfilled = 0
        no_show_weeks = 0
        playing_starter_points: list[float] = []
        unfilled_detail: list[dict] = []

        for selection in stat["selectionLog"]:
            gw = int(selection["gw"])
            frame_indices = by_week[(season, gw)]
            row_by_element = {
                int(data.at[index, "element"]): int(index) for index in frame_indices
            }
            squad = {
                int(element): {
                    "position": int(positions[row_by_element[int(element)]]),
                    "team": int(teams[row_by_element[int(element)]]),
                }
                for element in selection["squad"]
                if int(element) in row_by_element
            }
            xi = [int(element) for element in selection["xi"]]
            bench = [int(element) for element in selection["bench"]]

            missing = [
                element
                for element in xi
                if minutes[row_by_element[element]] <= 0
            ]
            if missing:
                no_show_weeks += 1
            starter_no_shows += len(missing)
            for element in xi:
                index = row_by_element[element]
                if minutes[index] > 0:
                    playing_starter_points.append(float(actual[index]))

            final_xi, week_recovered = final_xi_after_autosubs(
                xi, bench, squad, row_by_element, minutes
            )
            week_unfilled = sum(
                minutes[row_by_element[element]] <= 0 for element in final_xi
            )
            recovered += week_recovered
            unfilled += week_unfilled
            if week_unfilled:
                unfilled_detail.append(
                    {
                        "gw": gw,
                        "slots": int(week_unfilled),
                        "originalNoShows": [
                            str(names[row_by_element[element]]) for element in missing
                        ],
                        "playingBench": [
                            str(names[row_by_element[element]])
                            for element in bench
                            if minutes[row_by_element[element]] > 0
                        ],
                    }
                )

        mean_starter = float(np.mean(playing_starter_points))
        median_starter = float(np.median(playing_starter_points))
        rows.append(
            {
                "season": season,
                "points": int(baseline_totals[seasons.index(season)]),
                "starterNoShows": int(starter_no_shows),
                "weeksWithNoShow": int(no_show_weeks),
                "autosubsRecovered": int(recovered),
                "unfilledStarterSlots": int(unfilled),
                "twoPointAppearanceFloor": int(2 * unfilled),
                "typicalStarterReplacementEstimate": round(unfilled * mean_starter, 1),
                "meanPlayingStarterPoints": round(mean_starter, 3),
                "medianPlayingStarterPoints": round(median_starter, 3),
                "chips": stat["chips"],
                "unfilledDetail": unfilled_detail,
            }
        )

    result = {
        "status": "availability audit of the effective, points-scoring XI",
        "baseline": season_summary(baseline_totals, seasons),
        "seasons": rows,
    }
    output = lens.ROOT / "analysis" / "data" / "availability_leak_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
