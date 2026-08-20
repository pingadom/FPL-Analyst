"""Causal recursive validation of fieldability-aware FPL decisions."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

import calibrate_model as lens
from availability_leak_audit import final_xi_after_autosubs
from captain_fixture_history_validation import add_fixture_history
from captain_route_consensus_validation import selected_consensus_metric
from frontier_ranker_validation import STRATEGY
from multiscale_horizon_validation import add_targets
from probabilistic_minutes_validation import season_summary
from wildcard_freehit_ablation import champion_forecasts


PENALTIES = (0.0, 1.5, 3.0)


def availability_summary(data, totals, stats, seasons: list[str]) -> dict:
    minutes = data["minutes"].to_numpy(float)
    fixture_count = data["fixture_count"].to_numpy(int)
    positions = data["position_id"].to_numpy(int)
    teams = data["team_id"].to_numpy(int)
    by_week = {
        (str(season), int(gw)): frame.index.to_numpy(int)
        for (season, gw), frame in data.groupby(["season", "GW"], sort=False)
    }
    no_fixture_xi = 0
    unfilled = 0
    total_no_shows = 0
    season_rows = []
    for stat in stats:
        season = str(stat["season"])
        if season not in lens.EVALUATION_SEASONS:
            continue
        season_unfilled = 0
        season_no_fixture = 0
        season_no_shows = 0
        for selection in stat["selectionLog"]:
            frame_indices = by_week[(season, int(selection["gw"]))]
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
            season_no_fixture += sum(
                fixture_count[row_by_element[element]] <= 0 for element in xi
            )
            season_no_shows += sum(
                minutes[row_by_element[element]] <= 0 for element in xi
            )
            final_xi, _ = final_xi_after_autosubs(
                xi, bench, squad, row_by_element, minutes
            )
            season_unfilled += sum(
                minutes[row_by_element[element]] <= 0 for element in final_xi
            )
        no_fixture_xi += season_no_fixture
        total_no_shows += season_no_shows
        unfilled += season_unfilled
        season_rows.append(
            {
                "season": season.replace("-", "/"),
                "points": int(totals[seasons.index(season)]),
                "knownNoFixtureXiSlots": int(season_no_fixture),
                "starterNoShows": int(season_no_shows),
                "unfilledStarterSlots": int(season_unfilled),
            }
        )
    return {
        "knownNoFixtureXiSlots": int(no_fixture_xi),
        "starterNoShows": int(total_no_shows),
        "unfilledStarterSlots": int(unfilled),
        "seasons": season_rows,
    }


def main() -> None:
    original, _ = lens.load_or_build_prepared_history()
    data = add_fixture_history(add_targets(original.reset_index(drop=True)))
    immediate, plan, frozen_captain = champion_forecasts(data)
    captain = selected_consensus_metric(data, immediate, frozen_captain)
    seasons = list(dict.fromkeys(data["season"].tolist()))

    rows = []
    for name, strategy in [
        ("frozen", STRATEGY),
        *[
            (
                f"fieldability-{penalty:g}",
                replace(
                    STRATEGY,
                    name=f"Fieldability {penalty:g}",
                    enforce_fieldability=True,
                    fieldability_penalty=penalty,
                ),
            )
            for penalty in PENALTIES
        ],
    ]:
        print(f"Running {name}", flush=True)
        totals, stats = lens.simulate_candidate(
            data,
            immediate,
            strategy,
            chip_policy=lens.AUDITED_CHAMPION_CHIP_POLICY,
            plan_scores=plan,
            captain_scores=captain,
            audit_selections=True,
        )
        row = {
            "name": name,
            **season_summary(totals, seasons),
            "availability": availability_summary(data, totals, stats, seasons),
            "totals": totals,
        }
        rows.append(row)

    control = rows[0]
    control_totals = control["totals"]
    for row in rows:
        delta = row["totals"][2:] - control_totals[2:]
        row["paired"] = {
            "averageDelta": round(float(delta.mean()), 1),
            "minimumDelta": int(delta.min()),
            "positiveSeasons": int((delta > 0).sum()),
            "negativeSeasons": int((delta < 0).sum()),
            "seasonDeltas": delta.astype(int).tolist(),
        }
        del row["totals"]

    # Structural exclusion with no soft coefficient is predeclared as the
    # safest candidate. Soft penalties are diagnostics and cannot select it.
    selected = rows[1]
    passed = bool(
        selected["availability"]["knownNoFixtureXiSlots"] == 0
        and selected["paired"]["averageDelta"] >= 0
        and selected["paired"]["minimumDelta"] >= -8
    )
    result = {
        "status": (
            "fieldability challenger passed historical engineering gate"
            if passed
            else "fieldability implemented; historical promotion gate failed"
        ),
        "method": (
            "Current blanks are hard-excluded from the XI and incoming transfer "
            "pool. Soft appearance-risk penalties are tested separately; exact "
            "historical injury-news states are not reconstructed."
        ),
        "baseline": rows[0],
        "selected": selected,
        "diagnostics": rows[2:],
        "passed": passed,
        "productionPromotion": False,
    }
    output = lens.ROOT / "analysis" / "data" / "breakthrough_fieldability_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "baseline": {
                    "average": rows[0]["average"],
                    **rows[0]["availability"],
                },
                "variants": [
                    {
                        "name": row["name"],
                        "average": row["average"],
                        "paired": row["paired"],
                        "availability": {
                            key: row["availability"][key]
                            for key in (
                                "knownNoFixtureXiSlots",
                                "starterNoShows",
                                "unfilledStarterSlots",
                            )
                        },
                    }
                    for row in rows[1:]
                ],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
