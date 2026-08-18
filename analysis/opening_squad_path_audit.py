"""Inspect opening-squad forecast fit separately from recursive path effects."""

from __future__ import annotations

import json

import numpy as np

import calibrate_model as lens
from bench_efficiency_validation import championship_forecasts


def make_state(frame, indices):
    return {
        int(frame.loc[index, "element"]): {
            "position": int(frame.loc[index, "position_id"]),
            "team": int(frame.loc[index, "team_id"]),
        }
        for index in indices
    }


def fixed_squad_points(data, season_frame, indices, scores, captain_scores, weeks=6):
    state = make_state(season_frame, indices)
    total = 0.0
    weekly = []
    for gw in sorted(season_frame["GW"].unique())[:weeks]:
        frame = season_frame[season_frame["GW"].eq(gw)]
        row_by_element = dict(zip(frame["element"].astype(int), frame.index.astype(int)))
        active_state = {
            element: details for element, details in state.items() if element in row_by_element
        }
        if len(active_state) != 15:
            weekly.append(0.0)
            continue
        xi, bench = lens.choose_xi(active_state, row_by_element, scores)
        captain_order = sorted(
            xi, key=lambda element: captain_scores[row_by_element[element]], reverse=True
        )
        captain, vice = captain_order[:2]
        points = lens.realised_week_points(
            xi,
            bench,
            captain,
            vice,
            active_state,
            row_by_element,
            data["points"].to_numpy(float),
            data["minutes"].to_numpy(float),
        )
        total += points
        weekly.append(round(float(points), 1))
    return round(total, 1), weekly


def describe(data, frame, season_frame, indices, scores, plan_scores, captain_scores):
    state = make_state(frame, indices)
    rows = {int(frame.loc[index, "element"]): int(index) for index in indices}
    xi, bench = lens.choose_xi(state, rows, scores)
    captain = max(xi, key=lambda element: captain_scores[rows[element]])
    floors = {
        position: int(frame.loc[frame["position_id"].eq(position), "price"].min())
        for position in lens.SQUAD_QUOTAS
    }
    fixed_six, fixed_weekly = fixed_squad_points(
        data, season_frame, indices, scores, captain_scores
    )
    initial_bench = set(bench)
    future_bench_starts = 0
    future_bench_points = 0.0
    for gw in sorted(season_frame["GW"].unique())[1:6]:
        week = season_frame[season_frame["GW"].eq(gw)]
        week_rows = dict(zip(week["element"].astype(int), week.index.astype(int)))
        if not all(element in week_rows for element in state):
            continue
        week_xi, _ = lens.choose_xi(state, week_rows, scores)
        promoted = initial_bench.intersection(week_xi)
        future_bench_starts += len(promoted)
        future_bench_points += sum(
            float(data.loc[week_rows[element], "points"]) for element in promoted
        )
    return {
        "spend": round(float(frame.loc[indices, "price"].sum()) / 10, 1),
        "benchSpend": round(
            sum(int(frame.loc[rows[element], "price"]) for element in bench) / 10, 1
        ),
        "benchPremium": round(
            sum(
                max(
                    0,
                    int(frame.loc[rows[element], "price"])
                    - floors[int(state[element]["position"])],
                )
                for element in bench
            )
            / 10,
            1,
        ),
        "predictedGw1": round(
            float(sum(scores[rows[element]] for element in xi) + scores[rows[captain]]),
            2,
        ),
        "planXi": round(float(sum(plan_scores[rows[element]] for element in xi)), 2),
        "fixedSquadFirstSixActual": fixed_six,
        "fixedSquadWeeklyActual": fixed_weekly,
        "initialBenchStartsGw2To6": future_bench_starts,
        "initialBenchPointsGw2To6": round(future_bench_points, 1),
        "captain": str(frame.loc[rows[captain], "display_name"]),
        "xi": [str(frame.loc[rows[element], "display_name"]) for element in xi],
        "bench": [
            {
                "name": str(frame.loc[rows[element], "display_name"]),
                "price": round(float(frame.loc[rows[element], "price"]) / 10, 1),
                "playProbability": round(
                    100 * float(frame.loc[rows[element], "play_probability"]), 1
                ),
            }
            for element in bench
        ],
    }


def main() -> None:
    data, _ = lens.load_or_build_prepared_history()
    data = data.reset_index(drop=True)
    scores, plan_scores, captain_scores = championship_forecasts(data)
    captain_utility = scores * (0.55 + 0.45 * captain_scores)
    seasons = []
    for season, season_frame in data.groupby("season", sort=False):
        if season not in lens.EVALUATION_SEASONS:
            continue
        first = season_frame[season_frame["GW"].eq(season_frame["GW"].min())]
        variants = {
            "heuristic": lens.initial_squad(first, plan_scores),
            "exactNoRules": lens.initial_squad(
                first,
                plan_scores,
                exact_optimiser=True,
                lineup_scores=scores,
                captain_utility_scores=captain_utility,
            ),
            "exactHardRules": lens.initial_squad(
                first,
                plan_scores,
                minimum_spend_gap=5,
                bench_premium_limit=20,
                bench_premium_penalty=0.022,
                exact_optimiser=True,
                lineup_scores=scores,
                captain_utility_scores=captain_utility,
            ),
        }
        rows = {
            name: describe(
                data,
                first,
                season_frame,
                indices,
                scores,
                plan_scores,
                captain_scores,
            )
            for name, indices in variants.items()
        }
        seasons.append({"season": season.replace("-", "/"), "variants": rows})
        print(
            season,
            {
                name: (
                    row["spend"],
                    row["benchSpend"],
                    row["predictedGw1"],
                    row["fixedSquadFirstSixActual"],
                    row["captain"],
                )
                for name, row in rows.items()
            },
            flush=True,
        )
    result = {
        "status": "research-only; diagnostic",
        "method": "Opening squads are held fixed for six Gameweeks; weekly XI and captain are reselected causally. This removes transfer-path effects.",
        "seasons": seasons,
        "averages": {
            name: {
                "spend": round(
                    float(np.mean([row["variants"][name]["spend"] for row in seasons])),
                    2,
                ),
                "benchSpend": round(
                    float(
                        np.mean(
                            [row["variants"][name]["benchSpend"] for row in seasons]
                        )
                    ),
                    2,
                ),
                "predictedGw1": round(
                    float(
                        np.mean(
                            [row["variants"][name]["predictedGw1"] for row in seasons]
                        )
                    ),
                    2,
                ),
                "fixedSquadFirstSixActual": round(
                    float(
                        np.mean(
                            [
                                row["variants"][name]["fixedSquadFirstSixActual"]
                                for row in seasons
                            ]
                        )
                    ),
                    2,
                ),
            }
            for name in ("heuristic", "exactNoRules", "exactHardRules")
        },
    }
    output = lens.ROOT / "analysis" / "data" / "opening_squad_path_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["averages"], indent=2))


if __name__ == "__main__":
    main()
