"""Attribute regret along the actual recursive champion path."""

from __future__ import annotations

import json

import numpy as np

import calibrate_model as lens
from bench_efficiency_validation import championship_forecasts
from frontier_ranker_validation import STRATEGY


def main() -> None:
    data, _ = lens.load_or_build_prepared_history()
    data = data.reset_index(drop=True)
    scores, plan, captain_scores = championship_forecasts(data)
    totals, stats = lens.simulate_candidate(
        data,
        scores,
        STRATEGY,
        plan_scores=plan,
        captain_scores=captain_scores,
        audit_selections=True,
    )
    actual = data["points"].to_numpy(float)
    minutes = data["minutes"].to_numpy(float)
    rows = []
    context = lens.simulation_context(data)
    for season_index, season_context in enumerate(context["seasons"]):
        season = season_context["season"]
        if season not in lens.EVALUATION_SEASONS:
            continue
        selection_by_gw = {
            int(item["gw"]): item for item in stats[season_index]["selectionLog"]
        }
        transfer_by_gw = {
            int(item["gw"]): item for item in stats[season_index]["transferLog"]
        }
        captain_regret = 0.0
        lineup_regret = 0.0
        model_fresh_gap = 0.0
        transfer_gains = []
        for gw in season_context["weeks"]:
            selection = selection_by_gw[int(gw)]
            frame_indices = season_context["weekIndices"][gw]
            frame = data.loc[frame_indices]
            row_by_element = dict(
                zip(frame["element"].astype(int), frame.index.astype(int))
            )
            squad = {
                element: {
                    "position": int(data.loc[row_by_element[element], "position_id"]),
                    "team": int(data.loc[row_by_element[element], "team_id"]),
                }
                for element in selection["squad"]
                if element in row_by_element
            }
            xi = [element for element in selection["xi"] if element in row_by_element]
            if len(squad) == 15 and len(xi) == 11:
                best_captain = max(xi, key=lambda element: actual[row_by_element[element]])
                captain_regret += max(
                    0.0,
                    actual[row_by_element[best_captain]]
                    - actual[row_by_element[selection["captain"]]],
                )
                best_xi, best_bench = lens.choose_xi(
                    squad,
                    row_by_element,
                    actual,
                )
                actual_captains = sorted(
                    best_xi,
                    key=lambda element: actual[row_by_element[element]],
                    reverse=True,
                )
                oracle = lens.realised_week_breakdown(
                    best_xi,
                    best_bench,
                    actual_captains[0],
                    actual_captains[1],
                    squad,
                    row_by_element,
                    actual,
                    minutes,
                )["normal"]
                predicted_captains = sorted(
                    xi,
                    key=lambda element: captain_scores[row_by_element[element]],
                    reverse=True,
                )
                chosen_bench = [
                    element for element in selection["squad"] if element not in set(xi)
                ]
                chosen = lens.realised_week_breakdown(
                    xi,
                    chosen_bench,
                    predicted_captains[0],
                    predicted_captains[1],
                    squad,
                    row_by_element,
                    actual,
                    minutes,
                )["normal"]
                lineup_regret += max(0.0, oracle - chosen)

            # A weekly fresh squad is not feasible policy; this is a forecast/pool
            # diagnostic showing whether the model's best current legal squad is
            # materially better than the path it has reached.
            fresh_indices = lens.initial_squad(frame, plan)
            fresh_state = {
                int(data.loc[index, "element"]): {
                    "position": int(data.loc[index, "position_id"]),
                    "team": int(data.loc[index, "team_id"]),
                }
                for index in fresh_indices
            }
            fresh_xi, fresh_bench = lens.choose_xi(
                fresh_state, row_by_element, scores
            )
            fresh_captains = sorted(
                fresh_xi,
                key=lambda element: captain_scores[row_by_element[element]],
                reverse=True,
            )
            fresh_points = lens.realised_week_breakdown(
                fresh_xi,
                fresh_bench,
                fresh_captains[0],
                fresh_captains[1],
                fresh_state,
                row_by_element,
                actual,
                minutes,
            )["normal"]
            model_fresh_gap += fresh_points - float(
                stats[season_index]["weeklyPoints"][
                    season_context["weeks"].index(gw)
                ]
            )

            transfer = transfer_by_gw.get(int(gw))
            if transfer:
                future_gws = [
                    future for future in season_context["weeks"] if gw <= future <= gw + 5
                ]
                incoming = transfer.get("inElements", [])
                outgoing = transfer.get("outElements", [])
                incoming_points = 0.0
                outgoing_points = 0.0
                for future_gw in future_gws:
                    future_frame = data.loc[season_context["weekIndices"][future_gw]]
                    future_rows = dict(
                        zip(
                            future_frame["element"].astype(int),
                            future_frame.index.astype(int),
                        )
                    )
                    incoming_points += sum(
                        actual[future_rows[element]] for element in incoming if element in future_rows
                    )
                    outgoing_points += sum(
                        actual[future_rows[element]] for element in outgoing if element in future_rows
                    )
                transfer_gains.append(incoming_points - outgoing_points)
        rows.append(
            {
                "season": str(season).replace("-", "/"),
                "points": int(round(float(totals[season_index]))),
                "captainOracleRegret": round(captain_regret),
                "lineupCaptainOracleRegret": round(lineup_regret),
                "freshModelSquadActualGap": round(model_fresh_gap),
                "transfers": len(transfer_gains),
                "positiveSixWeekTransfers": int(sum(gain > 0 for gain in transfer_gains)),
                "negativeSixWeekTransfers": int(sum(gain < 0 for gain in transfer_gains)),
                "averageSixWeekTransferRawGain": round(
                    float(np.mean(transfer_gains)) if transfer_gains else 0.0,
                    2,
                ),
            }
        )
    result = {
        "method": (
            "Regret on the champion's actual recursive path. Oracles are diagnostic "
            "upper bounds, not achievable claims. Transfer gain is raw six-week "
            "incoming-minus-outgoing points and ignores wider squad opportunity cost."
        ),
        "average": {
            key: round(float(np.mean([row[key] for row in rows])), 1)
            for key in [
                "captainOracleRegret",
                "lineupCaptainOracleRegret",
                "freshModelSquadActualGap",
                "averageSixWeekTransferRawGain",
            ]
        },
        "seasons": rows,
    }
    output = lens.ROOT / "analysis" / "data" / "champion_path_regret.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
