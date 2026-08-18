"""Causal exposure audit for durable premium assets and season leaders."""

from __future__ import annotations

import json

import numpy as np

import calibrate_model as lens
from bench_efficiency_validation import championship_forecasts
from frontier_ranker_validation import STRATEGY


TRACKED = {
    "Salah": "salah",
    "Haaland": "haaland",
    "Fernandes": "fernandes",
}


def main() -> None:
    data, _ = lens.load_or_build_prepared_history()
    data = data.reset_index(drop=True)
    scores, plan_scores, captain_scores = championship_forecasts(data)
    seasons = list(dict.fromkeys(data["season"].tolist()))
    _, stats = lens.simulate_candidate(
        data,
        scores,
        STRATEGY,
        plan_scores=plan_scores,
        captain_scores=captain_scores,
        audit_selections=True,
    )
    rows = []
    leader_rows = []
    for season_index, season in enumerate(seasons):
        if season not in lens.EVALUATION_SEASONS:
            continue
        frame = data[data["season"].eq(season)].copy()
        frame["forecast"] = scores[frame.index]
        selections = {
            int(row["gw"]): row for row in stats[season_index]["selectionLog"]
        }
        season_totals = (
            frame.groupby(["element", "display_name"], sort=False)["points"]
            .sum()
            .sort_values(ascending=False)
        )
        leaders = set(
            int(element) for element, _ in season_totals.head(5).index.tolist()
        )
        leader_weeks = 0
        leader_squad_weeks = 0
        for _, week in frame.groupby("GW", sort=True):
            selected = set(selections[int(week["GW"].iloc[0])]["squad"])
            present = leaders.intersection(set(week["element"].astype(int)))
            leader_weeks += len(present)
            leader_squad_weeks += len(present.intersection(selected))
        leader_rows.append(
            {
                "season": season.replace("-", "/"),
                "topFiveHindsightSquadRate": round(
                    100 * leader_squad_weeks / max(1, leader_weeks), 1
                ),
                "leaders": [str(name) for _, name in season_totals.head(5).index],
            }
        )

        for label, token in TRACKED.items():
            matches = frame[
                frame["display_name"].astype(str).str.contains(
                    token, case=False, na=False
                )
            ]
            if matches.empty:
                continue
            element = int(
                matches.groupby("element")["points"].sum().idxmax()
            )
            player = matches[matches["element"].eq(element)].sort_values("GW")
            eligible = player[player["fixture_count"] > 0]
            squad_weeks = []
            xi_weeks = []
            captain_weeks = []
            omitted_points = 0.0
            ranks = []
            for _, item in eligible.iterrows():
                gw = int(item["GW"])
                decision = selections[gw]
                in_squad = element in decision["squad"]
                squad_weeks.append(in_squad)
                xi_weeks.append(element in decision["xi"])
                captain_weeks.append(element == decision["captain"])
                if not in_squad:
                    omitted_points += float(item["points"])
                local = frame[
                    frame["GW"].eq(gw)
                    & frame["position_id"].eq(int(item["position_id"]))
                ]
                ranks.append(
                    int(
                        local["forecast"].rank(
                            method="min", ascending=False
                        ).loc[item.name]
                    )
                )
            rows.append(
                {
                    "season": season.replace("-", "/"),
                    "asset": label,
                    "seasonPoints": round(float(eligible["points"].sum())),
                    "averagePrice": round(float(eligible["price"].mean()) / 10, 2),
                    "averagePositionForecastRank": round(float(np.mean(ranks)), 1),
                    "squadRate": round(100 * float(np.mean(squad_weeks)), 1),
                    "xiRate": round(100 * float(np.mean(xi_weeks)), 1),
                    "captainRate": round(100 * float(np.mean(captain_weeks)), 1),
                    "rawPointsWhileOmitted": round(omitted_points),
                    "warning": "Omitted raw points are not transfer regret; the replacement also scores and price affects the rest of the squad.",
                }
            )
    result = {
        "method": "Replay the frozen champion once and inspect its actual weekly selections. Named assets are sanity checks, never forced constraints.",
        "assets": rows,
        "seasonLeaders": leader_rows,
    }
    output = lens.ROOT / "analysis" / "data" / "premium_asset_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
