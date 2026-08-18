"""Attribute forecast error where it can change an FPL decision.

This is deliberately an exposure-weighted audit.  Aggregate player-week MAE is
dominated by footballers the optimiser would never buy.  Here we replay the
frozen champion, tag its squad/XI/captain/transfers and the credible positional
frontier, then decompose each realised score into FPL scoring routes.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import calibrate_model as lens
from frontier_ranker_validation import STRATEGY
from multiscale_horizon_validation import add_targets
from wildcard_freehit_ablation import champion_forecasts


POSITION = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def _route_frame(data: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    work = data.copy()
    fixture_multiplier = 0.72 + 0.56 * work["fixture_now"].fillna(0.5)
    minutes_factor = work["expected_minutes"] / 90
    goal_points = work["position_id"].map({1: 6, 2: 6, 3: 5, 4: 4})
    clean_points = work["position_id"].map({1: 4, 2: 4, 3: 1, 4: 0})
    group = [work["season"], work["GW"], work["position_id"]]
    goal_vulnerability = (
        work["opponent_goal_vulnerability"]
        / work["opponent_goal_vulnerability"].groupby(group).transform("median").clip(lower=0.01)
    ).clip(0.68, 1.42)
    assist_vulnerability = (
        work["opponent_assist_vulnerability"]
        / work["opponent_assist_vulnerability"].groupby(group).transform("median").clip(lower=0.01)
    ).clip(0.72, 1.35)
    fixture_count = work["fixture_count"].clip(lower=1)
    work["forecast"] = scores
    work["pred_appearance"] = (
        work["play_probability"] + work["sixty_probability"]
    ) * fixture_count
    work["actual_appearance"] = (
        work["appearances_observed"] + work["sixty_observed"]
    )
    work["pred_attack"] = (
        work["goal_rate"] * goal_points * goal_vulnerability
        + work["assist_rate"] * 3 * assist_vulnerability
    ) * minutes_factor * fixture_multiplier * fixture_count
    work["actual_attack"] = work["goals"] * goal_points + work["assists"] * 3
    blended_clean = (
        0.82 * work["team_clean_probability"] + 0.18 * work["clean_sheet_rate"]
    ).clip(0.03, 0.78)
    work["pred_clean"] = (
        blended_clean * clean_points * work["sixty_probability"] * fixture_count
    )
    work["actual_clean"] = work["clean_sheets"] * clean_points
    work["pred_bonus"] = work["bonus_rate"] * minutes_factor * fixture_multiplier * fixture_count
    work["actual_bonus"] = work["bonus"]
    work["pred_known_routes"] = work[
        ["pred_appearance", "pred_attack", "pred_clean", "pred_bonus"]
    ].sum(axis=1)
    work["actual_known_routes"] = work[
        ["actual_appearance", "actual_attack", "actual_clean", "actual_bonus"]
    ].sum(axis=1)
    # This remainder contains saves, goals conceded, cards, own goals, penalties,
    # defensive contributions and ensemble corrections.  Keeping it explicit is
    # more honest than forcing it into one of the four identifiable routes.
    work["pred_other"] = work["forecast"] - work["pred_known_routes"]
    work["actual_other"] = work["points"] - work["actual_known_routes"]
    for route in ["appearance", "attack", "clean", "bonus", "other"]:
        work[f"error_{route}"] = work[f"pred_{route}"] - work[f"actual_{route}"]

    per_fixture_minutes = work["minutes"] / fixture_count
    minute_gap = work["expected_minutes"] - per_fixture_minutes
    no_show = (work["minutes"] <= 0) & (work["play_probability"] >= 0.55)
    reduced = (~no_show) & (minute_gap >= 25)
    surprise_start = (per_fixture_minutes - work["expected_minutes"] >= 25)
    route_columns = [f"error_{name}" for name in ["appearance", "attack", "clean", "bonus", "other"]]
    route_names = np.array(["minutes/appearance", "attacking returns", "team clean sheet", "bonus", "other events"])
    largest = np.abs(work[route_columns].to_numpy(float)).argmax(axis=1)
    work["primaryCause"] = route_names[largest]
    work.loc[no_show, "primaryCause"] = "no-show/availability"
    work.loc[reduced, "primaryCause"] = "reduced minutes/benching"
    work.loc[surprise_start, "primaryCause"] = "unexpected minutes"
    work["absoluteError"] = (work["forecast"] - work["points"]).abs()
    work["signedError"] = work["forecast"] - work["points"]
    work["position"] = work["position_id"].map(POSITION)
    return work


def _summarise(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"rows": 0}
    weight = frame["decisionWeight"].to_numpy(float)
    signed = frame["signedError"].to_numpy(float)
    absolute = frame["absoluteError"].to_numpy(float)
    denominator = max(weight.sum(), 1e-9)
    causes = []
    for cause, local in frame.groupby("primaryCause"):
        local_weight = local["decisionWeight"].to_numpy(float)
        causes.append(
            {
                "cause": str(cause),
                "weightedRows": round(float(local_weight.sum()), 1),
                "shareOfWeightedAbsoluteError": round(
                    float((local["absoluteError"].to_numpy(float) * local_weight).sum())
                    / max(float((absolute * weight).sum()), 1e-9),
                    4,
                ),
                "bias": round(
                    float((local["signedError"].to_numpy(float) * local_weight).sum())
                    / max(float(local_weight.sum()), 1e-9),
                    3,
                ),
            }
        )
    return {
        "rows": int(len(frame)),
        "weightedRows": round(float(weight.sum()), 1),
        "forecast": round(float(np.average(frame["forecast"], weights=weight)), 3),
        "actual": round(float(np.average(frame["points"], weights=weight)), 3),
        "bias": round(float((signed * weight).sum() / denominator), 3),
        "mae": round(float((absolute * weight).sum() / denominator), 3),
        "noShowRate": round(float(np.average(frame["minutes"].eq(0), weights=weight)), 4),
        "causes": sorted(causes, key=lambda row: row["shareOfWeightedAbsoluteError"], reverse=True),
    }


def main() -> None:
    original, _ = lens.load_or_build_prepared_history()
    data = add_targets(original.reset_index(drop=True))
    scores, plan, captain_scores = champion_forecasts(data)
    _, stats = lens.simulate_candidate(
        data,
        scores,
        STRATEGY,
        plan_scores=plan,
        captain_scores=captain_scores,
        audit_selections=True,
    )
    work = _route_frame(data, scores)
    work["owned"] = False
    work["xi"] = False
    work["captain"] = False
    work["transferIn"] = False
    work["transferOut"] = False
    context = lens.simulation_context(data)
    for season_index, season_context in enumerate(context["seasons"]):
        if season_context["season"] not in lens.EVALUATION_SEASONS:
            continue
        selections = {int(row["gw"]): row for row in stats[season_index]["selectionLog"]}
        transfers = {int(row["gw"]): row for row in stats[season_index]["transferLog"]}
        for gw in season_context["weeks"]:
            indices = np.asarray(season_context["weekIndices"][gw], dtype=int)
            elements = work.loc[indices, "element"].to_numpy(int)
            selection = selections[int(gw)]
            work.loc[indices, "owned"] = np.isin(elements, selection["squad"])
            work.loc[indices, "xi"] = np.isin(elements, selection["xi"])
            work.loc[indices, "captain"] = elements == int(selection["captain"])
            transfer = transfers.get(int(gw), {})
            work.loc[indices, "transferIn"] = np.isin(elements, transfer.get("inElements", []))
            work.loc[indices, "transferOut"] = np.isin(elements, transfer.get("outElements", []))

    valid = work["season"].isin(lens.EVALUATION_SEASONS) & work["fixture_count"].gt(0)
    rank = work.groupby(["season", "GW", "position_id"])["forecast"].rank(
        method="first", ascending=False
    )
    work["frontier"] = rank <= 20
    work["decisionWeight"] = (
        0.20 * work["frontier"].astype(float)
        + 0.50 * work["owned"].astype(float)
        + 1.00 * work["xi"].astype(float)
        + 1.50 * work["captain"].astype(float)
        + 1.25 * (work["transferIn"] | work["transferOut"]).astype(float)
    )
    boundary = work[valid & work["decisionWeight"].gt(0)].copy()
    exposure_specs = {
        "credibleFrontier": boundary["frontier"],
        "owned": boundary["owned"],
        "startingXI": boundary["xi"],
        "captain": boundary["captain"],
        "transferBoundary": boundary["transferIn"] | boundary["transferOut"],
        "allDecisionWeighted": pd.Series(True, index=boundary.index),
    }
    by_exposure = {name: _summarise(boundary[mask]) for name, mask in exposure_specs.items()}
    by_position = {
        position: _summarise(frame)
        for position, frame in boundary.groupby("position", sort=False)
    }
    by_season = {
        str(season).replace("-", "/"): _summarise(frame)
        for season, frame in boundary.groupby("season", sort=False)
    }
    worst = boundary.sort_values("absoluteError", ascending=False).head(30)
    result = {
        "status": "diagnostic; no model was promoted from this audit alone",
        "method": (
            "Frozen champion recursive replay. Decision weights are 0.2 frontier, "
            "0.5 owned, 1.0 XI, 1.5 captain and 1.25 transfer-boundary; tags stack. "
            "Route attribution uses only information and FPL events present in the historical snapshots."
        ),
        "exposure": by_exposure,
        "position": by_position,
        "season": by_season,
        "worstMisses": [
            {
                "season": str(row.season).replace("-", "/"),
                "gw": int(row.GW),
                "player": str(row.display_name),
                "position": str(row.position),
                "forecast": round(float(row.forecast), 2),
                "actual": round(float(row.points), 1),
                "minutes": round(float(row.minutes)),
                "expectedMinutes": round(float(row.expected_minutes)),
                "cause": str(row.primaryCause),
                "owned": bool(row.owned),
                "captain": bool(row.captain),
                "transferBoundary": bool(row.transferIn or row.transferOut),
            }
            for row in worst.itertuples()
        ],
    }
    output = lens.ROOT / "analysis" / "data" / "decision_boundary_error_autopsy.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"exposure": by_exposure, "position": by_position}, indent=2))


if __name__ == "__main__":
    main()
