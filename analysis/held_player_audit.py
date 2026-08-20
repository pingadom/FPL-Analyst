"""Forensic audit of persistent player holds in a fixed Lens 8 policy replay.

The output distinguishes three ideas that are easy to conflate:

* a realised run of low FPL scores;
* evidence the model could see before the deadline; and
* hindsight replacement points, which explain cost but may not justify a decision.

No future result is used to label a warning as causal. Future replacement points
are retained in a separately named diagnostic field only.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import calibrate_model as lens
from elite_policy_search import candidate_from_weights


OUTPUT = lens.ROOT / "analysis" / "data" / "held_player_audit.json"
RETURN_THRESHOLD = 5.0
MIN_STARTING_DROUGHT = 4


def chip_policy_from_artifact(payload: dict) -> lens.ChipPolicy:
    enabled = payload.get("enabledChips")
    return lens.ChipPolicy(
        float(payload["wildcardGap"]),
        float(payload["freeHitGap"]),
        float(payload["benchScore"]),
        float(payload["tripleScore"]),
        float(payload["afconBonus"]),
        int(payload["firstWildcardMinGw"]),
        int(payload["secondWildcardMinGw"]),
        tuple(enabled) if enabled else None,
    )


def strategy_from_artifact(name: str) -> lens.SimulationStrategy:
    for strategy in (lens.WEEKLY_CHASE_STRATEGY, lens.JOINT_OPTION_STRATEGY):
        if strategy.name == name:
            return strategy
    raise ValueError(f"Unknown stored decision strategy: {name}")


def causal_decision_scores(
    data: pd.DataFrame,
    plan: np.ndarray,
    immediate: np.ndarray,
    strategy: lens.SimulationStrategy,
) -> np.ndarray:
    """Recreate the pre-deadline utility surface used to order transfer options."""
    if strategy.decision_immediate_share is None:
        decision = plan.copy()
    else:
        share = float(np.clip(strategy.decision_immediate_share, 0.0, 1.0))
        decision = (
            (1.0 - share) * plan
            + share * immediate * 4.5
            - strategy.decision_uncertainty_penalty
            * data["prediction_uncertainty"].to_numpy(float)
        )
    if strategy.enforce_fieldability:
        decision -= strategy.fieldability_penalty * (
            1.0 - data["play_probability"].to_numpy(float)
        )
    return decision


def realised_horizon(
    data: pd.DataFrame,
    season: str,
    weeks: list[int],
    week_position: int,
    element: int,
    horizon: int = 3,
) -> float:
    selected = weeks[week_position : week_position + horizon]
    rows = data[
        (data["season"] == season)
        & data["GW"].isin(selected)
        & (data["element"] == element)
    ]
    return float(rows["points"].sum())


def best_causal_replacement(
    frame: pd.DataFrame,
    row_by_element: dict[int, int],
    state_by_element: dict[int, dict],
    outgoing: int,
    bank: int,
    decision: np.ndarray,
    immediate: np.ndarray,
    plan: np.ndarray,
) -> dict | None:
    state = state_by_element[outgoing]
    budget = int(state["salePrice"]) + int(bank)
    position = int(state["position"])
    outgoing_team = int(state["team"])
    team_counts = Counter(int(item["team"]) for item in state_by_element.values())
    candidates: list[tuple[float, int]] = []
    for index in frame.index:
        index = int(index)
        element = int(frame.at[index, "element"])
        if element in state_by_element or int(frame.at[index, "position_id"]) != position:
            continue
        if int(frame.at[index, "price"]) > budget or int(frame.at[index, "fixture_count"]) <= 0:
            continue
        incoming_team = int(frame.at[index, "team_id"])
        if incoming_team != outgoing_team and team_counts[incoming_team] >= 3:
            continue
        candidates.append((float(decision[index]), index))
    if not candidates:
        return None
    _, index = max(candidates)
    held_index = row_by_element[outgoing]
    return {
        "element": int(frame.at[index, "element"]),
        "name": str(frame.at[index, "display_name"]),
        "team": str(frame.at[index, "team_name"]),
        "price": round(float(frame.at[index, "price"]) / 10, 1),
        "decisionDelta": round(float(decision[index] - decision[held_index]), 3),
        "planDelta": round(float(plan[index] - plan[held_index]), 3),
        "immediateDelta": round(float(immediate[index] - immediate[held_index]), 3),
    }


def week_records(
    data: pd.DataFrame,
    stats: list[dict],
    immediate: np.ndarray,
    plan: np.ndarray,
    decision: np.ndarray,
    strategy: lens.SimulationStrategy,
) -> list[dict]:
    records: list[dict] = []
    transfer_by_week = {
        (str(stat["season"]), int(move["gw"])): move
        for stat in stats
        for move in stat["transferLog"]
    }
    for stat in stats:
        season = str(stat["season"])
        if season not in lens.EVALUATION_SEASONS:
            continue
        weeks = [int(item["gw"]) for item in stat["selectionLog"]]
        for week_position, selection in enumerate(stat["selectionLog"]):
            gw = int(selection["gw"])
            frame = data[(data["season"] == season) & (data["GW"] == gw)]
            row_by_element = {
                int(frame.at[index, "element"]): int(index) for index in frame.index
            }
            state_by_element = {
                int(item["element"]): item for item in selection["persistentState"]
            }
            scoring_xi = set(int(value) for value in selection["xi"])
            scoring_squad = set(int(value) for value in selection["squad"])
            permanent = set(int(value) for value in selection["permanentSquad"])
            free_hit = scoring_squad != permanent
            transfer = transfer_by_week.get((season, gw))
            for element in permanent:
                index = row_by_element.get(element)
                if index is None:
                    continue
                replacement = best_causal_replacement(
                    frame,
                    row_by_element,
                    state_by_element,
                    element,
                    int(selection["bank"]),
                    decision,
                    immediate,
                    plan,
                )
                if replacement:
                    replacement["realisedNext3"] = round(
                        realised_horizon(
                            data,
                            season,
                            weeks,
                            week_position,
                            int(replacement["element"]),
                        ),
                        1,
                    )
                held_next3 = realised_horizon(
                    data, season, weeks, week_position, element
                )
                record = {
                    "season": season,
                    "gw": gw,
                    "weekPosition": week_position,
                    "element": element,
                    "name": str(data.at[index, "display_name"]),
                    "team": str(data.at[index, "team_name"]),
                    "position": int(data.at[index, "position_id"]),
                    "price": round(float(data.at[index, "price"]) / 10, 1),
                    "points": round(float(data.at[index, "points"]), 1),
                    "minutes": round(float(data.at[index, "minutes"]), 1),
                    "fixtureCount": int(data.at[index, "fixture_count"]),
                    "inScoringSquad": element in scoring_squad,
                    "inXI": element in scoring_xi,
                    "freeHit": free_hit,
                    "immediateXpts": round(float(immediate[index]), 3),
                    "planScore": round(float(plan[index]), 3),
                    "decisionScore": round(float(decision[index]), 3),
                    "componentXpts": round(float(data.at[index, "component_xpts"]), 3),
                    "playProbability": round(float(data.at[index, "play_probability"]), 4),
                    "startProbability": round(float(data.at[index, "start_probability"]), 4),
                    "expectedMinutes": round(float(data.at[index, "expected_minutes"]), 2),
                    "returnProbability": round(float(data.at[index, "return5_probability"]), 4),
                    "priceFallProbability": round(float(data.at[index, "price_fall_probability"]), 4),
                    "transferPressureRank": round(float(data.at[index, "transfer_pressure_rank"]), 4),
                    "recentRank": round(float(data.at[index, "recent"]), 4),
                    "minutesSecurityRank": round(float(data.at[index, "minutes_security"]), 4),
                    "officialZeroAvailabilitySignal": bool(
                        data.at[index, "official_zero_availability_signal"]
                    ),
                    "marketAvailabilityWarning": bool(
                        data.at[index, "market_availability_warning"]
                    ),
                    "severeAvailabilityWarning": bool(
                        data.at[index, "severe_availability_warning"]
                    ),
                    "bank": round(float(selection["bank"]) / 10, 1),
                    "transferMade": bool(transfer),
                    "heldRealisedNext3": round(held_next3, 1),
                    "replacement": replacement,
                }
                if replacement:
                    record["replacement"]["hindsightDeltaNext3"] = round(
                        float(replacement["realisedNext3"]) - held_next3, 1
                    )
                records.append(record)
    return records


def starting_droughts(records: list[dict], transfer_hurdle: float) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in records:
        grouped[(row["season"], row["element"])].append(row)

    droughts: list[dict] = []
    for (_, _), player_rows in grouped.items():
        player_rows.sort(key=lambda row: row["weekPosition"])
        streak: list[dict] = []

        def close_streak() -> None:
            nonlocal streak
            if len(streak) < MIN_STARTING_DROUGHT:
                streak = []
                return
            warnings: list[dict] = []
            for offset, row in enumerate(streak):
                labels = []
                if (
                    row["playProbability"] < 0.72
                    or row["expectedMinutes"] < 52
                    or row["severeAvailabilityWarning"]
                ):
                    labels.append("availability/minutes")
                if (
                    row["replacement"]
                    and row["replacement"]["planDelta"] >= transfer_hurdle
                ):
                    labels.append("model-rated replacement")
                if row["priceFallProbability"] >= 0.55 or row["transferPressureRank"] <= 0.15:
                    labels.append("market exit")
                if offset >= 2 and all(item["points"] < RETURN_THRESHOLD for item in streak[:offset]):
                    labels.append("known return drought")
                if labels:
                    warnings.append({"gw": row["gw"], "signals": labels})
            no_shows = sum(row["minutes"] <= 0 for row in streak)
            expected = float(sum(row["immediateXpts"] for row in streak))
            actual = float(sum(row["points"] for row in streak))
            stable_role = all(
                row["playProbability"] >= 0.78 and row["expectedMinutes"] >= 58
                for row in streak[: min(2, len(streak))]
            )
            replacement_opportunities = [
                row for row in streak
                if row["replacement"]
                and row["replacement"]["planDelta"] >= transfer_hurdle
            ]
            unallocated_opportunities = [
                row for row in replacement_opportunities if not row["transferMade"]
            ]
            objective_mismatches = [
                row for row in streak
                if row["replacement"]
                and row["replacement"]["decisionDelta"] >= transfer_hurdle
                and row["replacement"]["planDelta"] < transfer_hurdle
            ]
            if unallocated_opportunities:
                classification = "local transfer opportunity conflict"
                explanation = "An affordable replacement cleared the causal transfer objective and no competing transfer was made that deadline, yet the player was retained."
            elif replacement_opportunities:
                classification = "transfer prioritised elsewhere"
                explanation = "A local replacement cleared the hurdle, but the finite transfer was allocated elsewhere in the squad; this is a path-allocation question, not proof that the hold was irrational."
            elif no_shows >= 2 and any(
                "availability/minutes" in warning["signals"] for warning in warnings
            ):
                classification = "foreseeable availability decay"
                explanation = "Repeated absences were preceded or followed by a low causal play/minutes estimate; the response was too slow."
            elif stable_role and expected - actual >= 6:
                classification = "rough patch / outcome variance"
                explanation = "The player retained secure minutes and a competitive forecast. Low realised returns alone were not a defensible pre-deadline sell signal."
            elif warnings:
                classification = "mixed warning"
                explanation = "Some causal deterioration was visible, but it did not independently clear the configured transfer hurdle."
            else:
                classification = "unforeseeable or weak evidence"
                explanation = "The replay did not contain a strong pre-deadline sell signal; hindsight underperformance is not enough."
            best_hindsight = max(
                (
                    row["replacement"]["hindsightDeltaNext3"]
                    for row in streak
                    if row["replacement"]
                ),
                default=0.0,
            )
            droughts.append(
                {
                    "season": streak[0]["season"].replace("-", "/"),
                    "player": streak[0]["name"],
                    "team": streak[0]["team"],
                    "position": streak[0]["position"],
                    "startGw": streak[0]["gw"],
                    "endGw": streak[-1]["gw"],
                    "startingWeeks": len(streak),
                    "actualPoints": round(actual, 1),
                    "projectedPoints": round(expected, 1),
                    "projectionShortfall": round(expected - actual, 1),
                    "noShows": no_shows,
                    "averagePlayProbability": round(
                        float(np.mean([row["playProbability"] for row in streak])), 3
                    ),
                    "averageExpectedMinutes": round(
                        float(np.mean([row["expectedMinutes"] for row in streak])), 1
                    ),
                    "classification": classification,
                    "explanation": explanation,
                    "objectiveMismatchWeeks": [
                        row["gw"] for row in objective_mismatches
                    ],
                    "firstCausalWarning": warnings[0] if warnings else None,
                    "warningTimeline": warnings,
                    "bestHindsightReplacementDeltaNext3": round(best_hindsight, 1),
                    "weeks": [
                        {
                            "gw": row["gw"],
                            "points": row["points"],
                            "minutes": row["minutes"],
                            "xPts": row["immediateXpts"],
                            "playProbability": row["playProbability"],
                            "expectedMinutes": row["expectedMinutes"],
                            "replacement": row["replacement"],
                        }
                        for row in streak
                    ],
                }
            )
            streak = []

        previous_position: int | None = None
        for row in player_rows:
            consecutive = previous_position is None or row["weekPosition"] == previous_position + 1
            qualifies = (
                row["inXI"]
                and row["inScoringSquad"]
                and row["fixtureCount"] > 0
                and row["points"] < RETURN_THRESHOLD
            )
            if not consecutive or not qualifies:
                close_streak()
            if qualifies:
                streak.append(row)
            previous_position = row["weekPosition"]
        close_streak()
    droughts.sort(
        key=lambda row: (
            row["classification"] == "local transfer opportunity conflict",
            row["startingWeeks"],
            row["projectionShortfall"],
        ),
        reverse=True,
    )
    return droughts


def availability_summary(records: list[dict]) -> dict:
    starters = [
        row for row in records
        if row["inXI"] and row["inScoringSquad"] and row["fixtureCount"] > 0
    ]
    below_live_floor = [
        row for row in starters
        if row["startProbability"] < 0.70 or row["playProbability"] < 0.84
    ]
    severe = [row for row in starters if row["playProbability"] < 0.70]
    no_shows = [row for row in starters if row["minutes"] <= 0]
    return {
        "starterSelections": len(starters),
        "belowCurrentLiveFloor": len(below_live_floor),
        "belowCurrentLiveFloorShare": round(len(below_live_floor) / max(1, len(starters)), 4),
        "below70PlayProbability": len(severe),
        "starterNoShows": len(no_shows),
        "noShowRate": round(len(no_shows) / max(1, len(starters)), 4),
        "warning": "The carried weekly XI applies a 60% severe-risk safety floor. Rows below the stricter live 84% standard can be valid rotation risks; a tested 78% historical hard floor reduced points and was rejected.",
    }


def main() -> None:
    data, _ = lens.load_or_build_prepared_history()
    artifact = json.loads(lens.OUTPUT.read_text(encoding="utf-8-sig"))
    candidate = candidate_from_weights(artifact["model"]["weights"])
    immediate, plan, _ = lens.candidate_forecasts(
        data,
        candidate,
        robust_planning=bool(artifact["model"].get("robustPlanningEnabled", False)),
    )
    strategy = strategy_from_artifact(str(artifact["model"]["strategy"]))
    decision = causal_decision_scores(data, plan, immediate, strategy)
    fresh = lens.precompute_fresh_squads(data, plan)
    free_hit = lens.precompute_fresh_squads(data, immediate, one_week_only=True)
    totals, stats = lens.simulate_candidate(
        data,
        immediate,
        strategy,
        chip_policy=chip_policy_from_artifact(artifact["chipStrategy"]["policy"]),
        plan_scores=plan,
        fresh_squads=fresh,
        free_hit_squads=free_hit,
        audit_selections=True,
    )
    records = week_records(data, stats, immediate, plan, decision, strategy)
    droughts = starting_droughts(records, strategy.transfer_hurdle)
    evaluation_totals = [
        int(totals[list(dict.fromkeys(data["season"].tolist())).index(season)])
        for season in lens.EVALUATION_SEASONS
    ]
    classifications = Counter(row["classification"] for row in droughts)
    result = {
        "status": "Lens 8 fixed-final-policy held-player diagnostic",
        "modelVersion": artifact["model"]["version"],
        "strategy": strategy.name,
        "method": "Continuous starting-XI droughts are four or more consecutive active-fixture starts below five realised points. Causal warnings use only the current deadline's forecast, minutes, market and affordable-replacement values. Future three-week replacement points are explicitly hindsight diagnostics.",
        "replay": {
            "scope": "The final stored weights, chip policy and decision strategy are replayed unchanged across every evaluation season. This is a forensic diagnostic, not the published season-by-season walk-forward score.",
            "preRepairAveragePoints": 2096.6,
            "averagePoints": round(float(np.mean(evaluation_totals)), 1),
            "seasonPoints": evaluation_totals,
        },
        "availability": availability_summary(records),
        "droughtSummary": {
            "count": len(droughts),
            "classifications": dict(classifications),
            "totalStartingWeeks": int(sum(row["startingWeeks"] for row in droughts)),
            "totalProjectionShortfall": round(
                float(sum(row["projectionShortfall"] for row in droughts)), 1
            ),
        },
        "droughts": droughts,
        "repairedInconsistencies": [
            {
                "id": "historical-lineup-floor",
                "finding": "Exact rebuilds retain the causal XI floor and one exceptional-upside allowance. Carried squads use a 60% severe-risk safety floor; the proposed 78% weekly floor was rejected after losing 16.3 points in ablation.",
            },
            {
                "id": "sequential-transfer-availability",
                "finding": "Candidate screening remains availability-aware, while the final transfer hurdle uses causal planning points. Forcing the same penalised objective at both stages lost 26.4 points in the final high-precision ablation and was rejected.",
            },
            {
                "id": "live-null-status",
                "finding": "Nullable live chances now fall back by official status: available 100, doubtful 75, and injured/suspended/unavailable/not-in-squad 0.",
            },
            {
                "id": "live-overlay-xpts",
                "finding": "The deadline API now propagates an availability change through minutes, projected points, six-week value, captain utility, distributions, components and optimiser features.",
            },
            {
                "id": "historical-official-availability",
                "finding": "Trusted deadline xP-zero reduces historical start/play probabilities only when corroborated by extreme selling, a prior zero/no-show or a curtailed appearance; per-GW feed-quality guards disable corrupted all-zero snapshots.",
            },
        ],
        "availabilitySources": {
            "live": {
                "source": "Official FPL bootstrap-static player status, chance_of_playing_next_round, news and news_added",
                "use": "Forecast input immediately before every deadline",
                "url": "https://fantasy.premierleague.com/api/bootstrap-static/",
            },
            "historicalCausal": {
                "source": "Archived official FPL players_raw snapshots in the vaastav FPL repository commit history",
                "use": "Select only a commit timestamped before the simulated deadline and reject news_added after the deadline",
                "url": "https://github.com/vaastav/Fantasy-Premier-League",
            },
            "retrospectiveReason": {
                "source": "withqwerty availability-data weekly matchday classifications derived from Transfermarkt",
                "use": "Explain an absence after the event; never feed it into a causal backtest",
                "url": "https://github.com/withqwerty/availability-data",
            },
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "replay": result["replay"],
                "availability": result["availability"],
                "droughtSummary": result["droughtSummary"],
                "topDroughts": droughts[:12],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
