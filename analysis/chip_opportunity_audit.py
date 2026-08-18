"""Isolate Triple Captain and Bench Boost timing from recursive chip paths."""

from __future__ import annotations

import json

import numpy as np

import calibrate_model as lens
from bench_efficiency_validation import championship_forecasts
from frontier_ranker_validation import STRATEGY


def best_in_window(rows: list[dict], key: str, start: int, end: int) -> dict | None:
    eligible = [row for row in rows if start <= int(row["gw"]) <= end]
    return max(eligible, key=lambda row: float(row[key]), default=None)


def best_structural(
    rows: list[dict], key: str, start: int, end: int, structural_key: str, minimum: int
) -> dict | None:
    eligible = [
        row
        for row in rows
        if start <= int(row["gw"]) <= end and int(row[structural_key]) >= minimum
    ]
    return max(eligible, key=lambda row: float(row[key]), default=None)


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
        tracked_player_name="Salah",
    )
    season_rows = []
    for season_index, season in enumerate(seasons):
        if season not in lens.EVALUATION_SEASONS:
            continue
        opportunities = stats[season_index]["chipOpportunities"]
        single_use = season != "2025-26"
        windows = [(1, 38)] if single_use else [(1, 19), (20, 38)]
        row = {"season": season.replace("-", "/"), "windows": []}
        for start, end in windows:
            predicted_tc = best_in_window(
                opportunities, "predictedTripleCaptainGain", start, end
            )
            oracle_tc = best_in_window(
                opportunities, "actualTripleCaptainGain", start, end
            )
            predicted_bb = best_in_window(
                opportunities, "predictedBenchBoostGain", start, end
            )
            oracle_bb = best_in_window(
                opportunities, "actualBenchBoostGain", start, end
            )
            structural_tc = best_structural(
                opportunities,
                "predictedTripleCaptainGain",
                start,
                end,
                "captainFixtureCount",
                2,
            )
            structural_bb = best_structural(
                opportunities,
                "predictedBenchBoostGain",
                start,
                end,
                "benchDoubleCount",
                1,
            )
            row["windows"].append(
                {
                    "start": start,
                    "end": end,
                    "predictedTriple": predicted_tc,
                    "oracleTriple": oracle_tc,
                    "predictedBench": predicted_bb,
                    "oracleBench": oracle_bb,
                    "structuralPredictedTriple": structural_tc,
                    "structuralPredictedBench": structural_bb,
                }
            )
        season_rows.append(row)

    predicted_tc_actual = [
        window["predictedTriple"]["actualTripleCaptainGain"]
        for row in season_rows
        for window in row["windows"]
    ]
    oracle_tc_actual = [
        window["oracleTriple"]["actualTripleCaptainGain"]
        for row in season_rows
        for window in row["windows"]
    ]
    predicted_bb_actual = [
        window["predictedBench"]["actualBenchBoostGain"]
        for row in season_rows
        for window in row["windows"]
    ]
    oracle_bb_actual = [
        window["oracleBench"]["actualBenchBoostGain"]
        for row in season_rows
        for window in row["windows"]
    ]
    structural_tc_actual = [
        window["structuralPredictedTriple"]["actualTripleCaptainGain"]
        for row in season_rows
        for window in row["windows"]
        if window["structuralPredictedTriple"] is not None
    ]
    structural_bb_actual = [
        window["structuralPredictedBench"]["actualBenchBoostGain"]
        for row in season_rows
        for window in row["windows"]
        if window["structuralPredictedBench"] is not None
    ]
    result = {
        "method": "Hold the no-chip recursive squad path fixed. In each legal chip window, compare the deadline-predicted best TC/BB week with the hindsight best week on that same path.",
        "averages": {
            "predictedTripleActualGain": round(float(np.mean(predicted_tc_actual)), 1),
            "oracleTripleActualGain": round(float(np.mean(oracle_tc_actual)), 1),
            "tripleTimingRegret": round(
                float(np.mean(oracle_tc_actual) - np.mean(predicted_tc_actual)), 1
            ),
            "predictedBenchActualGain": round(float(np.mean(predicted_bb_actual)), 1),
            "oracleBenchActualGain": round(float(np.mean(oracle_bb_actual)), 1),
            "benchTimingRegret": round(
                float(np.mean(oracle_bb_actual) - np.mean(predicted_bb_actual)), 1
            ),
            "structuralTripleWindows": len(structural_tc_actual),
            "structuralTripleActualGain": round(
                float(np.mean(structural_tc_actual)), 1
            )
            if structural_tc_actual
            else None,
            "structuralBenchWindows": len(structural_bb_actual),
            "structuralBenchActualGain": round(
                float(np.mean(structural_bb_actual)), 1
            )
            if structural_bb_actual
            else None,
        },
        "seasons": season_rows,
    }
    output = lens.ROOT / "analysis" / "data" / "chip_opportunity_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["averages"], indent=2))


if __name__ == "__main__":
    main()
