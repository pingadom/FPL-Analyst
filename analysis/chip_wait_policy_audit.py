"""Causal sequential use-versus-wait policy for TC and Bench Boost."""

from __future__ import annotations

import json

import numpy as np

import calibrate_model as lens
from bench_efficiency_validation import championship_forecasts
from frontier_ranker_validation import STRATEGY


QUANTILES = (0.50, 0.65, 0.75, 0.85, 0.90)


def windows(season: str) -> list[tuple[int, int]]:
    return [(1, 19), (20, 38)] if season == "2025-26" else [(1, 38)]


def signal(row: dict, chip: str) -> float:
    if chip == "Triple Captain":
        return float(row["predictedTripleCaptainGain"]) * max(
            1, int(row["captainFixtureCount"])
        )
    return float(row["predictedBenchBoostGain"]) + 0.15 * int(
        row["benchDoubleCount"]
    )


def structural(row: dict, chip: str) -> bool:
    if chip == "Triple Captain":
        return int(row["captainFixtureCount"]) >= 2
    return int(row["benchDoubleCount"]) >= 1


def actual(row: dict, chip: str) -> float:
    key = (
        "actualTripleCaptainGain"
        if chip == "Triple Captain"
        else "actualBenchBoostGain"
    )
    return float(row[key])


def choose(rows: list[dict], chip: str, threshold: float) -> dict:
    ordered = sorted(rows, key=lambda row: int(row["gw"]))
    for row in ordered:
        if structural(row, chip) and signal(row, chip) >= threshold:
            return row
    # Chips expire. If no structural opportunity clears the bar, use the final
    # structural chance; only fall back to the last week when there was none.
    structural_rows = [row for row in ordered if structural(row, chip)]
    return structural_rows[-1] if structural_rows else ordered[-1]


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
    )
    opportunities = {
        season: stats[index]["chipOpportunities"]
        for index, season in enumerate(seasons)
    }

    policies = []
    for quantile in QUANTILES:
        season_rows = []
        prior_signals = {"Triple Captain": [], "Bench Boost": []}
        for season_index, season in enumerate(seasons):
            gains = {"Triple Captain": 0.0, "Bench Boost": 0.0}
            choices = []
            for chip in gains:
                threshold = (
                    float(np.quantile(prior_signals[chip], quantile))
                    if prior_signals[chip]
                    else (15.0 if chip == "Triple Captain" else 11.0)
                )
                for start, end in windows(season):
                    local = [
                        row
                        for row in opportunities[season]
                        if start <= int(row["gw"]) <= end
                    ]
                    selected = choose(local, chip, threshold)
                    gain = actual(selected, chip)
                    gains[chip] += gain
                    choices.append(
                        {
                            "chip": chip,
                            "window": [start, end],
                            "threshold": round(threshold, 3),
                            "gw": int(selected["gw"]),
                            "signal": round(signal(selected, chip), 3),
                            "actualGain": gain,
                            "structural": structural(selected, chip),
                        }
                    )
            season_rows.append(
                {
                    "season": season.replace("-", "/"),
                    "gain": round(sum(gains.values()), 1),
                    "choices": choices,
                }
            )
            for chip in prior_signals:
                prior_signals[chip].extend(
                    signal(row, chip)
                    for row in opportunities[season]
                    if structural(row, chip)
                )
        training = [
            row["gain"]
            for index, row in enumerate(season_rows)
            if index < len(lens.TRAINING_SEASONS)
        ]
        evaluation = [
            row["gain"]
            for index, row in enumerate(season_rows)
            if index >= len(lens.TRAINING_SEASONS)
        ]
        policies.append(
            {
                "quantile": quantile,
                "trainingAverageGain": round(float(np.mean(training)), 1),
                "trainingMinimumGain": round(float(np.min(training)), 1),
                "evaluationAverageGain": round(float(np.mean(evaluation)), 1),
                "evaluationMinimumGain": round(float(np.min(evaluation)), 1),
                "seasons": season_rows,
            }
        )
    selected = max(
        policies,
        key=lambda row: row["trainingAverageGain"]
        - 0.25 * np.std(
            [
                item["gain"]
                for index, item in enumerate(row["seasons"])
                if index < len(lens.TRAINING_SEASONS)
            ]
        ),
    )
    result = {
        "status": "training-selected fixed-path chip timing diagnostic",
        "warning": "TC/BB gains are evaluated on a fixed no-chip squad path; Wildcard and Free Hit require paired recursive evaluation.",
        "selected": selected,
        "policies": policies,
    }
    output = lens.ROOT / "analysis" / "data" / "chip_wait_policy_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "selectedQuantile": selected["quantile"],
                "trainingAverageGain": selected["trainingAverageGain"],
                "evaluationAverageGain": selected["evaluationAverageGain"],
                "evaluationMinimumGain": selected["evaluationMinimumGain"],
                "all": [
                    {
                        "quantile": row["quantile"],
                        "training": row["trainingAverageGain"],
                        "evaluation": row["evaluationAverageGain"],
                    }
                    for row in policies
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
