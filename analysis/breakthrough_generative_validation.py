"""Validate the correlated route generator without changing transfer paths."""

from __future__ import annotations

import json

import numpy as np

import calibrate_model as lens
from breakthrough_engine import ScenarioConfig, sample_correlated_player_scenarios
from multiscale_horizon_validation import add_targets
from wildcard_freehit_ablation import champion_forecasts


def brier(probability: np.ndarray, outcome: np.ndarray) -> float:
    return float(np.mean((probability - outcome) ** 2))


def main() -> None:
    original, _ = lens.load_or_build_prepared_history()
    data = add_targets(original.reset_index(drop=True))
    immediate, _, _ = champion_forecasts(data)
    actual = data["points"].to_numpy(float)
    scenario_mean = np.zeros(len(data), dtype=float)
    scenario_p10 = np.zeros(len(data), dtype=float)
    scenario_p90 = np.zeros(len(data), dtype=float)
    scenario_blank = np.zeros(len(data), dtype=float)
    mean_drifts = []
    team_correlations = []

    for group_number, ((season, gw), frame) in enumerate(
        data.groupby(["season", "GW"], sort=False)
    ):
        indices = frame.index.to_numpy(int)
        bundle = sample_correlated_player_scenarios(
            frame.reset_index(drop=True),
            immediate[indices],
            ScenarioConfig(draws=128, seed=20260820 + group_number),
        )
        scenario_mean[indices] = bundle.means
        scenario_p10[indices] = np.quantile(bundle.points, 0.10, axis=0)
        scenario_p90[indices] = np.quantile(bundle.points, 0.90, axis=0)
        scenario_blank[indices] = np.mean(bundle.points < 3.0, axis=0)
        mean_drifts.append(bundle.metadata["meanAbsoluteMeanDrift"])
        for _, team_frame in frame.groupby("team_id"):
            local = [frame.index.get_loc(index) for index in team_frame.index]
            defenders = [
                index
                for index in local
                if int(frame.iloc[index]["position_id"]) in (1, 2)
                and float(frame.iloc[index]["sixty_probability"]) >= 0.5
            ][:4]
            if len(defenders) >= 2:
                corr = np.corrcoef(bundle.points[:, defenders].T)
                values = corr[np.triu_indices(len(defenders), 1)]
                team_correlations.extend(values[np.isfinite(values)].tolist())

    evaluation = data["season"].isin(lens.EVALUATION_SEASONS).to_numpy(bool)
    base_mae = float(np.mean(np.abs(immediate[evaluation] - actual[evaluation])))
    scenario_mae = float(
        np.mean(np.abs(scenario_mean[evaluation] - actual[evaluation]))
    )
    coverage = float(
        np.mean(
            (actual[evaluation] >= scenario_p10[evaluation])
            & (actual[evaluation] <= scenario_p90[evaluation])
        )
    )
    actual_blank = (actual[evaluation] < 3.0).astype(float)
    scenario_brier = brier(scenario_blank[evaluation], actual_blank)
    base_blank = data["blank_probability"].to_numpy(float)[evaluation]
    base_blank_brier = brier(base_blank, actual_blank)

    result = {
        "status": "scenario boundary validated for paired decisions",
        "method": (
            "Explicit appearance/60-minute draws, shared club attack shocks, "
            "shared clean-sheet outcomes and player-route residuals. The mean is "
            "audited but not substituted into the recursive transfer forecast."
        ),
        "rows": int(evaluation.sum()),
        "metrics": {
            "basePointMae": round(base_mae, 5),
            "scenarioMeanPointMae": round(scenario_mae, 5),
            "meanAbsoluteMeanDrift": round(float(np.mean(mean_drifts)), 5),
            "central80Coverage": round(coverage, 5),
            "baseBlankBrier": round(base_blank_brier, 6),
            "scenarioBlankBrier": round(scenario_brier, 6),
            "meanWithinTeamGkDefCorrelation": round(
                float(np.mean(team_correlations)), 5
            ),
        },
        "decision": (
            "Use for common-random-number action and chip comparisons only. "
            "Do not directly replace the frozen point mean."
        ),
        "passed": bool(
            np.mean(mean_drifts) <= 0.75
            and 0.55 <= coverage <= 0.95
            and np.mean(team_correlations) > 0.05
        ),
    }
    output = lens.ROOT / "analysis" / "data" / "breakthrough_generative_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

