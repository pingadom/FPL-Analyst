"""Measure residual dependence needed by risk-aware squad scenarios."""

from __future__ import annotations

import json
from itertools import combinations

import numpy as np
import pandas as pd

import calibrate_model as lens
from bench_efficiency_validation import championship_forecasts


def correlation(pairs: list[tuple[float, float]]) -> dict:
    if len(pairs) < 2:
        return {"pairs": len(pairs), "correlation": None}
    values = np.asarray(pairs, dtype=float)
    return {
        "pairs": len(pairs),
        "correlation": round(float(np.corrcoef(values[:, 0], values[:, 1])[0, 1]), 4),
        "jointDownsideRate": round(
            float(np.mean((values[:, 0] < -0.5) & (values[:, 1] < -0.5))), 4
        ),
        "jointUpsideRate": round(
            float(np.mean((values[:, 0] > 0.5) & (values[:, 1] > 0.5))), 4
        ),
    }


def main() -> None:
    data, _ = lens.load_or_build_prepared_history()
    data = data.reset_index(drop=True)
    scores, _, _ = championship_forecasts(data)
    work = data[
        data["season"].isin(lens.EVALUATION_SEASONS)
        & data["fixture_count"].gt(0)
        & data["expected_minutes"].ge(30)
    ].copy()
    work["z"] = (
        work["points"].to_numpy(float) - scores[work.index.to_numpy(int)]
    ) / work["prediction_uncertainty"].clip(lower=1.0).to_numpy(float)
    same_team = {"defenceDefence": [], "attackAttack": [], "mixed": []}
    for _, group in work.groupby(["season", "GW", "team_id"], sort=False):
        rows = list(group[["position_id", "z"]].itertuples(index=False, name=None))
        for left, right in combinations(rows, 2):
            pair = (float(left[1]), float(right[1]))
            if int(left[0]) <= 2 and int(right[0]) <= 2:
                same_team["defenceDefence"].append(pair)
            elif int(left[0]) >= 3 and int(right[0]) >= 3:
                same_team["attackAttack"].append(pair)
            else:
                same_team["mixed"].append(pair)

    # A deterministic different-team control, matched within deadline and broad
    # role. It is not used for fitting, only to distinguish shared football shocks
    # from common forecast misspecification.
    controls = {"defenceDefence": [], "attackAttack": [], "mixed": []}
    for _, group in work.groupby(["season", "GW"], sort=False):
        rows = list(
            group[["team_id", "position_id", "z"]]
            .sort_values(["position_id", "team_id"])
            .itertuples(index=False, name=None)
        )
        for index in range(0, len(rows) - 1, 2):
            left, right = rows[index], rows[index + 1]
            if int(left[0]) == int(right[0]):
                continue
            pair = (float(left[2]), float(right[2]))
            if int(left[1]) <= 2 and int(right[1]) <= 2:
                controls["defenceDefence"].append(pair)
            elif int(left[1]) >= 3 and int(right[1]) >= 3:
                controls["attackAttack"].append(pair)
            else:
                controls["mixed"].append(pair)

    work["uncertaintyBin"] = pd.qcut(
        work["prediction_uncertainty"], 5, labels=False, duplicates="drop"
    )
    calibration = []
    for bin_id, group in work.groupby("uncertaintyBin", sort=True):
        residual = group["points"].to_numpy(float) - scores[group.index.to_numpy(int)]
        calibration.append(
            {
                "quintile": int(bin_id) + 1,
                "predictedStd": round(float(group["prediction_uncertainty"].mean()), 3),
                "residualStd": round(float(np.std(residual)), 3),
                "mae": round(float(np.mean(np.abs(residual))), 3),
                "rows": int(len(group)),
            }
        )
    result = {
        "method": (
            "Champion residuals divided by deadline-known uncertainty; pairwise "
            "dependence for players projected at least 30 minutes."
        ),
        "sameTeam": {key: correlation(value) for key, value in same_team.items()},
        "differentTeamControl": {
            key: correlation(value) for key, value in controls.items()
        },
        "uncertaintyCalibration": calibration,
    }
    output = lens.ROOT / "analysis" / "data" / "correlated_residual_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
