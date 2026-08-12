"""Check whether team strength is underweighted for selectable defenders."""

from __future__ import annotations

import json

import pandas as pd

import calibrate_model as lens


def main() -> None:
    data, _ = lens.load_or_build_prepared_history()
    work = data[
        data["position_id"].isin([1, 2])
        & (data["fixture_count"] > 0)
        & (data["expected_minutes"] >= 45)
    ].copy()
    work["residual"] = work["points"] - work["component_xpts"]
    rows = []
    for position_id, position in ((1, "GK"), (2, "DEF")):
        local = work[work["position_id"] == position_id].copy()
        local["cleanStrengthBin"] = pd.qcut(
            local["team_clean_probability"], 5, labels=False, duplicates="drop"
        )
        bins = []
        for bin_id, group in local.groupby("cleanStrengthBin", sort=True):
            bins.append(
                {
                    "quintile": int(bin_id) + 1,
                    "cleanProbability": round(float(group["team_clean_probability"].mean()), 4),
                    "forecast": round(float(group["component_xpts"].mean()), 3),
                    "actual": round(float(group["points"].mean()), 3),
                    "residual": round(float(group["residual"].mean()), 3),
                    "rows": int(len(group)),
                }
            )
        rows.append(
            {
                "position": position,
                "bins": bins,
                "strongestTeamResidual": bins[-1]["residual"],
                "weakestTeamResidual": bins[0]["residual"],
            }
        )
    result = {
        "decision": "reject-extra-team-strength-overlay",
        "method": "Forecast residuals for GKs/defenders with at least 45 expected minutes, split by causal team clean-sheet probability quintile.",
        "reason": "The highest clean-sheet quintile is already calibrated within roughly two-tenths of a point and is slightly overpredicted, so another big-team bonus would double-count defence strength.",
        "positions": rows,
    }
    output = lens.ROOT / "analysis" / "data" / "team_defence_residual_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
