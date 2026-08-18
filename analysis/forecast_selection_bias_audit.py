"""Measure calibration where the optimiser actually selects players."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import calibrate_model as lens
from bench_efficiency_validation import championship_forecasts


def main() -> None:
    data, _ = lens.load_or_build_prepared_history()
    data = data.reset_index(drop=True)
    scores, _, _ = championship_forecasts(data)
    work = data[["season", "GW", "position_id", "points", "fixture_count"]].copy()
    work["forecast"] = scores
    work = work[work["fixture_count"] > 0].copy()
    work["rank"] = work.groupby(["season", "GW", "position_id"])["forecast"].rank(
        method="first", ascending=False
    )
    work["bucket"] = pd.cut(
        work["rank"],
        bins=[0, 3, 8, 15, 30, np.inf],
        labels=["top3", "4-8", "9-15", "16-30", "31+"],
    )
    rows = []
    for (season, bucket), frame in work.groupby(["season", "bucket"], observed=True):
        if season not in lens.EVALUATION_SEASONS:
            continue
        rows.append(
            {
                "season": season.replace("-", "/"),
                "bucket": str(bucket),
                "rows": len(frame),
                "forecast": round(float(frame["forecast"].mean()), 3),
                "actual": round(float(frame["points"].mean()), 3),
                "bias": round(float((frame["forecast"] - frame["points"]).mean()), 3),
            }
        )
    aggregate = []
    for bucket, frame in work[
        work["season"].isin(lens.EVALUATION_SEASONS)
    ].groupby("bucket", observed=True):
        aggregate.append(
            {
                "bucket": str(bucket),
                "rows": len(frame),
                "forecast": round(float(frame["forecast"].mean()), 3),
                "actual": round(float(frame["points"].mean()), 3),
                "bias": round(float((frame["forecast"] - frame["points"]).mean()), 3),
                "mae": round(float((frame["forecast"] - frame["points"]).abs().mean()), 3),
            }
        )
    result = {
        "method": "Calibration by within-deadline, within-position forecast rank; positive bias means overprediction.",
        "aggregate": aggregate,
        "seasons": rows,
    }
    output = lens.ROOT / "analysis" / "data" / "forecast_selection_bias_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
