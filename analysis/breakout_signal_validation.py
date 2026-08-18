"""Causal emerging-role challenger for fast breakout recognition."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import calibrate_model as lens
from bench_efficiency_validation import championship_forecasts, variant_summary
from frontier_ranker_validation import STRATEGY
from listwise_ranker_validation import quantile_map


SHARES = (0.05, 0.10, 0.15, 0.25)


def grouped_percentile(data: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    series = pd.Series(values, index=data.index)
    return series.groupby(
        [data["season"], data["GW"], data["position_id"]], sort=False
    ).rank(method="average", pct=True).fillna(0.5).to_numpy(float)


def main() -> None:
    data, _ = lens.load_or_build_prepared_history()
    data = data.reset_index(drop=True)
    scores, plan, captain = championship_forecasts(data)
    underlying_acceleration = grouped_percentile(
        data,
        data["recent_underlying_raw"].to_numpy(float)
        - data["long_underlying_raw"].to_numpy(float),
    )
    points_acceleration = grouped_percentile(
        data,
        data["recent_raw"].to_numpy(float)
        - data["long_raw"].to_numpy(float),
    )
    role_security = grouped_percentile(
        data,
        0.60 * data["start_probability"].to_numpy(float)
        + 0.40 * data["sixty_probability"].to_numpy(float),
    )
    market_information = grouped_percentile(
        data,
        data["transfer_pressure_rank"].to_numpy(float)
        + 0.50 * data["price_rise_probability"].to_numpy(float)
        - 0.50 * data["price_fall_probability"].to_numpy(float),
    )
    team_attack = grouped_percentile(data, data["team_attack_raw"].to_numpy(float))
    baseline_rank = grouped_percentile(data, scores)
    # Baseline quality remains the majority vote. The additional inputs only
    # distinguish players whose underlying role is improving before long-run
    # averages have caught up.
    breakout_raw = (
        0.55 * baseline_rank
        + 0.15 * underlying_acceleration
        + 0.10 * points_acceleration
        + 0.10 * role_security
        + 0.06 * market_information
        + 0.04 * team_attack
    )
    breakout_mapped_score = quantile_map(data, breakout_raw, scores)
    breakout_mapped_plan = quantile_map(data, breakout_raw, plan)
    seasons = list(dict.fromkeys(data["season"].tolist()))
    configs = [("baseline", scores, plan)]
    for share in SHARES:
        configs.append(
            (
                f"breakout{int(share * 100)}",
                (1 - share) * scores + share * breakout_mapped_score,
                (1 - share) * plan + share * breakout_mapped_plan,
            )
        )
    rows = []
    raw_results = {}
    for name, challenger_score, challenger_plan in configs:
        print(f"Running {name}", flush=True)
        totals, stats = lens.simulate_candidate(
            data,
            challenger_score,
            STRATEGY,
            plan_scores=challenger_plan,
            captain_scores=captain,
            tracked_player_name="Salah",
        )
        training = totals[: len(lens.TRAINING_SEASONS)]
        row = {
            "name": name,
            "trainingStability": round(
                float(training.mean() - 0.25 * training.std()), 3
            ),
            "summary": variant_summary(totals, stats, seasons),
        }
        rows.append(row)
        raw_results[name] = (
            totals,
            stats,
            challenger_score,
            challenger_plan,
        )
    selected = max(rows, key=lambda row: row["trainingStability"])
    _, _, selected_score, selected_plan = raw_results[selected["name"]]
    print(f"Running selected with audited chips: {selected['name']}", flush=True)
    chip_totals, chip_stats = lens.simulate_candidate(
        data,
        selected_score,
        STRATEGY,
        chip_policy=lens.AUDITED_CHAMPION_CHIP_POLICY,
        plan_scores=selected_plan,
        captain_scores=captain,
        tracked_player_name="Salah",
    )
    result = {
        "status": "training-selected causal breakout challenger",
        "method": (
            "Within-deadline role acceleration combines improving underlying "
            "output, recent points, start security, transfer/price information and "
            "team attack while retaining a 55% baseline-quality anchor."
        ),
        "selected": selected,
        "selectedWithAuditedChips": variant_summary(chip_totals, chip_stats, seasons),
        "experiments": rows,
    }
    output = lens.ROOT / "analysis" / "data" / "breakout_signal_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "selected": selected["name"],
                "selectedAverage": selected["summary"]["average"],
                "selectedWithChips": result["selectedWithAuditedChips"]["average"],
                "experiments": [
                    {
                        "name": row["name"],
                        "trainingStability": row["trainingStability"],
                        "average": row["summary"]["average"],
                        "minimum": row["summary"]["minimum"],
                    }
                    for row in rows
                ],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
