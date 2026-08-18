"""Causal correction for selection-induced optimiser bias.

The top few players in a deadline forecast are selected from noisy estimates and
therefore have systematically positive forecast errors.  This experiment learns
only the *relative* bias of rank buckets, preserving each position/deadline's
overall forecast scale and the transfer hurdles calibrated on that scale.
"""

from __future__ import annotations

import json

import numpy as np

import calibrate_model as lens
from bench_efficiency_validation import championship_forecasts, variant_summary
from frontier_ranker_validation import STRATEGY


RANK_ENDS = np.array([3, 8, 15, 30, 10_000])
CONFIGS = (
    ("baseline", 0.00, 0.00),
    ("immediate50", 0.50, 0.00),
    ("plan50", 0.00, 0.50),
    ("both25", 0.25, 0.25),
    ("both50", 0.50, 0.50),
    ("both75", 0.75, 0.75),
)


def rank_buckets(data, forecast: np.ndarray) -> np.ndarray:
    buckets = np.full(len(data), 4, dtype=int)
    observed = data["fixture_count"].to_numpy(int) > 0
    for indices in data.groupby(
        ["season_order", "GW", "position_id"], sort=True
    ).groups.values():
        local = np.asarray(indices, dtype=int)
        local = local[observed[local]]
        order = local[np.argsort(forecast[local], kind="stable")[::-1]]
        ranks = np.arange(1, len(order) + 1)
        buckets[order] = np.searchsorted(RANK_ENDS, ranks, side="left")
    return buckets


def causal_relative_bias(
    data,
    forecast: np.ndarray,
    target: np.ndarray,
    target_end_gw: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    """Estimate rank-bucket bias using only labels known before each deadline."""
    orders = data["season_order"].to_numpy(int)
    gameweeks = data["GW"].to_numpy(int)
    positions = data["position_id"].to_numpy(int)
    observed = data["fixture_count"].to_numpy(int) > 0
    buckets = rank_buckets(data, forecast)
    residual = forecast - target
    corrected_bias = np.zeros(len(data), dtype=float)
    added = np.zeros(len(data), dtype=bool)
    bucket_sum = np.zeros((5, len(RANK_ENDS)), dtype=float)
    bucket_count = np.zeros((5, len(RANK_ENDS)), dtype=float)
    position_sum = np.zeros(5, dtype=float)
    position_count = np.zeros(5, dtype=float)
    global_bucket_sum = np.zeros(len(RANK_ENDS), dtype=float)
    global_bucket_count = np.zeros(len(RANK_ENDS), dtype=float)
    global_sum = 0.0
    global_count = 0.0
    audit = []

    for (season_order, gw), group_indices in data.groupby(
        ["season_order", "GW"], sort=True
    ).groups.items():
        known = (
            (~added)
            & observed
            & (
                (orders < int(season_order))
                | (
                    (orders == int(season_order))
                    & (target_end_gw < int(gw))
                )
            )
        )
        newly_known = np.flatnonzero(known)
        for index in newly_known:
            position = int(positions[index])
            bucket = int(buckets[index])
            value = float(residual[index])
            bucket_sum[position, bucket] += value
            bucket_count[position, bucket] += 1
            position_sum[position] += value
            position_count[position] += 1
            global_bucket_sum[bucket] += value
            global_bucket_count[bucket] += 1
            global_sum += value
            global_count += 1
        added[newly_known] = True

        indices = np.asarray(group_indices, dtype=int)
        global_mean = global_sum / global_count if global_count else 0.0
        for index in indices:
            if not observed[index]:
                continue
            position = int(positions[index])
            bucket = int(buckets[index])
            global_bucket_mean = (
                global_bucket_sum[bucket] / global_bucket_count[bucket]
                if global_bucket_count[bucket]
                else global_mean
            )
            position_mean = (
                position_sum[position] / position_count[position]
                if position_count[position]
                else global_mean
            )
            # Partial pooling prevents the first few deadlines or rare goalkeeper
            # buckets from producing unstable corrections.
            local_mean = (
                bucket_sum[position, bucket] + 80 * global_bucket_mean
            ) / (bucket_count[position, bucket] + 80)
            corrected_bias[index] = local_mean - position_mean
        if int(gw) == 1:
            audit.append(
                {
                    "season": str(data.loc[indices[0], "season"]),
                    "knownRows": int(added.sum()),
                    "globalMeanBias": round(global_mean, 4),
                }
            )
    return corrected_bias, audit


def main() -> None:
    data, _ = lens.load_or_build_prepared_history()
    data = data.reset_index(drop=True)
    scores, plan, captain = championship_forecasts(data)
    immediate_bias, immediate_audit = causal_relative_bias(
        data,
        scores,
        data["points"].to_numpy(float),
        data["GW"].to_numpy(int),
    )
    plan_bias, plan_audit = causal_relative_bias(
        data,
        plan,
        data["horizon_target"].to_numpy(float),
        data["horizon_target_end_gw"].to_numpy(int),
    )
    seasons = list(dict.fromkeys(data["season"].tolist()))
    rows = []
    raw_results = {}
    for name, immediate_share, plan_share in CONFIGS:
        corrected_scores = np.clip(scores - immediate_share * immediate_bias, 0, None)
        corrected_plan = np.clip(plan - plan_share * plan_bias, 0, None)
        print(f"Running {name}", flush=True)
        totals, stats = lens.simulate_candidate(
            data,
            corrected_scores,
            STRATEGY,
            plan_scores=corrected_plan,
            captain_scores=captain,
            tracked_player_name="Salah",
        )
        training = totals[: len(lens.TRAINING_SEASONS)]
        row = {
            "name": name,
            "immediateShare": immediate_share,
            "planShare": plan_share,
            "trainingStability": round(
                float(training.mean() - 0.25 * training.std()), 3
            ),
            "summary": variant_summary(totals, stats, seasons),
        }
        rows.append(row)
        raw_results[name] = (totals, stats)
    selected = max(rows, key=lambda row: row["trainingStability"])
    result = {
        "status": "training-selected causal optimiser-bias challenger",
        "method": (
            "Position-specific rank-bucket forecast residuals are learned only "
            "after their labels mature, partially pooled, centred to preserve "
            "the position-wide scale, and subtracted before recursive selection."
        ),
        "rankBuckets": ["1-3", "4-8", "9-15", "16-30", "31+"],
        "immediateAudit": immediate_audit,
        "horizonAudit": plan_audit,
        "selected": selected,
        "experiments": rows,
    }
    output = lens.ROOT / "analysis" / "data" / "optimizer_curse_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "selected": selected["name"],
                "selectedAverage": selected["summary"]["average"],
                "selectedMinimum": selected["summary"]["minimum"],
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
