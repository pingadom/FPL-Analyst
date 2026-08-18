"""Use multi-horizon ranks only where independent causal views agree."""

from __future__ import annotations

import json

import numpy as np

import calibrate_model as lens
from bench_efficiency_validation import championship_forecasts, variant_summary
from frontier_ranker_validation import STRATEGY
from listwise_ranker_validation import quantile_map
from multiscale_horizon_validation import (
    adaptive_value,
    add_targets,
    causal_online_ridge_horizons,
    causal_ridge_horizons,
    structural_horizons,
)
from multiscale_phase_validation import event_number


def group_percentile(data, values: np.ndarray) -> np.ndarray:
    result = np.zeros(len(values), dtype=float)
    for _, group_indices in data.groupby(
        ["season", "GW", "position_id"], sort=False
    ).groups.items():
        indices = np.asarray(group_indices, dtype=int)
        order = indices[np.argsort(values[indices], kind="stable")]
        if len(order) == 1:
            result[order] = 1.0
        else:
            result[order] = np.linspace(0.0, 1.0, len(order))
    return result


def selective_plan(
    baseline: np.ndarray,
    online: np.ndarray,
    prior: np.ndarray,
    baseline_rank: np.ndarray,
    online_rank: np.ndarray,
    prior_rank: np.ndarray,
    events: np.ndarray,
    maximum_share: float,
    rank_deadzone: float,
    continuous: bool,
    position_scale: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    online_delta = online_rank - baseline_rank
    prior_delta = prior_rank - baseline_rank
    agreement = (np.sign(online_delta) == np.sign(prior_delta)) & (
        np.abs(online_delta) >= rank_deadzone
    )
    if continuous:
        strength = np.clip(
            (np.abs(online_delta) - rank_deadzone) / max(0.01, 0.35 - rank_deadzone),
            0,
            1,
        )
    else:
        strength = np.ones(len(baseline), dtype=float)
    share = maximum_share * agreement * strength * (events >= 13)
    if position_scale is not None:
        share *= position_scale
    plan = baseline + share * (online - baseline)
    return plan, {
        "agreementRateAfterGW13": round(
            float(agreement[events >= 13].mean()), 4
        ),
        "meanEffectiveShareAfterGW13": round(
            float(share[events >= 13].mean()), 4
        ),
        "p90EffectiveShareAfterGW13": round(
            float(np.quantile(share[events >= 13], 0.90)), 4
        ),
    }


def main() -> None:
    original, _ = lens.load_or_build_prepared_history()
    data = add_targets(original.reset_index(drop=True))
    scores, baseline_plan, captain = championship_forecasts(data)
    structural = structural_horizons(data, scores)
    learned, _ = causal_ridge_horizons(data, structural)
    online, _ = causal_online_ridge_horizons(data, structural, learned)
    prior_value = quantile_map(
        data, adaptive_value(data, learned, 3.0), baseline_plan
    )
    online_value = quantile_map(
        data, adaptive_value(data, online, 3.0), baseline_plan
    )
    events = event_number(data)
    baseline_rank = group_percentile(data, baseline_plan)
    prior_rank = group_percentile(data, prior_value)
    online_rank = group_percentile(data, online_value)
    positions = data["position_id"].to_numpy(int)
    defender_scale = np.choose(
        positions - 1,
        [0.65, 1.25, 1.0, 1.0],
    )

    configs: dict[str, tuple[np.ndarray, dict]] = {
        "baseline": (baseline_plan, {}),
    }
    full_share = 0.10 * (events >= 13)
    configs["fullOnline10"] = (
        baseline_plan + full_share * (online_value - baseline_plan),
        {
            "agreementRateAfterGW13": 1.0,
            "meanEffectiveShareAfterGW13": 0.10,
            "p90EffectiveShareAfterGW13": 0.10,
        },
    )
    for name, maximum, deadzone, continuous, positional in [
        ("consensus15Deadzone10", 0.15, 0.10, False, False),
        ("consensus20Deadzone15", 0.20, 0.15, False, False),
        ("consensus25Deadzone20", 0.25, 0.20, False, False),
        ("continuousConsensus25", 0.25, 0.05, True, False),
        ("defenceWeightedConsensus20", 0.20, 0.10, True, True),
    ]:
        configs[name] = selective_plan(
            baseline_plan,
            online_value,
            prior_value,
            baseline_rank,
            online_rank,
            prior_rank,
            events,
            maximum,
            deadzone,
            continuous,
            defender_scale if positional else None,
        )

    seasons = list(dict.fromkeys(data["season"].tolist()))
    rows = []
    for name, (plan, gate_audit) in configs.items():
        print(f"Running {name}", flush=True)
        totals, stats = lens.simulate_candidate(
            data,
            scores,
            STRATEGY,
            plan_scores=plan,
            captain_scores=captain,
            tracked_player_name="Salah",
        )
        training = totals[: len(lens.TRAINING_SEASONS)]
        rows.append(
            {
                "name": name,
                "gate": gate_audit,
                "trainingStability": round(
                    float(training.mean() - 0.25 * training.std()), 3
                ),
                "summary": variant_summary(totals, stats, seasons),
            }
        )
    baseline = rows[0]
    selected = max(rows, key=lambda row: row["trainingStability"])
    paired = [
        {
            "season": old["season"],
            "baseline": old["points"],
            "challenger": new["points"],
            "delta": new["points"] - old["points"],
        }
        for old, new in zip(
            baseline["summary"]["seasons"], selected["summary"]["seasons"]
        )
    ]
    robust = bool(
        selected["name"] != "baseline"
        and selected["summary"]["average"] > baseline["summary"]["average"]
        and selected["summary"]["minimum"] >= baseline["summary"]["minimum"]
        and sum(row["delta"] > 0 for row in paired) >= 5
    )
    result = {
        "status": "selective multi-timescale policy challenger",
        "method": (
            "The multi-horizon rank can alter a player only after GW12, when "
            "prior-season and causal in-season models move that player in the "
            "same direction by more than a declared rank deadzone."
        ),
        "selected": selected,
        "pairedVsBaseline": paired,
        "robustPromotion": robust,
        "experiments": rows,
    }
    output = lens.ROOT / "analysis" / "data" / "selective_horizon_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "selected": selected["name"],
                "robustPromotion": robust,
                "paired": paired,
                "experiments": [
                    {
                        "name": row["name"],
                        "gate": row["gate"],
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
