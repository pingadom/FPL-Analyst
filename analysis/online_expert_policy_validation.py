"""Causal online expert weighting for changing Premier League regimes."""

from __future__ import annotations

import json
import math

import numpy as np

import calibrate_model as lens
from bench_efficiency_validation import variant_summary
from captain_ranker_validation import rank_blend
from frontier_ranker_validation import PLAYER_CANDIDATE, STRATEGY
from listwise_ranker_validation import quantile_map


CONFIGS = (
    ("fast", 0.75, 0.50, 1.50),
    ("balanced", 0.90, 0.75, 2.00),
    ("slow", 0.96, 1.00, 3.00),
)


def sigmoid_scalar(value: float) -> float:
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1 + exponential)


def top_value(
    indices: np.ndarray,
    forecast: np.ndarray,
    target: np.ndarray,
    count: int,
) -> float:
    if not len(indices):
        return 0.0
    chosen = indices[np.argsort(forecast[indices], kind="stable")[-count:]]
    return float(np.mean(target[chosen]))


def causal_mix(
    data,
    base: np.ndarray,
    challenger: np.ndarray,
    target: np.ndarray,
    target_end: np.ndarray,
    decay: float,
    temperature: float,
    delayed: bool,
) -> tuple[np.ndarray, dict]:
    result = base.copy()
    states = {position: 0.0 for position in lens.SQUAD_QUOTAS}
    weights = np.zeros(len(data), dtype=float)
    pending: list[tuple[int, int, int, float]] = []
    groups = data.groupby(["season_order", "GW"], sort=True).groups
    audit = []
    for (season_order, gw), group_indices in groups.items():
        if delayed:
            still_pending = []
            for pending_order, pending_end, position, advantage in pending:
                matured = (
                    pending_order < int(season_order)
                    or (
                        pending_order == int(season_order)
                        and pending_end < int(gw)
                    )
                )
                if matured:
                    states[position] = (
                        decay * states[position] + (1 - decay) * advantage
                    )
                else:
                    still_pending.append(
                        (pending_order, pending_end, position, advantage)
                    )
            pending = still_pending
        local_all = np.asarray(group_indices, dtype=int)
        local_weights = {}
        for position, squad_count in lens.SQUAD_QUOTAS.items():
            local = local_all[
                (data.loc[local_all, "position_id"].to_numpy(int) == position)
                & (data.loc[local_all, "fixture_count"].to_numpy(int) > 0)
            ]
            # A 25% challenger prior is updated by realized top-of-frontier value.
            logit_prior = math.log(0.25 / 0.75)
            weight = float(
                np.clip(
                    sigmoid_scalar(logit_prior + states[position] / temperature),
                    0.05,
                    0.50,
                )
            )
            weights[local] = weight
            result[local] = (1 - weight) * base[local] + weight * challenger[local]
            local_weights[position] = round(weight, 3)
            if not len(local):
                continue
            advantage = top_value(
                local, challenger, target, squad_count
            ) - top_value(local, base, target, squad_count)
            if delayed:
                maturity = int(np.max(target_end[local]))
                pending.append((int(season_order), maturity, position, advantage))
            else:
                states[position] = (
                    decay * states[position] + (1 - decay) * advantage
                )
        if int(gw) in {1, 13, 25, 38}:
            audit.append(
                {
                    "season": str(data.loc[local_all[0], "season"]),
                    "gw": int(gw),
                    "weights": {
                        lens.POSITION_LABELS[position]: value
                        for position, value in local_weights.items()
                    },
                }
            )
    result[data["fixture_count"].to_numpy(int) == 0] = 0
    return result, {
        "averageWeight": round(float(np.mean(weights[weights > 0])), 4),
        "snapshots": audit,
    }


def main() -> None:
    data, _ = lens.load_or_build_prepared_history()
    data = data.reset_index(drop=True)
    immediate, horizon, _ = lens.candidate_forecasts(
        data,
        PLAYER_CANDIDATE,
        robust_planning=False,
        schedule_censored=True,
    )
    stable_plan = 0.75 * immediate * 4.5 + 0.25 * horizon
    frontier_raw = np.load(lens.CACHE / "frontier-causal-predictions-v2.npz")[
        "prediction"
    ]
    horizon_raw = np.load(lens.CACHE / "listwise-horizon_target-v1.npz")[
        "prediction"
    ]
    captain_raw = np.load(lens.CACHE / "captain-listwise-v1.npz")["prediction"]
    immediate_challenger = quantile_map(data, frontier_raw, immediate)
    plan_challenger = quantile_map(data, horizon_raw, stable_plan)
    fixed_score = 0.75 * immediate + 0.25 * immediate_challenger
    fixed_plan = 0.75 * stable_plan + 0.25 * plan_challenger
    captain = rank_blend(data, immediate, captain_raw, 0.50)
    configs = [("fixedChampion", fixed_score, fixed_plan, {})]
    for config_name, decay, immediate_temperature, plan_temperature in CONFIGS:
        adaptive_score, immediate_audit = causal_mix(
            data,
            immediate,
            immediate_challenger,
            data["points"].to_numpy(float),
            data["GW"].to_numpy(int),
            decay,
            immediate_temperature,
            delayed=False,
        )
        adaptive_plan, plan_audit = causal_mix(
            data,
            stable_plan,
            plan_challenger,
            data["horizon_target"].to_numpy(float),
            data["horizon_target_end_gw"].to_numpy(int),
            decay,
            plan_temperature,
            delayed=True,
        )
        configs.extend(
            [
                (
                    f"{config_name}-immediate",
                    adaptive_score,
                    fixed_plan,
                    {"immediate": immediate_audit},
                ),
                (
                    f"{config_name}-plan",
                    fixed_score,
                    adaptive_plan,
                    {"plan": plan_audit},
                ),
                (
                    f"{config_name}-both",
                    adaptive_score,
                    adaptive_plan,
                    {"immediate": immediate_audit, "plan": plan_audit},
                ),
            ]
        )
    seasons = list(dict.fromkeys(data["season"].tolist()))
    rows = []
    raw_results = {}
    for name, score, plan, audit in configs:
        print(f"Running {name}", flush=True)
        totals, stats = lens.simulate_candidate(
            data,
            score,
            STRATEGY,
            plan_scores=plan,
            captain_scores=captain,
            tracked_player_name="Salah",
        )
        training = totals[: len(lens.TRAINING_SEASONS)]
        row = {
            "name": name,
            "trainingStability": round(
                float(training.mean() - 0.25 * training.std()), 3
            ),
            "weightAudit": audit,
            "summary": variant_summary(totals, stats, seasons),
        }
        rows.append(row)
        raw_results[name] = (totals, stats, score, plan)
    selected = max(rows, key=lambda row: row["trainingStability"])
    selected_totals, _, selected_score, selected_plan = raw_results[selected["name"]]
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
        "status": "training-selected causal online-expert challenger",
        "method": (
            "Per-position frontier and horizon challenger shares update from "
            "realized top-of-frontier value. Immediate labels update after the "
            "deadline; horizon labels update only after the six-week window matures."
        ),
        "selected": selected,
        "selectedWithAuditedChips": variant_summary(chip_totals, chip_stats, seasons),
        "experiments": rows,
    }
    output = lens.ROOT / "analysis" / "data" / "online_expert_policy_validation.json"
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
