"""Walk-forward validation of liquidity-aware premium transfer packages.

The configurations are predeclared and selected only on 2016/17-2017/18.  The
evaluation seasons are reported once and cannot choose the winning policy.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

import calibrate_model as lens
from bench_efficiency_validation import championship_forecasts, variant_summary
from frontier_ranker_validation import STRATEGY


CONFIGS = (
    {
        "name": "same-week-packages-only",
        "package_deferred_routes": False,
        "package_route_discount": 0.55,
        "package_liquidity_states": 4,
        "package_setup_loss_limit": 0.0,
        "package_setup_hurdle": 99.0,
        "package_future_hurdle_scale": 1.0,
        "package_target_limit": 6,
    },
    {
        "name": "route-conservative",
        "package_route_discount": 0.35,
        "package_liquidity_states": 3,
        "package_setup_loss_limit": 1.5,
        "package_setup_hurdle": 2.0,
        "package_future_hurdle_scale": 0.65,
        "package_target_limit": 5,
    },
    {
        "name": "route-balanced",
        "package_route_discount": 0.55,
        "package_liquidity_states": 4,
        "package_setup_loss_limit": 3.0,
        "package_setup_hurdle": 1.5,
        "package_future_hurdle_scale": 0.50,
        "package_target_limit": 6,
    },
    {
        "name": "route-aggressive",
        "package_route_discount": 0.75,
        "package_liquidity_states": 6,
        "package_setup_loss_limit": 4.0,
        "package_setup_hurdle": 0.5,
        "package_future_hurdle_scale": 0.35,
        "package_target_limit": 8,
    },
)


def training_stability(totals: np.ndarray) -> float:
    training = totals[: len(lens.TRAINING_SEASONS)]
    return float(training.mean() - 0.25 * training.std())


def main() -> None:
    data, _ = lens.load_or_build_prepared_history()
    data = data.reset_index(drop=True)
    scores, plan_scores, captain_scores = championship_forecasts(data)
    seasons = list(dict.fromkeys(data["season"].tolist()))

    print("Running frozen baseline", flush=True)
    base_totals, base_stats = lens.simulate_candidate(
        data,
        scores,
        STRATEGY,
        plan_scores=plan_scores,
        captain_scores=captain_scores,
        tracked_player_name="Salah",
    )
    baseline = variant_summary(base_totals, base_stats, seasons)
    experiments = []
    raw_results: dict[str, tuple[np.ndarray, list[dict], lens.SimulationStrategy]] = {}
    for index, config in enumerate(CONFIGS, start=1):
        strategy = replace(
            STRATEGY,
            package_route_search=True,
            **config,
        )
        print(f"Running {index}/{len(CONFIGS)}: {strategy.name}", flush=True)
        totals, stats = lens.simulate_candidate(
            data,
            scores,
            strategy,
            plan_scores=plan_scores,
            captain_scores=captain_scores,
            tracked_player_name="Salah",
        )
        summary = variant_summary(totals, stats, seasons)
        row = {
            "name": strategy.name,
            "parameters": {
                key: value for key, value in config.items() if key != "name"
            },
            "trainingStability": round(training_stability(totals), 3),
            "summary": summary,
        }
        experiments.append(row)
        raw_results[strategy.name] = (totals, stats, strategy)

    selected = max(experiments, key=lambda row: row["trainingStability"])
    selected_totals, _, selected_strategy = raw_results[selected["name"]]
    print(f"Running selected policy with audited TC/BB: {selected['name']}", flush=True)
    chip_totals, chip_stats = lens.simulate_candidate(
        data,
        scores,
        selected_strategy,
        chip_policy=lens.AUDITED_CHAMPION_CHIP_POLICY,
        plan_scores=plan_scores,
        captain_scores=captain_scores,
        tracked_player_name="Salah",
    )
    result = {
        "status": "training-selected causal package-search challenger",
        "method": (
            "Preserve liquidity states inside the transfer beam and value only a "
            "single legal next-transfer option using the current deadline's "
            "schedule-censored horizon. No future result or final schedule is read."
        ),
        "selectionRule": (
            "Maximise mean minus 0.25 standard deviation on 2016/17-2017/18 only."
        ),
        "baseline": baseline,
        "selected": selected,
        "selectedNoChipLift": {
            "average": round(float(selected_totals[2:].mean() - base_totals[2:].mean()), 1),
            "minimum": int(round(float(selected_totals[2:].min() - base_totals[2:].min()))),
        },
        "selectedWithAuditedChips": variant_summary(chip_totals, chip_stats, seasons),
        "experiments": experiments,
    }
    output = lens.ROOT / "analysis" / "data" / "premium_route_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "baseline": baseline["average"],
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
                    for row in experiments
                ],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
