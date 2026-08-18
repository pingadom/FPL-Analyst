"""Full recursive and chip validation of the captain fixture finalists."""

from __future__ import annotations

import json

import numpy as np

import calibrate_model as lens
from captain_fixture_history_validation import (
    add_fixture_history,
    causal_predictions,
    percentile,
)
from frontier_ranker_validation import STRATEGY
from late_action_captain_chip_validation import evaluate
from live_action_ensemble_validation import mapped_seed_predictions, policy
from multiscale_horizon_validation import add_targets
from multiscale_phase_validation import event_number
from premium_captain_validation import captain_variants
from probabilistic_minutes_validation import season_summary
from wildcard_freehit_ablation import champion_forecasts


def captain_scores(data, immediate: np.ndarray, captain: np.ndarray) -> dict[str, np.ndarray]:
    components = captain_variants(data, immediate, captain)
    frozen = components["frozen"]
    ceiling = (components["ceiling25"] - 0.75 * frozen) / 0.25
    context = causal_predictions(data, immediate, False)
    history = causal_predictions(data, immediate, True)
    context_ranks = np.column_stack(
        [percentile(data, context[:, seed]) for seed in range(context.shape[1])]
    )
    history_ranks = np.column_stack(
        [percentile(data, history[:, seed]) for seed in range(history.shape[1])]
    )
    context_mean = context_ranks.mean(axis=1)
    history_mean = history_ranks.mean(axis=1)
    history_median = np.median(history_ranks, axis=1)
    return {
        "frozen": captain,
        "ceiling15": 0.85 * frozen + 0.15 * ceiling,
        "context20Mean": 0.80 * frozen + 0.20 * context_mean,
        "history20Median": 0.80 * frozen + 0.20 * history_median,
        "crossModel20Mean": 0.80 * frozen
        + 0.20 * (0.50 * context_mean + 0.50 * history_mean),
        "crossModel20Conservative": 0.80 * frozen
        + 0.20 * np.minimum(context_mean, history_mean),
    }


def summary_with_delta(
    totals: np.ndarray,
    baseline: np.ndarray,
    seasons: list[str],
) -> dict:
    summary = season_summary(totals, seasons)
    delta = totals[2:] - baseline[2:]
    return {
        **summary,
        "averageDelta": round(float(delta.mean()), 1),
        "developmentDelta": round(float(delta[:-2].mean()), 1),
        "holdoutDelta": round(float(delta[-2:].mean()), 1),
        "minimumDelta": int(summary["minimum"] - int(round(float(baseline[2:].min())))),
        "worstSeasonDelta": int(delta.min()),
        "improvedSeasons": int((delta > 0).sum()),
        "declinedSeasons": int((delta < 0).sum()),
        "seasonDeltas": delta.astype(int).tolist(),
    }


def main() -> None:
    original, _ = lens.load_or_build_prepared_history()
    data = add_fixture_history(add_targets(original.reset_index(drop=True)))
    immediate, plan, captain = champion_forecasts(data)
    captains = captain_scores(data, immediate, captain)
    seasons = list(dict.fromkeys(data["season"].tolist()))

    no_chip_totals = {}
    no_chip_models = {}
    for name, captain_score in captains.items():
        print(f"Full captain replay, no chips: {name}", flush=True)
        totals, _ = lens.simulate_candidate(
            data,
            immediate,
            STRATEGY,
            plan_scores=plan,
            captain_scores=captain_score,
        )
        no_chip_totals[name] = totals
    baseline = no_chip_totals["frozen"]
    for name, totals in no_chip_totals.items():
        no_chip_models[name] = summary_with_delta(totals, baseline, seasons)
        print(name, no_chip_models[name]["average"], no_chip_models[name]["seasonDeltas"], flush=True)

    stable = [
        name
        for name, row in no_chip_models.items()
        if name not in {"frozen", "ceiling15"}
        and row["developmentDelta"] > 0
        and row["holdoutDelta"] >= 0
        and row["worstSeasonDelta"] >= -8
        and row["declinedSeasons"] <= 2
    ]
    selected = max(
        stable,
        key=lambda name: (
            no_chip_models[name]["averageDelta"]
            - 0.20 * abs(no_chip_models[name]["worstSeasonDelta"]),
            no_chip_models[name]["minimum"],
        ),
        default=None,
    )

    mapped = mapped_seed_predictions(data, plan)
    raw_action, consensus, _ = policy(mapped, plan, "vote80", 0.05)
    action_active = (
        (event_number(data) >= 25)
        & (data["position_id"].to_numpy(int) != 1)
        & consensus
    )
    action_plan = np.where(action_active, raw_action, plan)
    benchmark = json.loads(
        (lens.ROOT / "analysis" / "data" / "historical_rank_benchmarks.json").read_text(
            encoding="utf-8"
        )
    )
    targets = {
        row["season"].replace("/", "-"): int(row["points"])
        for row in benchmark["seasons"]
    }
    chip_names = ["frozen", "ceiling15"] + stable
    chip_models = {}
    chip_totals = {}
    for name in chip_names:
        print(f"Full captain replay, live action + audited chips: {name}", flush=True)
        totals, _ = lens.simulate_candidate(
            data,
            immediate,
            STRATEGY,
            chip_policy=lens.AUDITED_CHAMPION_CHIP_POLICY,
            plan_scores=action_plan,
            captain_scores=captains[name],
        )
        chip_totals[name] = totals
        chip_models[name] = evaluate(totals, seasons, targets)
    chip_baseline = chip_totals["frozen"]
    chip_comparisons = {
        name: summary_with_delta(totals, chip_baseline, seasons)
        for name, totals in chip_totals.items()
    }
    chip_selected = max(
        stable,
        key=lambda name: (
            chip_comparisons[name]["averageDelta"]
            - 0.20 * abs(chip_comparisons[name]["worstSeasonDelta"]),
            chip_comparisons[name]["minimum"],
        ),
        default=None,
    )
    result = {
        "status": "prospective captain shadow finalist" if chip_selected else "research-only; full recursive gate failed",
        "method": "Full recursive replay of fixed captain ranks; finalists are then paired with the exact live-compatible late-action plan and unchanged audited chip policy.",
        "noChipModels": no_chip_models,
        "noChipStable": stable,
        "noChipSelected": selected,
        "actionChipModels": chip_models,
        "actionChipComparisons": chip_comparisons,
        "selected": chip_selected,
        "productionPromotion": False,
    }
    output = lens.ROOT / "analysis" / "data" / "captain_fixture_recursive_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "noChipSelected": selected,
                "chipSelected": chip_selected,
                "noChip": {
                    name: {
                        "average": row["average"],
                        "minimum": row["minimum"],
                        "averageDelta": row["averageDelta"],
                    }
                    for name, row in no_chip_models.items()
                },
                "actionChips": {
                    name: {
                        "average": chip_models[name]["average"],
                        "minimum": chip_models[name]["minimum"],
                        "averageDelta": chip_comparisons[name]["averageDelta"],
                        "worstSeasonDelta": chip_comparisons[name]["worstSeasonDelta"],
                    }
                    for name in chip_models
                },
            },
            indent=2,
        )
    )
    print(f"Wrote {output.relative_to(lens.ROOT)}")


if __name__ == "__main__":
    main()
