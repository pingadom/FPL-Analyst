"""Stabilise the live-compatible action ranker with cross-seed consensus."""

from __future__ import annotations

import json

import numpy as np

import calibrate_model as lens
import live_action_feature_ablation as ablation
from bench_efficiency_validation import variant_summary
from frontier_ranker_validation import STRATEGY
from listwise_ranker_validation import quantile_map
from multiscale_horizon_validation import add_targets
from multiscale_phase_validation import event_number
from transfer_action_ranker_validation import SHIFTS
from wildcard_freehit_ablation import champion_forecasts


SEED_NAMES = ["liveExtendedRoutes"] + [
    f"liveExtendedRoutesSeed{seed}" for seed in range(2, 6)
]


def mapped_seed_predictions(data, champion_plan: np.ndarray) -> np.ndarray:
    mapped = []
    for name in SEED_NAMES:
        cached = np.load(
            lens.CACHE
            / f"live-action-ablation-{name}-v{ablation.CACHE_VERSION}.npz"
        )
        shifts = cached["shifts"]
        mapped.append(
            np.column_stack(
                [
                    quantile_map(data, shifts[:, index], champion_plan)
                    for index in range(len(SHIFTS))
                ]
            )
        )
    return np.stack(mapped, axis=2)  # rows x price bands x seeds


def policy(
    mapped: np.ndarray,
    champion_plan: np.ndarray,
    method: str,
    share: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    deltas = mapped - champion_plan[:, None, None]
    signs = np.sign(deltas)
    if method == "meanBands":
        band_delta = deltas.mean(axis=2)
        active = np.sign(band_delta[:, 0]) == np.sign(band_delta[:, 1])
        consensus = mapped.mean(axis=(1, 2))
        vote = np.abs(signs.mean(axis=(1, 2)))
    elif method == "medianBands":
        band_delta = np.median(deltas, axis=2)
        active = np.sign(band_delta[:, 0]) == np.sign(band_delta[:, 1])
        consensus = np.median(mapped, axis=(1, 2))
        vote = np.abs(signs.mean(axis=(1, 2)))
    elif method in {"vote80", "unanimous"}:
        within_seed = np.sign(deltas[:, 0, :]) == np.sign(deltas[:, 1, :])
        seed_delta = deltas.mean(axis=1)
        seed_sign = np.sign(seed_delta)
        positive = ((seed_sign > 0) & within_seed).mean(axis=1)
        negative = ((seed_sign < 0) & within_seed).mean(axis=1)
        threshold = 0.80 if method == "vote80" else 1.0
        active = (positive >= threshold) | (negative >= threshold)
        consensus = np.median(mapped.mean(axis=1), axis=1)
        vote = np.maximum(positive, negative)
    else:
        raise ValueError(method)
    return champion_plan + share * active * (consensus - champion_plan), active, vote


def main() -> None:
    original, _ = lens.load_or_build_prepared_history()
    data = add_targets(original.reset_index(drop=True))
    immediate, champion_plan, captain = champion_forecasts(data)
    seasons = list(dict.fromkeys(data["season"].tolist()))
    base_totals, base_stats = lens.simulate_candidate(
        data,
        immediate,
        STRATEGY,
        plan_scores=champion_plan,
        captain_scores=captain,
        tracked_player_name="Salah",
    )
    baseline = variant_summary(base_totals, base_stats, seasons)
    mapped = mapped_seed_predictions(data, champion_plan)
    eligible = (event_number(data) >= 25) & (
        data["position_id"].to_numpy(int) != 1
    )
    rows = []
    for method in ("meanBands", "medianBands", "vote80", "unanimous"):
        for share in (0.025, 0.05, 0.075):
            raw_plan, consensus, vote = policy(mapped, champion_plan, method, share)
            active = eligible & consensus
            plan = np.where(active, raw_plan, champion_plan)
            totals, stats = lens.simulate_candidate(
                data,
                immediate,
                STRATEGY,
                plan_scores=plan,
                captain_scores=captain,
                tracked_player_name="Salah",
            )
            summary = variant_summary(totals, stats, seasons)
            evaluation_delta = totals[2:] - base_totals[2:]
            rows.append(
                {
                    "name": f"{method}{int(share * 1000):03d}",
                    "method": method,
                    "share": share,
                    "summary": summary,
                    "averageDelta": round(float(evaluation_delta.mean()), 1),
                    "minimumDelta": int(summary["minimum"] - baseline["minimum"]),
                    "pairedDeltas": evaluation_delta.astype(int).tolist(),
                    "developmentAverageDelta": round(float(evaluation_delta[:4].mean()), 1),
                    "holdoutAverageDelta": round(float(evaluation_delta[4:].mean()), 1),
                    "activeRows": int(active.sum()),
                    "meanVoteWhenActive": round(float(vote[active].mean()), 3)
                    if active.any()
                    else 0.0,
                }
            )
            print(
                json.dumps(
                    {
                        "name": rows[-1]["name"],
                        "average": summary["average"],
                        "minimum": summary["minimum"],
                        "deltas": rows[-1]["pairedDeltas"],
                    }
                ),
                flush=True,
            )

    stable = [
        row
        for row in rows
        if row["averageDelta"] > 0
        and row["developmentAverageDelta"] >= 0
        and row["holdoutAverageDelta"] >= 0
        and row["minimumDelta"] >= 0
    ]
    selected = max(
        stable,
        key=lambda row: (
            row["averageDelta"] - 0.25 * abs(row["developmentAverageDelta"] - row["holdoutAverageDelta"]),
            row["minimumDelta"],
        ),
        default=None,
    )
    result = {
        "status": "prospective shadow candidate" if selected else "research-only; consensus gate failed",
        "method": "Five-seed rank consensus over two overlapping near-price bands; GW25+, non-GK only.",
        "baseline": baseline,
        "variants": rows,
        "selected": selected["name"] if selected else None,
        "selectionRule": "Positive development and holdout average deltas, no lower historical minimum, then maximise stability-adjusted average gain.",
        "prospectiveShadow": selected is not None,
        "productionPromotion": False,
    }
    output = lens.ROOT / "analysis" / "data" / "live_action_ensemble_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected": result["selected"], "stableCount": len(stable)}, indent=2))
    print(f"Wrote {output.relative_to(lens.ROOT)}")


if __name__ == "__main__":
    main()
