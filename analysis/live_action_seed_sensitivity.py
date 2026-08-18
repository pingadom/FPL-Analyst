"""Seed sensitivity for the live-compatible late action challenger."""

from __future__ import annotations

import json

import numpy as np

import calibrate_model as lens
import live_action_feature_ablation as ablation
from bench_efficiency_validation import variant_summary
from multiscale_horizon_validation import add_targets
from probabilistic_component_challenger import causal_route_predictions
from wildcard_freehit_ablation import champion_forecasts


SEED_VARIANTS = ["liveExtendedRoutes"] + [
    f"liveExtendedRoutesSeed{seed}" for seed in range(2, 6)
]


def main() -> None:
    for name in SEED_VARIANTS[1:]:
        ablation.VARIANTS[name] = (ablation.LIVE_EXTENDED_FEATURES, True)

    original, _ = lens.load_or_build_prepared_history()
    data = add_targets(original.reset_index(drop=True))
    immediate, champion_plan, captain = champion_forecasts(data)
    component, _ = causal_route_predictions(data, immediate)
    seasons = list(dict.fromkeys(data["season"].tolist()))
    base_totals, base_stats = lens.simulate_candidate(
        data,
        immediate,
        ablation.STRATEGY,
        plan_scores=champion_plan,
        captain_scores=captain,
        tracked_player_name="Salah",
    )
    baseline = variant_summary(base_totals, base_stats, seasons)
    rows = []
    for name in SEED_VARIANTS:
        shifts = ablation.causal_predictions(
            name,
            data,
            ablation.LIVE_EXTENDED_FEATURES,
            True,
            champion_plan,
            component,
        )
        summary, totals = ablation.recursive_summary(
            data, immediate, champion_plan, captain, shifts, seasons
        )
        evaluation_delta = np.asarray(totals[2:], dtype=float) - base_totals[2:]
        rows.append(
            {
                "name": name,
                "summary": summary,
                "averageDelta": round(float(evaluation_delta.mean()), 1),
                "minimumDelta": int(summary["minimum"] - baseline["minimum"]),
                "positiveSeasons": int((evaluation_delta > 0).sum()),
                "negativeSeasons": int((evaluation_delta < 0).sum()),
                "pairedDeltas": evaluation_delta.astype(int).tolist(),
                "developmentAverageDelta": round(float(evaluation_delta[:4].mean()), 1),
                "holdoutAverageDelta": round(float(evaluation_delta[4:].mean()), 1),
            }
        )
        print(
            json.dumps(
                {
                    "name": name,
                    "average": summary["average"],
                    "minimum": summary["minimum"],
                    "deltas": rows[-1]["pairedDeltas"],
                }
            ),
            flush=True,
        )

    gains = np.asarray([row["averageDelta"] for row in rows], dtype=float)
    holdout = np.asarray([row["holdoutAverageDelta"] for row in rows], dtype=float)
    minimum = np.asarray([row["minimumDelta"] for row in rows], dtype=float)
    robust = bool(
        np.median(gains) > 0
        and np.quantile(gains, 0.20) >= 0
        and np.median(holdout) >= 0
        and np.quantile(minimum, 0.20) >= 0
    )
    result = {
        "status": "prospective shadow seed gate passed" if robust else "research-only; seed gate failed",
        "method": "Five independently seeded causal near-price LambdaMART fits using the live-extended context and learned scoring routes.",
        "baseline": baseline,
        "seeds": rows,
        "aggregate": {
            "medianAverageDelta": round(float(np.median(gains)), 1),
            "p20AverageDelta": round(float(np.quantile(gains, 0.20)), 1),
            "medianHoldoutDelta": round(float(np.median(holdout)), 1),
            "medianMinimumDelta": round(float(np.median(minimum)), 1),
            "positiveAverageSeeds": int((gains > 0).sum()),
            "seedCount": len(rows),
        },
        "prospectiveShadow": robust,
        "productionPromotion": False,
    }
    output = lens.ROOT / "analysis" / "data" / "live_action_seed_sensitivity.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2))
    print(f"Wrote {output.relative_to(lens.ROOT)}")


if __name__ == "__main__":
    main()
