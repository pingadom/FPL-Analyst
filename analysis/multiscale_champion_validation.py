"""Stress-test the promising phase-gated multi-timescale policy and chips."""

from __future__ import annotations

import json

import numpy as np

import calibrate_model as lens
from bench_efficiency_validation import championship_forecasts
from frontier_ranker_validation import STRATEGY
from hybrid_decision_validation import summarize
from listwise_ranker_validation import quantile_map
from multiscale_horizon_validation import (
    adaptive_value,
    add_targets,
    causal_online_ridge_horizons,
    causal_ridge_horizons,
    structural_horizons,
)
from multiscale_phase_validation import event_number, phased_plan


def main() -> None:
    original, _ = lens.load_or_build_prepared_history()
    data = add_targets(original.reset_index(drop=True))
    scores, baseline_plan, captain = championship_forecasts(data)
    structural = structural_horizons(data, scores)
    learned, _ = causal_ridge_horizons(data, structural)
    online, online_audit = causal_online_ridge_horizons(data, structural, learned)
    online_value = quantile_map(
        data, adaptive_value(data, online, 3.0), baseline_plan
    )
    events = event_number(data)
    plans = {
        "baseline": baseline_plan,
        "online10AfterGW13": phased_plan(
            baseline_plan, online_value, events, 13, 0.10
        ),
        "online15AfterGW13": phased_plan(
            baseline_plan, online_value, events, 13, 0.15
        ),
        "online20AfterGW13": phased_plan(
            baseline_plan, online_value, events, 13, 0.20
        ),
    }
    seasons = list(dict.fromkeys(data["season"].tolist()))
    benchmark = json.loads(
        (lens.ROOT / "analysis" / "data" / "historical_rank_benchmarks.json").read_text(
            encoding="utf-8"
        )
    )
    targets = {
        row["season"].replace("/", "-"): int(row["points"])
        for row in benchmark["seasons"]
    }
    rows: dict[str, dict] = {}
    raw: dict[str, tuple[np.ndarray, list[dict]]] = {}
    for name, plan in plans.items():
        print(f"Running {name}", flush=True)
        totals, stats = lens.simulate_candidate(
            data,
            scores,
            STRATEGY,
            plan_scores=plan,
            captain_scores=captain,
            tracked_player_name="Salah",
        )
        summary = summarize(totals, seasons, targets, stats)
        training = totals[: len(lens.TRAINING_SEASONS)]
        rows[name] = {
            **summary,
            "trainingStability": round(
                float(training.mean() - 0.25 * training.std()), 3
            ),
        }
        raw[name] = (totals, stats)

    plan = plans["online10AfterGW13"]
    print("Running online10AfterGW13 with audited chips", flush=True)
    chip_totals, chip_stats = lens.simulate_candidate(
        data,
        scores,
        STRATEGY,
        chip_policy=lens.AUDITED_CHAMPION_CHIP_POLICY,
        plan_scores=plan,
        captain_scores=captain,
        tracked_player_name="Salah",
    )
    rows["online10AfterGW13AuditedChips"] = {
        **summarize(chip_totals, seasons, targets, chip_stats),
        "trainingStability": None,
    }

    baseline_totals, _ = raw["baseline"]
    challenger_totals, _ = raw["online10AfterGW13"]
    evaluation_indices = [
        index for index, season in enumerate(seasons) if season in lens.EVALUATION_SEASONS
    ]
    paired_deltas = [
        int(round(challenger_totals[index] - baseline_totals[index]))
        for index in evaluation_indices
    ]
    result = {
        "status": "historical research champion; prospective validation required",
        "method": (
            "The established immediate/six-week stack remains in control through "
            "GW12. From GW13, 10% of planning rank comes from causal 1/3/6/10-GW "
            "value functions with player-specific tenure and exit cost. Models "
            "refresh at GW13 and GW25 using only fully matured labels."
        ),
        "selectionWarning": (
            "The phase and blend have now been inspected on the evaluation seasons. "
            "Treat this as the frozen historical champion, not an independent "
            "estimate of future performance."
        ),
        "pairedNoChipDeltas": paired_deltas,
        "improvedSeasons": int(sum(delta > 0 for delta in paired_deltas)),
        "unchangedSeasons": int(sum(delta == 0 for delta in paired_deltas)),
        "worseSeasons": int(sum(delta < 0 for delta in paired_deltas)),
        "onlineFitAudit": online_audit,
        "auditedChipPolicy": lens.AUDITED_CHAMPION_CHIP_POLICY.as_dict(),
        "variants": rows,
    }
    output = lens.ROOT / "analysis" / "data" / "multiscale_champion_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "pairedNoChipDeltas": paired_deltas,
                "variants": {
                    name: {
                        "average": row["average"],
                        "minimum": row["minimum"],
                        "targetHits": row["targetHits"],
                        "averageMargin": row["averageMargin"],
                        "trainingStability": row["trainingStability"],
                    }
                    for name, row in rows.items()
                },
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
