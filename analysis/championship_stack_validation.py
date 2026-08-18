"""Evaluate the strongest orthogonal research stack without promoting it."""

from __future__ import annotations

import json

import numpy as np

import calibrate_model as lens
from captain_ranker_validation import rank_blend
from frontier_ranker_validation import CHIP_POLICY, PLAYER_CANDIDATE, STRATEGY
from hybrid_decision_validation import summarize
from listwise_ranker_validation import quantile_map


def main() -> None:
    data, _ = lens.load_or_build_prepared_history()
    data = data.reset_index(drop=True)
    immediate, horizon, _ = lens.candidate_forecasts(
        data, PLAYER_CANDIDATE, robust_planning=False, schedule_censored=True
    )
    stable_plan = 0.75 * immediate * 4.5 + 0.25 * horizon
    frontier_raw = np.load(lens.CACHE / "frontier-causal-predictions-v2.npz")["prediction"]
    horizon_raw = np.load(lens.CACHE / "listwise-horizon_target-v1.npz")["prediction"]
    captain_raw = np.load(lens.CACHE / "captain-listwise-v1.npz")["prediction"]
    immediate_mapped = quantile_map(data, frontier_raw, immediate)
    plan_mapped = quantile_map(data, horizon_raw, stable_plan)
    score = 0.75 * immediate + 0.25 * immediate_mapped
    plan = 0.75 * stable_plan + 0.25 * plan_mapped
    captain_score = rank_blend(data, immediate, captain_raw, 0.50)
    seasons = list(dict.fromkeys(data["season"].tolist()))
    benchmark = json.loads(
        (lens.ROOT / "analysis" / "data" / "historical_rank_benchmarks.json").read_text(encoding="utf-8")
    )
    targets = {row["season"].replace("/", "-"): int(row["points"]) for row in benchmark["seasons"]}
    variants = {}
    totals, stats = lens.simulate_candidate(data, score, STRATEGY, plan_scores=plan)
    variants["hybrid"] = summarize(totals, seasons, targets, stats)
    captain_totals, captain_stats = lens.simulate_candidate(
        data, score, STRATEGY, plan_scores=plan, captain_scores=captain_score
    )
    variants["hybridCaptain50"] = summarize(captain_totals, seasons, targets, captain_stats)
    fresh = lens.precompute_fresh_squads(data, plan)
    free_hits = lens.precompute_fresh_squads(data, score)
    chip_totals, chip_stats = lens.simulate_candidate(
        data,
        score,
        STRATEGY,
        chip_policy=CHIP_POLICY,
        fresh_squads=fresh,
        free_hit_squads=free_hits,
        plan_scores=plan,
        captain_scores=captain_score,
    )
    variants["hybridCaptain50LegacyChips"] = summarize(
        chip_totals, seasons, targets, chip_stats
    )
    audited_chip_totals, audited_chip_stats = lens.simulate_candidate(
        data,
        score,
        STRATEGY,
        chip_policy=lens.AUDITED_CHAMPION_CHIP_POLICY,
        plan_scores=plan,
        captain_scores=captain_score,
    )
    variants["hybridCaptain50AuditedChips"] = summarize(
        audited_chip_totals, seasons, targets, audited_chip_stats
    )
    result = {
        "status": "historical champion; requires prospective shadow validation",
        "method": "25% causal frontier immediate rerank + 25% causal listwise six-week rerank + 50% causal captain rerank. Legacy chips remain as a diagnostic; the audited row enables only the walk-forward validated Bench Boost and Triple Captain gates.",
        "promotionRule": "Use the audited row as the historical research champion. Do not claim a rank guarantee or enable automatic Wildcard/Free Hit until frozen prospective evidence clears the promotion gate.",
        "auditedChipPolicy": lens.AUDITED_CHAMPION_CHIP_POLICY.as_dict(),
        "variants": variants,
    }
    output = lens.ROOT / "analysis" / "data" / "championship_stack_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: {key: value for key, value in row.items() if key != "evaluation"} for name, row in variants.items()}, indent=2))


if __name__ == "__main__":
    main()
