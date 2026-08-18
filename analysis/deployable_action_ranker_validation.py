"""Validate a near-price action ranker whose features are all live-deployable."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from xgboost import XGBRanker

import calibrate_model as lens
from bench_efficiency_validation import variant_summary
from decision_focused_horizon_validation import CHECKPOINTS, target_and_maturity
from frontier_ranker_validation import FEATURES, STRATEGY, matrix, selectable_frontier
from listwise_ranker_validation import quantile_map
from multiscale_horizon_validation import add_targets
from multiscale_phase_validation import event_number
from transfer_action_ranker_validation import PRICE_BAND, SHIFTS, query_order
from wildcard_freehit_ablation import champion_forecasts


CACHE_VERSION = 1


def action_model(seed: int) -> XGBRanker:
    return XGBRanker(
        n_estimators=145,
        max_depth=3,
        learning_rate=0.04,
        min_child_weight=16,
        subsample=0.84,
        colsample_bytree=0.82,
        reg_alpha=0.18,
        reg_lambda=3.2,
        objective="rank:pairwise",
        eval_metric="ndcg@10",
        lambdarank_pair_method="mean",
        lambdarank_num_pair_per_sample=12,
        n_jobs=-1,
        random_state=seed,
    )


def causal_predictions(data: pd.DataFrame, champion_plan: np.ndarray) -> np.ndarray:
    path = lens.CACHE / f"deployable-action-ranker-v{CACHE_VERSION}.npz"
    if path.exists():
        cached = np.load(path)
        if len(cached["shifts"]) == len(data):
            return cached["shifts"]
    target, maturity = target_and_maturity(data)
    output = np.tile(champion_plan[:, None], (1, len(SHIFTS)))
    seasons = list(dict.fromkeys(data["season"].tolist()))
    orders = data["season_order"].to_numpy(int)
    gws = data["GW"].to_numpy(int)
    positions = data["position_id"].to_numpy(int)
    observed = data["fixture_count"].to_numpy(int) > 0
    frontier = selectable_frontier(data)
    for season_order in range(1, len(seasons)):
        season_mask = orders == season_order
        for checkpoint_index, checkpoint in enumerate(CHECKPOINTS):
            interval_end = CHECKPOINTS[checkpoint_index + 1] - 1 if checkpoint_index + 1 < len(CHECKPOINTS) else 99
            for position in lens.SQUAD_QUOTAS:
                position_mask = positions == position
                train_mask = (
                    position_mask
                    & observed
                    & frontier
                    & ((orders < season_order) | (season_mask & (maturity < checkpoint)))
                )
                test_mask = season_mask & position_mask & (gws >= checkpoint) & (gws <= interval_end)
                if not test_mask.any():
                    continue
                train = data.loc[train_mask]
                test = data.loc[test_mask]
                train_x, medians = matrix(train)
                test_x, _ = matrix(test, medians)
                for shift_index, shift in enumerate(SHIFTS):
                    order, qid = query_order(train, shift)
                    fitted = action_model(710000 + 1000 * season_order + 100 * checkpoint_index + 10 * position + shift_index)
                    fitted.fit(train_x[order], target[train_mask][order], qid=qid)
                    output[test_mask, shift_index] = fitted.predict(test_x)
            print(f"Deployable action ranker {seasons[season_order]} GW{checkpoint}", flush=True)
    np.savez_compressed(path, shifts=output, features=json.dumps(FEATURES))
    return output


def mapped_consensus(data: pd.DataFrame, champion_plan: np.ndarray, raw_shifts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mapped = np.column_stack(
        [quantile_map(data, raw_shifts[:, index], champion_plan) for index in range(len(SHIFTS))]
    )
    delta = mapped - champion_plan[:, None]
    agreement = np.sign(delta[:, 0]) == np.sign(delta[:, 1])
    return mapped.mean(axis=1), agreement


def map_live(raw: np.ndarray, reference: np.ndarray, positions: np.ndarray) -> np.ndarray:
    mapped = np.zeros_like(raw, dtype=float)
    for position in lens.SQUAD_QUOTAS:
        indices = np.flatnonzero(positions == position)
        order = indices[np.argsort(raw[indices], kind="stable")]
        mapped[order] = np.sort(reference[indices])
    return mapped


def terminal_live_action_scores(
    data: pd.DataFrame,
    live: pd.DataFrame,
    reference_horizon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the validated reduced ranker through the last completed season."""
    target, _ = target_and_maturity(data)
    frontier = selectable_frontier(data)
    observed = data["fixture_count"].to_numpy(int) > 0
    positions = data["position_id"].to_numpy(int)
    live_positions = live["position_id"].to_numpy(int)
    mapped_shifts = np.tile(reference_horizon[:, None], (1, len(SHIFTS)))
    for position in lens.SQUAD_QUOTAS:
        train_mask = observed & frontier & (positions == position)
        test_mask = live_positions == position
        train = data.loc[train_mask]
        train_x, medians = matrix(train)
        test_x, _ = matrix(live.loc[test_mask], medians)
        for shift_index, shift in enumerate(SHIFTS):
            order, qid = query_order(train, shift)
            fitted = action_model(810000 + 10 * position + shift_index)
            fitted.fit(train_x[order], target[train_mask][order], qid=qid)
            raw = fitted.predict(test_x)
            full_raw = np.zeros(len(live))
            full_raw[test_mask] = raw
            mapped = map_live(full_raw, reference_horizon, live_positions)
            mapped_shifts[test_mask, shift_index] = mapped[test_mask]
    delta = mapped_shifts - reference_horizon[:, None]
    agreement = np.sign(delta[:, 0]) == np.sign(delta[:, 1])
    consensus = mapped_shifts.mean(axis=1)
    score = reference_horizon + 0.05 * agreement * (consensus - reference_horizon)
    return score, agreement


def main() -> None:
    original, _ = lens.load_or_build_prepared_history()
    data = add_targets(original.reset_index(drop=True))
    immediate, champion_plan, captain = champion_forecasts(data)
    raw = causal_predictions(data, champion_plan)
    consensus, agreement = mapped_consensus(data, champion_plan, raw)
    events = event_number(data)
    position = data["position_id"].to_numpy(int)
    active = (events >= 25) & agreement & (position != 1)
    plan = champion_plan + 0.05 * active.astype(float) * (consensus - champion_plan)
    seasons = list(dict.fromkeys(data["season"].tolist()))
    base_totals, base_stats = lens.simulate_candidate(
        data, immediate, STRATEGY, plan_scores=champion_plan, captain_scores=captain,
        tracked_player_name="Salah",
    )
    totals, stats = lens.simulate_candidate(
        data, immediate, STRATEGY, plan_scores=plan, captain_scores=captain,
        tracked_player_name="Salah",
    )
    baseline = variant_summary(base_totals, base_stats, seasons)
    challenger = variant_summary(totals, stats, seasons)
    paired = [
        {"season": old["season"], "before": old["points"], "after": new["points"], "delta": new["points"] - old["points"]}
        for old, new in zip(baseline["seasons"], challenger["seasons"])
    ]
    development = totals[2:8]
    base_development = base_totals[2:8]
    holdout = totals[8:]
    base_holdout = base_totals[8:]
    shadow = bool(
        development.mean() - 0.25 * development.std() > base_development.mean() - 0.25 * base_development.std()
        and holdout.mean() >= base_holdout.mean()
        and challenger["minimum"] >= baseline["minimum"]
    )
    result = {
        "status": "prospective shadow candidate" if shadow else "research-only; deployable feature gate failed",
        "method": "Two overlapping near-price LambdaMART ranks using only the 18 features available in both historical and live deadline frames; active GW25+, non-GK, 5% when both bands agree.",
        "features": FEATURES,
        "baseline": baseline,
        "challenger": challenger,
        "paired": paired,
        "developmentStability": round(float(development.mean() - 0.25 * development.std()), 3),
        "baselineDevelopmentStability": round(float(base_development.mean() - 0.25 * base_development.std()), 3),
        "holdoutAverage": round(float(holdout.mean()), 1),
        "baselineHoldoutAverage": round(float(base_holdout.mean()), 1),
        "prospectiveShadow": shadow,
        "productionPromotion": False,
    }
    output = lens.ROOT / "analysis" / "data" / "deployable_action_ranker_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "baseline": baseline["average"],
        "challenger": challenger["average"],
        "minimum": challenger["minimum"],
        "holdout": result["holdoutAverage"],
        "paired": paired,
        "prospectiveShadow": shadow,
    }, indent=2))


if __name__ == "__main__":
    main()
