"""Directly learn player-specific holding value on the selectable frontier."""

from __future__ import annotations

import json

import numpy as np
from xgboost import XGBRegressor

import calibrate_model as lens
from bench_efficiency_validation import championship_forecasts, variant_summary
from frontier_ranker_validation import STRATEGY, selectable_frontier
from listwise_ranker_validation import quantile_map
from multiscale_horizon_validation import (
    FEATURES,
    adaptive_value,
    add_targets,
    causal_online_ridge_horizons,
    causal_ridge_horizons,
    expected_tenure,
    feature_matrix,
    structural_horizons,
)
from multiscale_phase_validation import event_number


CACHE_VERSION = 1
CHECKPOINTS = (13, 25)


def model(seed: int) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=210,
        max_depth=3,
        learning_rate=0.035,
        min_child_weight=18,
        subsample=0.82,
        colsample_bytree=0.78,
        reg_alpha=0.18,
        reg_lambda=3.2,
        objective="reg:pseudohubererror",
        eval_metric="mae",
        n_jobs=-1,
        random_state=seed,
    )


def target_and_maturity(data) -> tuple[np.ndarray, np.ndarray]:
    targets = {
        horizon: data[f"target_h{horizon}"].to_numpy(float)
        for horizon in (1, 3, 6, 10)
    }
    target = adaptive_value(data, targets, exit_cost_scale=0.0)
    tenure = expected_tenure(data)
    end = data["target_h3_end_gw"].to_numpy(int).copy()
    end[tenure > 3] = data.loc[tenure > 3, "target_h6_end_gw"].to_numpy(int)
    end[tenure > 6] = data.loc[tenure > 6, "target_h10_end_gw"].to_numpy(int)
    return target, end


def causal_online_prediction(
    data,
    structural_adaptive: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    cache_path = lens.CACHE / f"decision-focused-adaptive-v{CACHE_VERSION}.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        if len(cached["prediction"]) == len(data):
            return cached["prediction"], json.loads(str(cached["audit"].item()))

    target, target_end = target_and_maturity(data)
    prediction = structural_adaptive.copy()
    seasons = list(dict.fromkeys(data["season"].tolist()))
    orders = data["season_order"].to_numpy(int)
    gameweeks = data["GW"].to_numpy(int)
    positions = data["position_id"].to_numpy(int)
    observed = data["fixture_count"].to_numpy(int) > 0
    frontier = selectable_frontier(data)
    audit: list[dict] = []
    for season_order in range(1, len(seasons)):
        season_mask = orders == season_order
        for checkpoint_index, checkpoint in enumerate(CHECKPOINTS):
            interval_end = (
                CHECKPOINTS[checkpoint_index + 1] - 1
                if checkpoint_index + 1 < len(CHECKPOINTS)
                else 99
            )
            for position in lens.SQUAD_QUOTAS:
                position_mask = positions == position
                train_mask = (
                    position_mask
                    & observed
                    & frontier
                    & (
                        (orders < season_order)
                        | (season_mask & (target_end < checkpoint))
                    )
                )
                test_mask = (
                    season_mask
                    & position_mask
                    & (gameweeks >= checkpoint)
                    & (gameweeks <= interval_end)
                )
                if not test_mask.any():
                    continue
                train = data.loc[train_mask]
                test = data.loc[test_mask]
                train_x, medians = feature_matrix(train)
                test_x, _ = feature_matrix(test, medians)
                age = season_order - train["season_order"].to_numpy(int)
                weights = np.power(0.88, np.maximum(age - 1, 0))
                current_season = (
                    train["season_order"].to_numpy(int) == season_order
                )
                weights *= np.where(current_season, 1.25, 1.0)
                # Focus additional loss on consequential, plausible selections
                # without turning the rare hindsight winner into the whole target.
                local_target = target[train_mask]
                threshold = np.quantile(local_target, 0.75)
                weights *= np.where(local_target >= threshold, 1.20, 1.0)
                fitted = model(
                    260900 + season_order * 100 + checkpoint_index * 10 + position
                )
                fitted.fit(train_x, local_target, sample_weight=weights)
                prediction[test_mask] = np.clip(fitted.predict(test_x), 0, 70)
                audit.append(
                    {
                        "season": seasons[season_order],
                        "checkpoint": checkpoint,
                        "position": int(position),
                        "trainingRows": int(train_mask.sum()),
                        "maturedCurrentSeasonRows": int(
                            (train_mask & season_mask).sum()
                        ),
                        "testRows": int(test_mask.sum()),
                    }
                )
            print(
                f"Decision-focused horizon {seasons[season_order]} GW{checkpoint}",
                flush=True,
            )
        prediction[season_mask & ~observed] = 0
    np.savez_compressed(
        cache_path,
        prediction=prediction,
        audit=json.dumps(audit),
        features=json.dumps(FEATURES),
    )
    return prediction, audit


def main() -> None:
    original, _ = lens.load_or_build_prepared_history()
    data = add_targets(original.reset_index(drop=True))
    scores, baseline_plan, captain = championship_forecasts(data)
    structural = structural_horizons(data, scores)
    structural_adaptive = adaptive_value(data, structural, 3.0)
    learned, _ = causal_ridge_horizons(data, structural)
    online_ridge, _ = causal_online_ridge_horizons(data, structural, learned)
    ridge_plan = quantile_map(
        data, adaptive_value(data, online_ridge, 3.0), baseline_plan
    )
    direct_raw, fit_audit = causal_online_prediction(data, structural_adaptive)
    direct_plan = quantile_map(data, direct_raw, baseline_plan)
    ensemble_raw = 0.50 * direct_plan + 0.50 * ridge_plan
    events = event_number(data)

    configs = {"baseline": baseline_plan}
    for share in (0.05, 0.10, 0.15):
        active = share * (events >= 13)
        configs[f"directAdaptive{int(share * 100)}"] = (
            baseline_plan + active * (direct_plan - baseline_plan)
        )
        configs[f"ridgeDirectEnsemble{int(share * 100)}"] = (
            baseline_plan + active * (ensemble_raw - baseline_plan)
        )

    seasons = list(dict.fromkeys(data["season"].tolist()))
    rows = []
    for name, plan in configs.items():
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
        "status": "decision-focused multi-timescale challenger",
        "method": (
            "Position-specific nonlinear models predict the exact discounted "
            "player-specific holding target on the selectable frontier. Each "
            "GW13/GW25 fit uses only labels whose required horizon has matured."
        ),
        "fitAudit": fit_audit,
        "selected": selected,
        "pairedVsBaseline": paired,
        "robustPromotion": robust,
        "experiments": rows,
    }
    output = lens.ROOT / "analysis" / "data" / "decision_focused_horizon_validation.json"
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
