"""Causal ranker for the transfer action actually available to an FPL manager.

Global player ranking is not the decision problem.  A legal transfer compares
players in the same position at nearby prices and values them over different
expected tenures.  This challenger trains LambdaMART only on those local price
neighbourhoods.  Two overlapping £1.5m bands avoid arbitrary bucket edges.

Predictions are made at GW13 and GW25.  Same-season labels are admitted only
after every gameweek needed by that player's adaptive holding target has
matured.  The first twelve events retain the frozen champion unchanged.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from xgboost import XGBRanker

import calibrate_model as lens
from bench_efficiency_validation import variant_summary
from decision_focused_horizon_validation import CHECKPOINTS, target_and_maturity
from feasible_decision_audit import decision_metrics
from frontier_ranker_validation import STRATEGY, selectable_frontier
from listwise_ranker_validation import quantile_map
from multiscale_horizon_validation import FEATURES, add_targets, feature_matrix
from multiscale_phase_validation import event_number
from probabilistic_component_challenger import causal_route_predictions
from wildcard_freehit_ablation import champion_forecasts


CACHE_VERSION = 2
PRICE_BAND = 15
SHIFTS = (0, 7)


def model(seed: int) -> XGBRanker:
    return XGBRanker(
        n_estimators=175,
        max_depth=3,
        learning_rate=0.035,
        min_child_weight=16,
        subsample=0.84,
        colsample_bytree=0.80,
        reg_alpha=0.18,
        reg_lambda=3.2,
        objective="rank:pairwise",
        eval_metric="ndcg@10",
        lambdarank_pair_method="mean",
        lambdarank_num_pair_per_sample=12,
        n_jobs=-1,
        random_state=seed,
    )


def action_matrix(
    frame: pd.DataFrame,
    champion_plan: np.ndarray,
    component: dict,
    medians: tuple[pd.Series, pd.Series] | None = None,
) -> tuple[np.ndarray, tuple[pd.Series, pd.Series]]:
    base_medians = medians[0] if medians is not None else None
    addition_medians = medians[1] if medians is not None else None
    base, base_medians = feature_matrix(frame, base_medians)
    indices = frame.index.to_numpy(int)
    routes = component["means"][indices]
    additions = pd.DataFrame(
        np.column_stack(
            [
                champion_plan[indices] / 35.0,
                component["total"][indices] / 8.0,
                component["stacked"][indices] / 8.0,
                component["sigma"][indices] / 6.0,
                routes[:, 0] / 2.0,
                routes[:, 1] / 2.0,
                routes[:, 2] / 0.5,
                routes[:, 3] / 0.5,
                routes[:, 4] / 1.2,
                routes[:, 5] / 5.0,
                routes[:, 6] / 4.0,
                routes[:, 7] / 2.0,
                routes[:, 8] / 2.0,
                routes[:, 9] / 2.0,
            ]
        ),
        index=frame.index,
        columns=[
            "champion_plan",
            "route_total",
            "route_stacked",
            "route_sigma",
            "route_appearance",
            "route_sixty",
            "route_goals",
            "route_assists",
            "route_clean",
            "route_saves",
            "route_conceded",
            "route_bonus",
            "route_dc",
            "route_other",
        ],
    ).replace([np.inf, -np.inf], np.nan)
    if addition_medians is None:
        addition_medians = additions.median().fillna(0)
    additions = additions.fillna(addition_medians)
    return (
        np.column_stack([base, additions.to_numpy(np.float32)]),
        (base_medians, addition_medians),
    )


def query_order(frame: pd.DataFrame, shift: int) -> tuple[np.ndarray, np.ndarray]:
    price_band = ((frame["price"].astype(int) + shift) // PRICE_BAND).astype(str)
    key = (
        frame["season_order"].astype(int).astype(str)
        + "-"
        + frame["GW"].astype(int).astype(str)
        + "-"
        + price_band
    )
    qid = pd.factorize(key, sort=True)[0]
    order = np.argsort(qid, kind="stable")
    return order, qid[order]


def causal_action_predictions(
    data: pd.DataFrame,
    champion_plan: np.ndarray,
    component: dict,
) -> tuple[np.ndarray, list[dict]]:
    cache_path = lens.CACHE / f"transfer-action-ranker-v{CACHE_VERSION}.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        if len(cached["prediction"]) == len(data):
            return cached["prediction"], json.loads(str(cached["audit"].item()))

    target, maturity = target_and_maturity(data)
    prediction_by_shift = np.tile(champion_plan[:, None], (1, len(SHIFTS)))
    seasons = list(dict.fromkeys(data["season"].tolist()))
    orders = data["season_order"].to_numpy(int)
    gws = data["GW"].to_numpy(int)
    positions = data["position_id"].to_numpy(int)
    observed = data["fixture_count"].to_numpy(int) > 0
    frontier = selectable_frontier(data)
    audit: list[dict] = []

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
                test_mask = (
                    season_mask
                    & position_mask
                    & (gws >= checkpoint)
                    & (gws <= interval_end)
                )
                if not test_mask.any():
                    continue
                train = data.loc[train_mask]
                test = data.loc[test_mask]
                train_x, medians = action_matrix(train, champion_plan, component)
                test_x, _ = action_matrix(test, champion_plan, component, medians)
                for shift_index, shift in enumerate(SHIFTS):
                    order, qid = query_order(train, shift)
                    fitted = model(
                        810000
                        + 1000 * season_order
                        + 100 * checkpoint_index
                        + 10 * position
                        + shift_index
                    )
                    fitted.fit(train_x[order], target[train_mask][order], qid=qid)
                    prediction_by_shift[test_mask, shift_index] = fitted.predict(test_x)
                audit.append(
                    {
                        "season": seasons[season_order],
                        "checkpoint": checkpoint,
                        "position": int(position),
                        "trainingRows": int(train_mask.sum()),
                        "maturedCurrentSeasonRows": int((train_mask & season_mask).sum()),
                        "testRows": int(test_mask.sum()),
                    }
                )
            print(f"Transfer-action ranks {seasons[season_order]} GW{checkpoint}", flush=True)

    prediction = prediction_by_shift.mean(axis=1)
    prediction[~observed] = -1e6
    np.savez_compressed(
        cache_path,
        prediction=prediction,
        shifts=prediction_by_shift,
        audit=json.dumps(audit),
        features=json.dumps(FEATURES),
    )
    return prediction, audit


def agreed_action_plan(data: pd.DataFrame, champion_plan: np.ndarray, share: float = 0.05) -> np.ndarray:
    """Apply the ranker only where both overlapping price bands agree."""
    cached = np.load(lens.CACHE / f"transfer-action-ranker-v{CACHE_VERSION}.npz")
    mapped_shifts = np.column_stack(
        [
            quantile_map(data, cached["shifts"][:, index], champion_plan)
            for index in range(len(SHIFTS))
        ]
    )
    consensus = mapped_shifts.mean(axis=1)
    shift_delta = mapped_shifts - champion_plan[:, None]
    agreement = np.sign(shift_delta[:, 0]) == np.sign(shift_delta[:, 1])
    active = share * (event_number(data) >= 13) * agreement
    return champion_plan + active * (consensus - champion_plan)


def main() -> None:
    original, _ = lens.load_or_build_prepared_history()
    data = add_targets(original.reset_index(drop=True))
    scores, champion_plan, captain = champion_forecasts(data)
    component, _ = causal_route_predictions(data, scores)
    raw, audit = causal_action_predictions(data, champion_plan, component)
    mapped = quantile_map(data, raw, champion_plan)
    cached_shifts = np.load(lens.CACHE / f"transfer-action-ranker-v{CACHE_VERSION}.npz")["shifts"]
    mapped_shifts = np.column_stack(
        [quantile_map(data, cached_shifts[:, index], champion_plan) for index in range(len(SHIFTS))]
    )
    consensus = mapped_shifts.mean(axis=1)
    shift_delta = mapped_shifts - champion_plan[:, None]
    agreement = np.sign(shift_delta[:, 0]) == np.sign(shift_delta[:, 1])
    strength = np.min(np.abs(shift_delta), axis=1)
    strength_rank = pd.Series(strength).groupby(
        [data["season"], data["GW"], data["position_id"]], sort=False
    ).rank(pct=True).to_numpy(float)
    events = event_number(data)
    target, _ = target_and_maturity(data)
    frontier = selectable_frontier(data)
    observed = data["fixture_count"].to_numpy(int) > 0
    evaluation = data["season"].isin(lens.EVALUATION_SEASONS).to_numpy()
    decision_mask = frontier & observed & evaluation & (events >= 13)

    plans: dict[str, np.ndarray] = {"champion": champion_plan}
    for share in (0.025, 0.05, 0.10, 0.15):
        active = share * (events >= 13)
        plans[f"action{int(share * 1000):03d}"] = champion_plan + active * (mapped - champion_plan)
    for share in (0.05, 0.10, 0.15):
        plans[f"consensus{int(share * 1000):03d}"] = agreed_action_plan(data, champion_plan, share)
    for threshold in (0.50, 0.67, 0.80):
        active = 0.10 * (events >= 13) * agreement * (strength_rank >= threshold)
        plans[f"strong{int(threshold * 100):02d}"] = champion_plan + active * (consensus - champion_plan)

    seasons = list(dict.fromkeys(data["season"].tolist()))
    rows = []
    for name, plan in plans.items():
        print(f"Recursive transfer-action challenger {name}", flush=True)
        totals, stats = lens.simulate_candidate(
            data,
            scores,
            STRATEGY,
            plan_scores=plan,
            captain_scores=captain,
            tracked_player_name="Salah",
        )
        development = totals[2:6]
        holdout = totals[6:]
        rows.append(
            {
                "name": name,
                "developmentStability": round(float(development.mean() - 0.25 * development.std()), 3),
                "holdoutAverage": round(float(holdout.mean()), 1),
                "summary": variant_summary(totals, stats, seasons),
            }
        )

    champion_row = rows[0]
    best_challenger = max(rows[1:], key=lambda row: row["developmentStability"])
    selected = max(rows, key=lambda row: row["developmentStability"])
    paired = [
        {
            "season": old["season"],
            "champion": old["points"],
            "challenger": new["points"],
            "delta": new["points"] - old["points"],
        }
        for old, new in zip(champion_row["summary"]["seasons"], best_challenger["summary"]["seasons"])
    ]
    result = {
        "status": "causal near-price transfer-action challenger",
        "method": "Two overlapping £1.5m price-band LambdaMART models rank legal positional substitutes by adaptive holding value; same-season targets must mature before training.",
        "selectionRule": "Blend selected on 2018/19-2021/22 stability; 2022/23-2025/26 is untouched holdout.",
        "fitAudit": audit,
        "decisionMetrics": {
            "champion": decision_metrics(data, champion_plan, target, decision_mask),
            "actionRanker": decision_metrics(data, mapped, target, decision_mask),
        },
        "selected": selected,
        "bestChallenger": best_challenger,
        "pairedBestChallengerVsChampion": paired,
        "robustPromotion": bool(
            best_challenger["developmentStability"] > champion_row["developmentStability"]
            and best_challenger["summary"]["average"] > champion_row["summary"]["average"]
            and best_challenger["holdoutAverage"] >= champion_row["holdoutAverage"]
            and sum(row["delta"] > 0 for row in paired) >= 5
        ),
        "experiments": rows,
    }
    output = lens.ROOT / "analysis" / "data" / "transfer_action_ranker_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "decisionMetrics": result["decisionMetrics"],
                "selected": selected["name"],
                "bestChallenger": best_challenger["name"],
                "robustPromotion": result["robustPromotion"],
                "paired": paired,
                "experiments": [
                    {
                        "name": row["name"],
                        "developmentStability": row["developmentStability"],
                        "average": row["summary"]["average"],
                        "minimum": row["summary"]["minimum"],
                        "holdoutAverage": row["holdoutAverage"],
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
