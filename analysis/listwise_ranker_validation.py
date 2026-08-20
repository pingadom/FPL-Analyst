"""Causal listwise ranker aligned to FPL's selectable decision frontier."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from xgboost import XGBRanker

import calibrate_model as lens
from frontier_ranker_validation import (
    FEATURES,
    PLAYER_CANDIDATE,
    STRATEGY,
    matrix,
    selectable_frontier,
)


CACHE_VERSION = 2


def fitted_ranker(seed: int) -> XGBRanker:
    return XGBRanker(
        n_estimators=240,
        max_depth=4,
        learning_rate=0.035,
        min_child_weight=12,
        subsample=0.82,
        colsample_bytree=0.82,
        reg_alpha=0.12,
        reg_lambda=2.4,
        objective="rank:ndcg",
        eval_metric="ndcg@15",
        lambdarank_pair_method="topk",
        lambdarank_num_pair_per_sample=15,
        n_jobs=-1,
        random_state=seed,
    )


def causal_rank_predictions(data: pd.DataFrame, target_column: str) -> np.ndarray:
    cache_path = lens.CACHE / f"listwise-{target_column}-v{CACHE_VERSION}.npz"
    baseline = data["component_xpts"].to_numpy(float)
    fingerprint = lens.frame_fingerprint(
        data,
        [*FEATURES, target_column, "component_xpts"],
        f"listwise-{target_column}-v{CACHE_VERSION}",
    )
    if cache_path.exists():
        cached = np.load(cache_path)
        if (
            len(cached["prediction"]) == len(data)
            and "fingerprint" in cached.files
            and str(cached["fingerprint"].item()) == fingerprint
        ):
            return cached["prediction"]
    prediction = baseline.copy()
    frontier = selectable_frontier(data)
    orders = data["season_order"].to_numpy(int)
    positions = data["position_id"].to_numpy(int)
    seasons = list(dict.fromkeys(data["season"].tolist()))
    for season_order in range(1, len(seasons)):
        for position in lens.SQUAD_QUOTAS:
            train_mask = (
                (orders < season_order)
                & (positions == position)
                & frontier
                & (data["fixture_count"].to_numpy(int) > 0)
            )
            test_mask = (orders == season_order) & (positions == position)
            train = data.loc[train_mask].sort_values(["season_order", "GW"], kind="stable")
            test = data.loc[test_mask]
            train_x, medians = matrix(train)
            test_x, _ = matrix(test, medians)
            relevance = np.rint(train[target_column].clip(0, 15)).to_numpy(np.int32)
            query = pd.factorize(
                train["season_order"].astype(str) + "-" + train["GW"].astype(str),
                sort=False,
            )[0]
            ranker = fitted_ranker(310000 + season_order * 10 + int(position))
            ranker.fit(train_x, relevance, qid=query)
            prediction[test_mask] = ranker.predict(test_x)
            print(f"Listwise {target_column}: {seasons[season_order]} position {position}")
        prediction[(orders == season_order) & (data["fixture_count"].to_numpy(int) == 0)] = -1e6
    np.savez_compressed(
        cache_path, prediction=prediction, fingerprint=fingerprint
    )
    return prediction


def quantile_map(data: pd.DataFrame, ranks: np.ndarray, reference: np.ndarray) -> np.ndarray:
    mapped = reference.copy()
    for group_indices in data.groupby(["season", "GW", "position_id"], sort=False).groups.values():
        indices = np.asarray(group_indices, dtype=int)
        order = indices[np.argsort(ranks[indices], kind="stable")]
        mapped[order] = np.sort(reference[indices])
    mapped[data["fixture_count"].to_numpy(int) == 0] = 0
    return mapped


def main() -> None:
    data, _ = lens.load_or_build_prepared_history()
    data = data.reset_index(drop=True)
    immediate, horizon, _ = lens.candidate_forecasts(
        data, PLAYER_CANDIDATE, robust_planning=False, schedule_censored=True
    )
    stable_plan = 0.75 * immediate * 4.5 + 0.25 * horizon
    immediate_rank = causal_rank_predictions(data, "points")
    horizon_rank = causal_rank_predictions(data, "horizon_target")
    mapped_immediate = quantile_map(data, immediate_rank, immediate)
    mapped_plan = quantile_map(data, horizon_rank, stable_plan)
    variants: dict[str, tuple[np.ndarray, np.ndarray]] = {"stable": (immediate, stable_plan)}
    for share in (0.10, 0.25, 0.40):
        variants[f"listImmediate{int(share * 100)}"] = (
            (1 - share) * immediate + share * mapped_immediate,
            stable_plan,
        )
        variants[f"listPlan{int(share * 100)}"] = (
            immediate,
            (1 - share) * stable_plan + share * mapped_plan,
        )
    variants["listBoth25"] = (
        0.75 * immediate + 0.25 * mapped_immediate,
        0.75 * stable_plan + 0.25 * mapped_plan,
    )
    seasons = list(dict.fromkeys(data["season"].tolist()))
    targets_payload = json.loads(
        (lens.ROOT / "analysis" / "data" / "historical_rank_benchmarks.json").read_text(encoding="utf-8")
    )
    targets = {row["season"].replace("/", "-"): int(row["points"]) for row in targets_payload["seasons"]}
    models = {}
    for name, (score, plan) in variants.items():
        totals, stats = lens.simulate_candidate(data, score, STRATEGY, plan_scores=plan)
        evaluation = []
        for index in range(2, len(seasons)):
            season = seasons[index]
            points = int(round(float(totals[index])))
            evaluation.append(
                {
                    "season": season.replace("-", "/"),
                    "points": points,
                    "target": targets[season],
                    "margin": points - targets[season],
                    "transfers": stats[index]["transfers"],
                }
            )
        models[name] = {
            "average": round(float(totals[2:].mean()), 1),
            "minimum": int(round(float(totals[2:].min()))),
            "targetHits": sum(row["margin"] >= 0 for row in evaluation),
            "averageMargin": round(float(np.mean([row["margin"] for row in evaluation])), 1),
            "evaluation": evaluation,
        }
        print(name, models[name]["average"])
    result = {
        "status": "challenger-only",
        "method": "Position-specific LambdaMART NDCG@15, trained only on prior seasons and deadline-selectable players; ranks are quantile-mapped to the frozen structural utility scale.",
        "selection": "Immediate, horizon and joint 10/25/40% blends specified before recursive evaluation.",
        "models": models,
    }
    output = lens.ROOT / "analysis" / "data" / "listwise_ranker_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: {key: value for key, value in row.items() if key != "evaluation"} for name, row in models.items()}, indent=2))


if __name__ == "__main__":
    main()
