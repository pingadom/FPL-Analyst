"""Causal captain-specific listwise challenger on the armband frontier."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from xgboost import XGBRanker

import calibrate_model as lens
from frontier_ranker_validation import FEATURES, PLAYER_CANDIDATE, STRATEGY, matrix


CAPTAIN_FEATURES = FEATURES + [
    "haul8_probability",
    "return5_probability",
    "team_attack_rating",
    "opponent_defence_rating",
    "minutes_security",
    "fixture_now",
]


def captain_matrix(frame: pd.DataFrame, medians: pd.Series | None = None):
    values = frame[CAPTAIN_FEATURES].replace([np.inf, -np.inf], np.nan).copy()
    values["price"] = values["price"] / 10.0
    values["selected"] = np.log1p(values["selected"].clip(lower=0))
    if medians is None:
        medians = values.median().fillna(0)
    return values.fillna(medians).to_numpy(np.float32), medians


def causal_captain_rank(data: pd.DataFrame, structural: np.ndarray) -> np.ndarray:
    path = lens.CACHE / "captain-listwise-v1.npz"
    if path.exists():
        cached = np.load(path)
        if len(cached["prediction"]) == len(data):
            return cached["prediction"]
    group_rank = pd.Series(structural, index=data.index).groupby(
        [data["season"], data["GW"]]
    ).rank(method="first", ascending=False)
    frontier = (group_rank <= 40) & (data["play_probability"] >= 0.45)
    orders = data["season_order"].to_numpy(int)
    prediction = structural.copy()
    seasons = list(dict.fromkeys(data["season"].tolist()))
    for season_order in range(1, len(seasons)):
        train_mask = (orders < season_order) & frontier.to_numpy(bool) & (data["fixture_count"].to_numpy(int) > 0)
        test_mask = orders == season_order
        train = data.loc[train_mask].sort_values(["season_order", "GW"], kind="stable")
        test = data.loc[test_mask]
        train_x, medians = captain_matrix(train)
        test_x, _ = captain_matrix(test, medians)
        query = pd.factorize(
            train["season_order"].astype(str) + "-" + train["GW"].astype(str), sort=False
        )[0]
        relevance = np.rint(train["points"].clip(0, 15)).to_numpy(np.int32)
        model = XGBRanker(
            n_estimators=260,
            max_depth=4,
            learning_rate=0.035,
            min_child_weight=10,
            subsample=0.84,
            colsample_bytree=0.84,
            reg_alpha=0.10,
            reg_lambda=2.2,
            objective="rank:ndcg",
            eval_metric="ndcg@5",
            lambdarank_pair_method="topk",
            lambdarank_num_pair_per_sample=8,
            n_jobs=-1,
            random_state=410000 + season_order,
        )
        model.fit(train_x, relevance, qid=query)
        prediction[test_mask] = model.predict(test_x)
        print(f"Captain listwise: {seasons[season_order]}")
    np.savez_compressed(path, prediction=prediction)
    return prediction


def rank_blend(data: pd.DataFrame, structural: np.ndarray, challenger: np.ndarray, share: float) -> np.ndarray:
    result = structural.copy()
    for indices in data.groupby(["season", "GW"], sort=False).groups.values():
        local = np.asarray(indices, dtype=int)
        structural_percentile = pd.Series(structural[local]).rank(pct=True).to_numpy(float)
        challenger_percentile = pd.Series(challenger[local]).rank(pct=True).to_numpy(float)
        result[local] = (1 - share) * structural_percentile + share * challenger_percentile
    return result


def main() -> None:
    data, _ = lens.load_or_build_prepared_history()
    data = data.reset_index(drop=True)
    scores, horizon, _ = lens.candidate_forecasts(
        data, PLAYER_CANDIDATE, robust_planning=False, schedule_censored=True
    )
    plan = 0.75 * scores * 4.5 + 0.25 * horizon
    challenger = causal_captain_rank(data, scores)
    variants = {"structuralCaptain": None}
    for share in (0.25, 0.50, 0.75, 1.0):
        variants[f"captainList{int(share * 100)}"] = rank_blend(data, scores, challenger, share)
    seasons = list(dict.fromkeys(data["season"].tolist()))
    benchmark = json.loads(
        (lens.ROOT / "analysis" / "data" / "historical_rank_benchmarks.json").read_text(encoding="utf-8")
    )
    targets = {row["season"].replace("/", "-"): int(row["points"]) for row in benchmark["seasons"]}
    models = {}
    for name, captain_score in variants.items():
        totals, _ = lens.simulate_candidate(
            data, scores, STRATEGY, plan_scores=plan, captain_scores=captain_score
        )
        evaluation = [
            {
                "season": seasons[index].replace("-", "/"),
                "points": int(round(float(totals[index]))),
                "target": targets[seasons[index]],
                "margin": int(round(float(totals[index]))) - targets[seasons[index]],
            }
            for index in range(2, len(seasons))
        ]
        models[name] = {
            "average": round(float(totals[2:].mean()), 1),
            "minimum": int(round(float(totals[2:].min()))),
            "targetHits": sum(row["margin"] >= 0 for row in evaluation),
            "averageMargin": round(float(np.mean([row["margin"] for row in evaluation])), 1),
            "evaluation": evaluation,
        }
        print(name, models[name]["average"])
    output = lens.ROOT / "analysis" / "data" / "captain_ranker_validation.json"
    output.write_text(
        json.dumps(
            {
                "status": "challenger-only",
                "method": "Causal LambdaMART NDCG@5 trained on each deadline's top-40 armband frontier; evaluated without changing squad or transfer forecasts.",
                "models": models,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({name: {key: value for key, value in row.items() if key != "evaluation"} for name, row in models.items()}, indent=2))


if __name__ == "__main__":
    main()
