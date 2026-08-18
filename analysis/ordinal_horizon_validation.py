"""Decision-aligned six-week ranker without saturated point labels.

The previous LambdaMART horizon label clipped all six-week totals above 15.  This
challenger assigns ordinal relevance within each deadline and position, and an
online variant refits at two predeclared checkpoints using only horizon labels
whose complete windows have finished.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from xgboost import XGBRanker

import calibrate_model as lens
from bench_efficiency_validation import championship_forecasts, variant_summary
from frontier_ranker_validation import FEATURES as BASE_FEATURES
from frontier_ranker_validation import STRATEGY, selectable_frontier
from listwise_ranker_validation import quantile_map


CACHE_VERSION = 1
CHECKPOINTS = (1, 13, 25)
FEATURES = BASE_FEATURES + [
    "component_horizon_censored",
    "causal_horizon_ridge",
    "horizon_weighted_games_censored",
    "transfer_pressure_rank",
    "price_rise_probability",
    "price_fall_probability",
    "team_rating_confidence",
    "team_regime_shift",
    "opponent_rating_confidence",
    "opponent_regime_shift",
    "fixture_censored",
    "team_context",
    "team_attack",
    "team_defence",
    "recent_underlying",
    "long_underlying",
    "recent_value",
    "long_value",
    "age_score",
    "competition_pressure",
    "rotation_volatility",
    "defensive_return_probability",
    "position_id",
    "GW",
]


def matrix(
    frame: pd.DataFrame,
    medians: pd.Series | None = None,
) -> tuple[np.ndarray, pd.Series]:
    values = frame[FEATURES].replace([np.inf, -np.inf], np.nan).astype(float).copy()
    values["price"] /= 150.0
    values["selected"] = np.log1p(values["selected"].clip(lower=0)) / 16.0
    values["component_horizon_censored"] /= 30.0
    values["causal_horizon_ridge"] /= 30.0
    values["horizon_weighted_games_censored"] /= 6.0
    values["expected_minutes"] /= 90.0
    values["minutes_std"] /= 40.0
    values["GW"] /= 38.0
    if medians is None:
        medians = values.median().fillna(0)
    return values.fillna(medians).to_numpy(np.float32), medians


def ordinal_relevance(frame: pd.DataFrame) -> np.ndarray:
    percentile = frame["horizon_target"].groupby(
        [frame["season_order"], frame["GW"], frame["position_id"]]
    ).rank(method="average", pct=True)
    return np.rint(31 * percentile).clip(0, 31).to_numpy(np.int32)


def model(seed: int) -> XGBRanker:
    return XGBRanker(
        n_estimators=180,
        max_depth=4,
        learning_rate=0.04,
        min_child_weight=14,
        subsample=0.84,
        colsample_bytree=0.84,
        reg_alpha=0.12,
        reg_lambda=2.6,
        objective="rank:ndcg",
        eval_metric="ndcg@15",
        lambdarank_pair_method="topk",
        lambdarank_num_pair_per_sample=20,
        n_jobs=-1,
        random_state=seed,
    )


def causal_predictions(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    cache_path = lens.CACHE / f"ordinal-horizon-online-v{CACHE_VERSION}.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        if len(cached["online"]) == len(data):
            return (
                cached["static"],
                cached["online"],
                json.loads(str(cached["audit"].item())),
            )
    baseline = data["component_horizon_censored"].to_numpy(float)
    static = baseline.copy()
    online = baseline.copy()
    orders = data["season_order"].to_numpy(int)
    gameweeks = data["GW"].to_numpy(int)
    target_end = data["horizon_target_end_gw"].to_numpy(int)
    observed = data["fixture_count"].to_numpy(int) > 0
    frontier = selectable_frontier(data)
    seasons = list(dict.fromkeys(data["season"].tolist()))
    audit = []
    for season_order in range(1, len(seasons)):
        season_mask = orders == season_order
        for checkpoint_index, checkpoint in enumerate(CHECKPOINTS):
            end = (
                CHECKPOINTS[checkpoint_index + 1] - 1
                if checkpoint_index + 1 < len(CHECKPOINTS)
                else 99
            )
            train_mask = observed & frontier & (
                (orders < season_order)
                | ((orders == season_order) & (target_end < checkpoint))
            )
            test_mask = season_mask & (gameweeks >= checkpoint) & (gameweeks <= end)
            train = data.loc[train_mask].sort_values(
                ["season_order", "GW", "position_id"], kind="stable"
            )
            test = data.loc[test_mask]
            train_x, medians = matrix(train)
            test_x, _ = matrix(test, medians)
            qid = pd.factorize(
                train["season_order"].astype(str)
                + "-"
                + train["GW"].astype(str)
                + "-"
                + train["position_id"].astype(str),
                sort=False,
            )[0]
            fitted = model(720000 + season_order * 10 + checkpoint_index)
            fitted.fit(train_x, ordinal_relevance(train), qid=qid)
            online[test_mask] = fitted.predict(test_x)
            if checkpoint == 1:
                full_test = data.loc[season_mask]
                full_x, _ = matrix(full_test, medians)
                static[season_mask] = fitted.predict(full_x)
            audit.append(
                {
                    "season": seasons[season_order],
                    "checkpoint": checkpoint,
                    "trainingRows": int(train_mask.sum()),
                    "maturedCurrentSeasonRows": int(
                        (train_mask & season_mask).sum()
                    ),
                    "testRows": int(test_mask.sum()),
                }
            )
            print(
                f"Ordinal horizon {seasons[season_order]} checkpoint {checkpoint}",
                flush=True,
            )
        static[season_mask & ~observed] = -1e6
        online[season_mask & ~observed] = -1e6
    payload = {
        "features": FEATURES,
        "checkpoints": list(CHECKPOINTS),
        "fits": audit,
    }
    np.savez_compressed(
        cache_path,
        static=static,
        online=online,
        audit=json.dumps(payload),
    )
    return static, online, payload


def main() -> None:
    data, _ = lens.load_or_build_prepared_history()
    data = data.reset_index(drop=True)
    scores, plan, captain = championship_forecasts(data)
    static_raw, online_raw, audit = causal_predictions(data)
    static_mapped = quantile_map(data, static_raw, plan)
    online_mapped = quantile_map(data, online_raw, plan)
    configs = [("baseline", plan)]
    for share in (0.10, 0.25, 0.40):
        configs.append(
            (f"staticOrdinal{int(share * 100)}", (1 - share) * plan + share * static_mapped)
        )
        configs.append(
            (f"onlineOrdinal{int(share * 100)}", (1 - share) * plan + share * online_mapped)
        )
    seasons = list(dict.fromkeys(data["season"].tolist()))
    rows = []
    raw_results = {}
    for name, challenger_plan in configs:
        print(f"Running {name}", flush=True)
        totals, stats = lens.simulate_candidate(
            data,
            scores,
            STRATEGY,
            plan_scores=challenger_plan,
            captain_scores=captain,
            tracked_player_name="Salah",
        )
        training = totals[: len(lens.TRAINING_SEASONS)]
        row = {
            "name": name,
            "trainingStability": round(
                float(training.mean() - 0.25 * training.std()), 3
            ),
            "summary": variant_summary(totals, stats, seasons),
        }
        rows.append(row)
        raw_results[name] = (totals, stats, challenger_plan)
    selected = max(rows, key=lambda row: row["trainingStability"])
    result = {
        "status": "training-selected ordinal horizon challenger",
        "method": (
            "Within-deadline, within-position relevance 0-31 avoids the old "
            "15-point ceiling. Online refits at GW13 and GW25 use only six-week "
            "labels whose windows have fully completed."
        ),
        "oldLabelSaturation": {
            "rowsAbove15": round(
                float(
                    (
                        data.loc[data["fixture_count"].gt(0), "horizon_target"] > 15
                    ).mean()
                ),
                4,
            )
        },
        "trainingAudit": audit,
        "selected": selected,
        "experiments": rows,
    }
    output = lens.ROOT / "analysis" / "data" / "ordinal_horizon_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "selected": selected["name"],
                "selectedAverage": selected["summary"]["average"],
                "selectedMinimum": selected["summary"]["minimum"],
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
