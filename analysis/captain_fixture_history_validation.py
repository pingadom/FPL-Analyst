"""Decision-focused captaincy with causal, shrunk fixture history.

Raw player-versus-opponent records are tiny and non-stationary.  This audit
therefore exposes them only through strong empirical-Bayes shrinkage toward the
player's deadline-known general record.  The learned challenger is fitted on
earlier seasons plus already completed same-season checkpoints, while the
captain evaluation holds the squad and XI fixed.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from xgboost import XGBRanker

import calibrate_model as lens
from captain_ranker_validation import CAPTAIN_FEATURES
from frontier_ranker_validation import STRATEGY
from multiscale_horizon_validation import add_targets
from premium_captain_validation import captain_variants
from probabilistic_minutes_validation import season_summary
from wildcard_freehit_ablation import champion_forecasts


CHECKPOINTS = (1, 13, 25)
SEEDS = (0, 1, 2)
CACHE_VERSION = 2
BASE_FEATURES = list(
    dict.fromkeys(
        CAPTAIN_FEATURES
        + [
            "prediction_p10",
            "prediction_p90",
            "blank_probability",
            "goal_rate",
            "assist_rate",
            "clean_sheet_rate",
            "bonus_rate",
            "team_form_rating",
            "team_clean_rating",
            "opponent_clean_rating",
            "team_rating_confidence",
            "opponent_rating_confidence",
            "prediction_uncertainty",
            "price_rise_probability",
            "price_fall_probability",
            "team_rest_days",
            "was_home",
            "position_id",
            "GW",
        ]
    )
)
HISTORY_FEATURES = [
    "h2h_matches",
    "h2h_points_shrunk",
    "h2h_points_uplift",
    "h2h_return_shrunk",
    "h2h_haul_shrunk",
    "h2h_recent_shrunk",
    "venue_h2h_matches",
    "venue_h2h_points_shrunk",
    "h2h_evidence",
]


def add_fixture_history(data: pd.DataFrame) -> pd.DataFrame:
    work = data.copy()
    work["opponent_key"] = (
        work["opponent_name"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9]", "", regex=True)
    )
    single_played = (
        work["fixture_count"].eq(1)
        & work["minutes"].gt(0)
        & work["opponent_key"].ne("")
    )
    work["history_event"] = single_played.astype(float)
    work["history_points"] = work["points"].where(single_played, 0.0)
    work["history_points_observed"] = work["points"].where(single_played)
    work["history_return"] = (work["points"] >= 5).where(single_played, False).astype(float)
    work["history_haul"] = (work["points"] >= 8).where(single_played, False).astype(float)

    player_group = work.groupby("player_key", sort=False)
    general_matches = player_group["history_event"].cumsum() - work["history_event"]
    general_returns = player_group["history_return"].cumsum() - work["history_return"]
    general_hauls = player_group["history_haul"].cumsum() - work["history_haul"]
    position_return_prior = work["position_id"].map({1: 0.24, 2: 0.22, 3: 0.31, 4: 0.30})
    position_haul_prior = work["position_id"].map({1: 0.09, 2: 0.08, 3: 0.15, 4: 0.15})
    general_return = (general_returns + 8 * position_return_prior) / (general_matches + 8)
    general_haul = (general_hauls + 10 * position_haul_prior) / (general_matches + 10)

    pair_keys = ["player_key", "opponent_key"]
    pair_group = work.groupby(pair_keys, sort=False)
    work["h2h_matches"] = pair_group["history_event"].cumsum() - work["history_event"]
    h2h_points = pair_group["history_points"].cumsum() - work["history_points"]
    h2h_returns = pair_group["history_return"].cumsum() - work["history_return"]
    h2h_hauls = pair_group["history_haul"].cumsum() - work["history_haul"]
    recent = pair_group["history_points_observed"].transform(
        lambda values: values.ewm(alpha=0.45, adjust=False).mean().shift(1)
    )
    strength = 8.0
    work["h2h_points_shrunk"] = (
        h2h_points + strength * work["long_raw"]
    ) / (work["h2h_matches"] + strength)
    work["h2h_points_uplift"] = (
        work["h2h_points_shrunk"] - work["long_raw"]
    ).clip(-1.5, 1.5)
    work["h2h_return_shrunk"] = (
        h2h_returns + strength * general_return
    ) / (work["h2h_matches"] + strength)
    work["h2h_haul_shrunk"] = (
        h2h_hauls + 10.0 * general_haul
    ) / (work["h2h_matches"] + 10.0)
    work["h2h_recent_shrunk"] = (
        work["h2h_matches"].clip(upper=4) * recent.fillna(work["long_raw"])
        + 6.0 * work["long_raw"]
    ) / (work["h2h_matches"].clip(upper=4) + 6.0)
    work["h2h_evidence"] = 1 - np.exp(-work["h2h_matches"] / 3.0)

    venue_keys = pair_keys + ["was_home"]
    venue_group = work.groupby(venue_keys, sort=False)
    work["venue_h2h_matches"] = (
        venue_group["history_event"].cumsum() - work["history_event"]
    )
    venue_points = venue_group["history_points"].cumsum() - work["history_points"]
    venue_strength = 10.0
    work["venue_h2h_points_shrunk"] = (
        venue_points + venue_strength * work["long_raw"]
    ) / (work["venue_h2h_matches"] + venue_strength)
    return work


def feature_matrix(
    frame: pd.DataFrame,
    include_history: bool,
    medians: pd.Series | None = None,
) -> tuple[np.ndarray, pd.Series]:
    features = BASE_FEATURES + (HISTORY_FEATURES if include_history else [])
    values = frame[features].replace([np.inf, -np.inf], np.nan).astype(float).copy()
    for column, divisor in {
        "price": 150.0,
        "expected_minutes": 90.0,
        "minutes_std": 40.0,
        "team_rest_days": 14.0,
        "GW": 38.0,
    }.items():
        if column in values:
            values[column] /= divisor
    if "selected" in values:
        values["selected"] = np.log1p(values["selected"].clip(lower=0)) / 16.0
    for column in ("h2h_matches", "venue_h2h_matches"):
        if column in values:
            values[column] = np.log1p(values[column]) / 3.0
    if medians is None:
        medians = values.median().fillna(0)
    return values.fillna(medians).to_numpy(np.float32), medians


def ranker(seed: int) -> XGBRanker:
    return XGBRanker(
        n_estimators=190,
        max_depth=3,
        learning_rate=0.035,
        min_child_weight=14,
        subsample=0.84,
        colsample_bytree=0.82,
        reg_alpha=0.16,
        reg_lambda=3.0,
        objective="rank:ndcg",
        eval_metric="ndcg@5",
        lambdarank_pair_method="topk",
        lambdarank_num_pair_per_sample=8,
        n_jobs=-1,
        random_state=seed,
    )


def causal_predictions(
    data: pd.DataFrame,
    structural: np.ndarray,
    include_history: bool,
) -> np.ndarray:
    label = "history" if include_history else "context"
    path = lens.CACHE / f"captain-fixture-{label}-v{CACHE_VERSION}.npz"
    if path.exists():
        cached = np.load(path)
        if cached["predictions"].shape == (len(data), len(SEEDS)):
            return cached["predictions"]

    structural_rank = pd.Series(structural, index=data.index).groupby(
        [data["season"], data["GW"]], sort=False
    ).rank(method="first", ascending=False)
    frontier = (
        (structural_rank <= 40)
        & (data["play_probability"] >= 0.45)
        & (data["fixture_count"] > 0)
    ).to_numpy(bool)
    orders = data["season_order"].to_numpy(int)
    gws = data["GW"].to_numpy(int)
    seasons = list(dict.fromkeys(data["season"].tolist()))
    predictions = np.tile(structural[:, None], (1, len(SEEDS)))
    audit = []
    for season_order in range(1, len(seasons)):
        season_mask = orders == season_order
        for checkpoint_index, checkpoint in enumerate(CHECKPOINTS):
            interval_end = (
                CHECKPOINTS[checkpoint_index + 1] - 1
                if checkpoint_index + 1 < len(CHECKPOINTS)
                else 99
            )
            train_mask = frontier & (
                (orders < season_order)
                | (season_mask & (gws < checkpoint))
            )
            test_mask = season_mask & (gws >= checkpoint) & (gws <= interval_end)
            train = data.loc[train_mask].sort_values(
                ["season_order", "GW"], kind="stable"
            )
            test = data.loc[test_mask]
            train_x, medians = feature_matrix(train, include_history)
            test_x, _ = feature_matrix(test, include_history, medians)
            query = pd.factorize(
                train["season_order"].astype(str) + "-" + train["GW"].astype(str),
                sort=False,
            )[0]
            target = np.rint(train["points"].clip(0, 15)).to_numpy(np.int32)
            for seed_index, seed in enumerate(SEEDS):
                fitted = ranker(
                    870000
                    + 10000 * int(include_history)
                    + 1000 * seed_index
                    + 100 * season_order
                    + checkpoint_index
                )
                fitted.fit(train_x, target, qid=query)
                predictions[test_mask, seed_index] = fitted.predict(test_x)
            audit.append(
                {
                    "season": seasons[season_order],
                    "checkpoint": checkpoint,
                    "trainingRows": int(train_mask.sum()),
                    "sameSeasonRows": int((train_mask & season_mask).sum()),
                    "testRows": int(test_mask.sum()),
                }
            )
            print(f"Captain {label}: {seasons[season_order]} GW{checkpoint}", flush=True)
    np.savez_compressed(path, predictions=predictions, audit=json.dumps(audit))
    return predictions


def percentile(data: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    return pd.Series(values, index=data.index).groupby(
        [data["season"], data["GW"]], sort=False
    ).rank(pct=True).to_numpy(float)


def history_score(data: pd.DataFrame) -> np.ndarray:
    points = percentile(data, data["h2h_points_shrunk"].to_numpy(float))
    recent = percentile(data, data["h2h_recent_shrunk"].to_numpy(float))
    returns = percentile(data, data["h2h_return_shrunk"].to_numpy(float))
    hauls = percentile(data, data["h2h_haul_shrunk"].to_numpy(float))
    venue = percentile(data, data["venue_h2h_points_shrunk"].to_numpy(float))
    evidence = data["h2h_evidence"].to_numpy(float)
    raw = 0.30 * points + 0.20 * recent + 0.20 * returns + 0.20 * hauls + 0.10 * venue
    neutral = percentile(data, data["long_raw"].to_numpy(float))
    return evidence * raw + (1 - evidence) * neutral


def realised_captain_bonus(
    indices: list[int],
    metric: np.ndarray,
    points: np.ndarray,
    minutes: np.ndarray,
) -> tuple[float, int, int]:
    order = sorted(indices, key=lambda index: metric[index], reverse=True)
    captain, vice = order[:2]
    active = captain if minutes[captain] > 0 else vice if minutes[vice] > 0 else -1
    return (float(points[active]) if active >= 0 else 0.0), captain, vice


def decision_evaluation(
    data: pd.DataFrame,
    stats: list[dict],
    base_totals: np.ndarray,
    frozen: np.ndarray,
    variants: dict[str, np.ndarray],
) -> tuple[dict, list[dict]]:
    context = lens.simulation_context(data)
    actual = data["points"].to_numpy(float)
    minutes = data["minutes"].to_numpy(float)
    results = {
        name: {
            "delta": np.zeros(len(context["seasons"]), dtype=float),
            "captainPoints": [],
            "oracleRegret": [],
            "changes": 0,
            "positiveChanges": 0,
            "negativeChanges": 0,
            "decisions": [],
        }
        for name in variants
    }
    frozen_choices = {}
    for season_index, season_context in enumerate(context["seasons"]):
        selections = {
            int(row["gw"]): row
            for row in stats[season_index]["selectionLog"]
        }
        for gw in season_context["weeks"]:
            week_indices = np.asarray(season_context["weekIndices"][gw], dtype=int)
            element_to_index = {
                int(data.at[index, "element"]): int(index) for index in week_indices
            }
            xi = [
                element_to_index[element]
                for element in selections[int(gw)]["xi"]
                if element in element_to_index
            ]
            if len(xi) != 11:
                continue
            frozen_bonus, frozen_captain, _ = realised_captain_bonus(
                xi, frozen, actual, minutes
            )
            frozen_choices[(season_index, int(gw))] = frozen_captain
            oracle = max(
                (actual[index] for index in xi if minutes[index] > 0),
                default=0.0,
            )
            for name, metric in variants.items():
                bonus, captain, vice = realised_captain_bonus(
                    xi, metric, actual, minutes
                )
                change = bonus - frozen_bonus
                record = results[name]
                record["delta"][season_index] += change
                record["captainPoints"].append(bonus)
                record["oracleRegret"].append(oracle - bonus)
                if captain != frozen_captain:
                    record["changes"] += 1
                    record["positiveChanges"] += int(change > 0)
                    record["negativeChanges"] += int(change < 0)
                    if data.at[week_indices[0], "season"] in lens.EVALUATION_SEASONS:
                        record["decisions"].append(
                            {
                                "season": str(data.at[week_indices[0], "season"]).replace("-", "/"),
                                "gw": int(gw),
                                "from": str(data.at[frozen_captain, "display_name"]),
                                "to": str(data.at[captain, "display_name"]),
                                "vice": str(data.at[vice, "display_name"]),
                                "delta": int(change),
                                "toH2HMatches": int(data.at[captain, "h2h_matches"]),
                                "toH2HUplift": round(float(data.at[captain, "h2h_points_uplift"]), 3),
                            }
                        )

    seasons = list(dict.fromkeys(data["season"].tolist()))
    rows = []
    for name, record in results.items():
        totals = base_totals + record["delta"]
        summary = season_summary(totals, seasons)
        evaluation_delta = record["delta"][2:]
        rows.append(
            {
                "name": name,
                **summary,
                "averageDelta": round(float(evaluation_delta.mean()), 1),
                "developmentDelta": round(float(evaluation_delta[:-2].mean()), 1),
                "holdoutDelta": round(float(evaluation_delta[-2:].mean()), 1),
                "worstSeasonDelta": int(evaluation_delta.min()),
                "improvedSeasons": int((evaluation_delta > 0).sum()),
                "declinedSeasons": int((evaluation_delta < 0).sum()),
                "seasonDeltas": evaluation_delta.astype(int).tolist(),
                "changedCaptains": int(record["changes"]),
                "positiveChanges": int(record["positiveChanges"]),
                "negativeChanges": int(record["negativeChanges"]),
                "captainPointsPerWeek": round(float(np.mean(record["captainPoints"])), 3),
                "oracleRegretPerWeek": round(float(np.mean(record["oracleRegret"])), 3),
                "changedDecisionLog": record["decisions"],
            }
        )
    return {row["name"]: row for row in rows}, rows


def correlation_audit(data: pd.DataFrame, structural: np.ndarray) -> list[dict]:
    rows = []
    evaluation = data["season"].isin(lens.EVALUATION_SEASONS)
    frontier_rank = pd.Series(structural, index=data.index).groupby(
        [data["season"], data["GW"]], sort=False
    ).rank(ascending=False, method="first")
    base = evaluation & (data["fixture_count"] > 0) & (frontier_rank <= 40)
    for minimum in (0, 1, 2, 3, 4):
        mask = base & (data["h2h_matches"] >= minimum)
        actual = data.loc[mask, "points"].to_numpy(float)
        rows.append(
            {
                "minimumPriorMatches": minimum,
                "rows": int(mask.sum()),
                "coverage": round(float(mask.sum() / max(base.sum(), 1)), 4),
                "rawH2HSpearman": round(
                    float(spearmanr(data.loc[mask, "h2h_points_shrunk"], actual).statistic),
                    4,
                ),
                "structuralSpearman": round(
                    float(spearmanr(structural[mask.to_numpy(bool)], actual).statistic),
                    4,
                ),
            }
        )
    return rows


def main() -> None:
    original, _ = lens.load_or_build_prepared_history()
    data = add_fixture_history(add_targets(original.reset_index(drop=True)))
    immediate, plan, captain = champion_forecasts(data)
    ceiling_components = captain_variants(data, immediate, captain)
    frozen = ceiling_components["frozen"]
    ceiling = (ceiling_components["ceiling25"] - 0.75 * frozen) / 0.25
    h2h = history_score(data)
    context_predictions = causal_predictions(data, immediate, False)
    history_predictions = causal_predictions(data, immediate, True)
    context_rank = percentile(data, context_predictions.mean(axis=1))
    learned_history_rank = percentile(data, history_predictions.mean(axis=1))
    h2h_rank = percentile(data, h2h)
    evidence = data["h2h_evidence"].to_numpy(float)
    agreement = (
        np.sign(learned_history_rank - frozen)
        == np.sign(h2h_rank - frozen)
    ) & (data["h2h_matches"].to_numpy(float) >= 2)

    variants: dict[str, np.ndarray] = {
        "frozen": frozen,
        "ceiling15": 0.85 * frozen + 0.15 * ceiling,
    }
    for share in (0.025, 0.05, 0.10):
        variants[f"historyDirect{int(share * 1000):03d}"] = (
            (1 - share) * frozen + share * h2h_rank
        )
    for share in (0.05, 0.10, 0.15, 0.20):
        variants[f"contextML{int(share * 100):02d}"] = (
            (1 - share) * frozen + share * context_rank
        )
        variants[f"historyML{int(share * 100):02d}"] = (
            (1 - share) * frozen + share * learned_history_rank
        )
        combined = 0.45 * ceiling + 0.55 * learned_history_rank
        variants[f"historyCeiling{int(share * 100):02d}"] = (
            (1 - share) * frozen + share * combined
        )
    for share in (0.10, 0.15, 0.20):
        variants[f"historyAgree{int(share * 100):02d}"] = frozen + (
            share * agreement * evidence * (learned_history_rank - frozen)
        )

    base_totals, stats = lens.simulate_candidate(
        data,
        immediate,
        STRATEGY,
        plan_scores=plan,
        captain_scores=captain,
        audit_selections=True,
    )
    models, rows = decision_evaluation(
        data, stats, base_totals, frozen, variants
    )
    for row in rows:
        print(
            row["name"],
            row["average"],
            row["seasonDeltas"],
            row["oracleRegretPerWeek"],
            flush=True,
        )
    eligible = [
        row
        for row in rows
        if row["name"] != "frozen"
        and row["developmentDelta"] > 0
        and row["holdoutDelta"] >= 0
        and row["worstSeasonDelta"] >= -8
        and row["declinedSeasons"] <= 2
    ]
    selected = max(
        eligible,
        key=lambda row: (
            row["averageDelta"] - 0.20 * abs(row["worstSeasonDelta"]),
            -row["oracleRegretPerWeek"],
        ),
        default=None,
    )
    result = {
        "status": "chip-interaction finalist" if selected else "research-only; captain stability gate failed",
        "method": "Fixed-XI captain decision evaluation. Player-opponent records are shifted, restricted to single-fixture appearances and shrunk by 8-10 prior matches; learned ranks refit at GW1/GW13/GW25 using only completed rows.",
        "baseline": models["frozen"],
        "correlationAudit": correlation_audit(data, immediate),
        "models": rows,
        "selected": selected["name"] if selected else None,
        "selectionRule": "Positive development, non-negative two-season holdout, worst season at least -8, no more than two declining seasons; maximise downside-adjusted average gain.",
        "productionPromotion": False,
    }
    output = lens.ROOT / "analysis" / "data" / "captain_fixture_history_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "selected": result["selected"],
                "baseline": {
                    key: models["frozen"][key]
                    for key in ("average", "minimum", "captainPointsPerWeek", "oracleRegretPerWeek")
                },
                "eligible": [row["name"] for row in eligible],
            },
            indent=2,
        )
    )
    print(f"Wrote {output.relative_to(lens.ROOT)}")


if __name__ == "__main__":
    main()
