"""Causal challenger trained where FPL selection mistakes are actually costly.

The model only learns from the selectable frontier in each historical deadline:
high projected players, high-value players, and plausible starters.  Predictions
are quantile-mapped back onto the frozen structural score distribution so this
experiment tests ordering, rather than allowing an uncalibrated tree scale to
drive the transfer engine.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from xgboost import XGBRegressor

import calibrate_model as lens


FEATURES = [
    "component_xpts",
    "role_ridge_xpts",
    "expected_minutes",
    "minutes_std",
    "play_probability",
    "start_probability",
    "sixty_probability",
    "recent_raw",
    "long_raw",
    "goal_rate",
    "assist_rate",
    "bonus_rate",
    "team_expected_goals_for",
    "team_expected_goals_against",
    "team_clean_probability",
    "price",
    "selected",
    "fixture_count",
]
CACHE_VERSION = 3
BLENDS = (0.10, 0.25, 0.40)
PLAYER_CANDIDATE = lens.Candidate(0.32, 0.05, 0.00, 0.13, 0.19, 0.03, 0.18, 0.10, 0.76)
STRATEGY = lens.SimulationStrategy(
    name="Audited stable joint planner",
    transfer_hurdle=16.0,
    bank_limit=5,
    force_weekly_review=False,
    safe_captain=False,
    max_hits=0,
    hit_immediate_hurdle=99.0,
    joint_chip_preflight=True,
    hold_option_value=0.25,
    captain_mode="expected",
    phase_banking=False,
    early_price_weight=0.6,
    joint_squad_optimiser=True,
    squad_captain_weight=0.70,
    squad_bench_weight=0.05,
)
CHIP_POLICY = lens.ChipPolicy(60, 20, 11, 15, 0.55, 10, 28)


@dataclass(frozen=True)
class FrontierMetrics:
    mae: float
    spearman: float
    top15_points: float
    top15_blank_rate: float
    missed_haul_rate: float


def matrix(frame: pd.DataFrame, medians: pd.Series | None = None) -> tuple[np.ndarray, pd.Series]:
    values = frame[FEATURES].replace([np.inf, -np.inf], np.nan).copy()
    values["price"] = values["price"] / 10.0
    values["selected"] = np.log1p(values["selected"].clip(lower=0))
    if medians is None:
        medians = values.median().fillna(0)
    return values.fillna(medians).to_numpy(np.float32), medians


def selectable_frontier(frame: pd.DataFrame) -> np.ndarray:
    work = frame.copy()
    group = work.groupby(["season", "GW", "position_id"], sort=False)
    projection_rank = group["component_xpts"].rank(method="first", ascending=False)
    value = work["component_xpts"] / np.maximum(work["price"] / 10.0, 3.5)
    value_rank = value.groupby([work["season"], work["GW"], work["position_id"]]).rank(
        method="first", ascending=False
    )
    return (
        (projection_rank <= 36)
        | (value_rank <= 24)
        | ((work["play_probability"] >= 0.58) & (projection_rank <= 60))
    ).to_numpy(bool)


def model(seed: int) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=260,
        max_depth=4,
        learning_rate=0.035,
        min_child_weight=10,
        subsample=0.82,
        colsample_bytree=0.82,
        reg_alpha=0.10,
        reg_lambda=2.2,
        objective="reg:pseudohubererror",
        eval_metric="mae",
        n_jobs=-1,
        random_state=seed,
    )


def weights(frame: pd.DataFrame, prediction_order: int) -> np.ndarray:
    age = prediction_order - frame["season_order"].to_numpy(int)
    result = np.power(0.86, np.maximum(age - 1, 0))
    actual = frame["points"].to_numpy(float)
    result *= np.where(actual >= 8, 1.65, np.where(actual >= 5, 1.25, 1.0))
    return result / result.mean()


def causal_predictions(data: pd.DataFrame) -> tuple[np.ndarray, list[dict]]:
    path = lens.CACHE / f"frontier-causal-predictions-v{CACHE_VERSION}.npz"
    structural = data["component_xpts"].to_numpy(float)
    fingerprint = lens.frame_fingerprint(
        data,
        [*FEATURES, "points", "component_xpts"],
        f"frontier-v{CACHE_VERSION}",
    )
    if path.exists():
        cache = np.load(path)
        if (
            len(cache["prediction"]) == len(data)
            and "fingerprint" in cache.files
            and str(cache["fingerprint"].item()) == fingerprint
        ):
            return cache["prediction"], json.loads(str(cache["audit"].item()))
    frontier = selectable_frontier(data)
    prediction = structural.copy()
    audit: list[dict] = []
    seasons = list(dict.fromkeys(data["season"].tolist()))
    orders = data["season_order"].to_numpy(int)
    positions = data["position_id"].to_numpy(int)
    for season_order in range(1, len(seasons)):
        season_mask = orders == season_order
        for position in lens.SQUAD_QUOTAS:
            train_mask = (
                (orders < season_order)
                & (positions == position)
                & frontier
                & (data["fixture_count"].to_numpy(int) > 0)
            )
            test_mask = season_mask & (positions == position)
            train = data.loc[train_mask]
            test = data.loc[test_mask]
            train_x, medians = matrix(train)
            test_x, _ = matrix(test, medians)
            fitted = model(260812 + season_order * 10 + int(position))
            fitted.fit(
                train_x,
                train["points"].clip(-2, 20).to_numpy(float),
                sample_weight=weights(train, season_order),
            )
            prediction[test_mask] = np.clip(fitted.predict(test_x), 0, 14)
            audit.append(
                {
                    "season": seasons[season_order],
                    "position": int(position),
                    "frontierTrainingRows": int(train_mask.sum()),
                    "testRows": int(test_mask.sum()),
                }
            )
            print(f"Frontier predicted {seasons[season_order]} position {position}")
        prediction[season_mask & (data["fixture_count"].to_numpy(int) == 0)] = 0
    np.savez_compressed(
        path,
        prediction=prediction,
        audit=json.dumps(audit),
        fingerprint=fingerprint,
    )
    return prediction, audit


def quantile_map(data: pd.DataFrame, raw: np.ndarray, structural: np.ndarray) -> np.ndarray:
    mapped = structural.copy()
    for _, group_indices in data.groupby(["season", "GW", "position_id"], sort=False).groups.items():
        indices = np.asarray(group_indices, dtype=int)
        raw_order = indices[np.argsort(raw[indices], kind="stable")]
        structural_sorted = np.sort(structural[indices])
        mapped[raw_order] = structural_sorted
    mapped[data["fixture_count"].to_numpy(int) == 0] = 0
    return mapped


def frontier_metrics(data: pd.DataFrame, forecast: np.ndarray) -> FrontierMetrics:
    frontier = selectable_frontier(data)
    observed = frontier & (data["fixture_count"].to_numpy(int) > 0)
    actual = data["points"].to_numpy(float)
    correlations: list[float] = []
    top_points: list[float] = []
    top_blanks: list[float] = []
    missed: list[float] = []
    for _, group_indices in data.groupby(["season", "GW"], sort=False).groups.items():
        indices = np.asarray(group_indices, dtype=int)
        indices = indices[observed[indices]]
        if len(indices) < 20:
            continue
        corr = spearmanr(forecast[indices], actual[indices]).statistic
        if np.isfinite(corr):
            correlations.append(float(corr))
        selected = indices[np.argsort(forecast[indices])[-15:]]
        actual_top = indices[np.argsort(actual[indices])[-15:]]
        top_points.append(float(actual[selected].mean()))
        top_blanks.append(float((actual[selected] <= 2).mean()))
        haul_pool = set(indices[actual[indices] >= 8].tolist())
        missed.append(0 if not haul_pool else len(haul_pool - set(selected.tolist())) / len(haul_pool))
    return FrontierMetrics(
        mae=round(float(np.abs(forecast[observed] - actual[observed]).mean()), 4),
        spearman=round(float(np.mean(correlations)), 4),
        top15_points=round(float(np.mean(top_points)), 4),
        top15_blank_rate=round(float(np.mean(top_blanks)), 4),
        missed_haul_rate=round(float(np.mean(missed)), 4),
    )


def plan_scores(data: pd.DataFrame, immediate: np.ndarray) -> np.ndarray:
    structural = data["component_xpts"].to_numpy(float)
    ratio = np.divide(immediate, structural, out=np.ones_like(immediate), where=structural > 0.20)
    horizon = data["component_horizon_censored"].to_numpy(float) * np.clip(ratio, 0.60, 1.45)
    return 0.75 * immediate * 4.5 + 0.25 * horizon


def fit_current(data: pd.DataFrame, current: list[dict]) -> list[dict]:
    rows = []
    for row in current:
        minute = row["minutesModel"]
        history = row["history"]
        team = row["teamContext"]
        components = row["components"]
        rows.append(
            {
                "component_xpts": row["projected"],
                "role_ridge_xpts": row["ensemble"]["roleProjection"],
                "expected_minutes": row["expectedMinutes"],
                "minutes_std": minute["minutesStd"],
                "play_probability": minute["playProbability"] / 100,
                "start_probability": minute["startProbability"] / 100,
                "sixty_probability": minute["sixtyProbability"] / 100,
                "recent_raw": history["average"],
                "long_raw": history["per90"],
                "goal_rate": components["goals"] / max(row["expectedMinutes"], 10),
                "assist_rate": components["assists"] / max(row["expectedMinutes"], 10),
                "bonus_rate": components["bonus"] / max(row["expectedMinutes"], 10),
                "team_expected_goals_for": team["expectedGoalsFor"],
                "team_expected_goals_against": team["expectedGoalsAgainst"],
                "team_clean_probability": team["cleanSheetProbability"] / 100,
                "price": row["price"] * 10,
                "selected": row["ownership"] * 50_000,
                "fixture_count": 1,
                "position_id": {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}[row["position"]],
            }
        )
    current_frame = pd.DataFrame(rows)
    final_raw = np.zeros(len(current))
    final_audit = []
    frontier = selectable_frontier(data)
    for position in lens.SQUAD_QUOTAS:
        train_mask = (
            (data["position_id"].to_numpy(int) == position)
            & frontier
            & (data["fixture_count"].to_numpy(int) > 0)
        )
        test_mask = current_frame["position_id"].to_numpy(int) == position
        train = data.loc[train_mask]
        train_x, medians = matrix(train)
        test_x, _ = matrix(current_frame.loc[test_mask], medians)
        fitted = model(270000 + int(position))
        fitted.fit(
            train_x,
            train["points"].clip(-2, 20).to_numpy(float),
            sample_weight=weights(train, int(data["season_order"].max()) + 1),
        )
        final_raw[test_mask] = np.clip(fitted.predict(test_x), 0, 14)
        final_audit.append({"position": int(position), "trainingRows": int(train_mask.sum())})
    mapped = np.zeros(len(current))
    percentile = np.zeros(len(current))
    structural = np.asarray([float(row["projected"]) for row in current])
    positions = current_frame["position_id"].to_numpy(int)
    for position in lens.SQUAD_QUOTAS:
        indices = np.flatnonzero(positions == position)
        order = indices[np.argsort(final_raw[indices], kind="stable")]
        mapped[order] = np.sort(structural[indices])
        percentile[order] = np.linspace(0, 100, len(indices), endpoint=True)
    return [
        {
            "id": int(row["id"]),
            "name": row["name"],
            "position": row["position"],
            "rawScore": round(float(final_raw[index]), 3),
            "mappedProjection": round(float(mapped[index]), 2),
            "rankPercentile": round(float(percentile[index]), 1),
            "blend25": round(0.75 * float(row["projected"]) + 0.25 * float(mapped[index]), 2),
        }
        for index, row in enumerate(current)
    ], final_audit


def main() -> None:
    data, _ = lens.load_or_build_prepared_history()
    data = data.reset_index(drop=True)
    seasons = list(dict.fromkeys(data["season"].tolist()))
    benchmark_payload = json.loads(
        (lens.ROOT / "analysis" / "data" / "historical_rank_benchmarks.json").read_text(encoding="utf-8")
    )
    targets = {
        row["season"].replace("/", "-"): int(row["points"])
        for row in benchmark_payload["seasons"]
    }

    def recursive_summary(totals: np.ndarray) -> dict:
        evaluation = [
            {
                "season": seasons[index].replace("-", "/"),
                "points": int(round(float(totals[index]))),
                "target": targets[seasons[index]],
                "margin": int(round(float(totals[index]))) - targets[seasons[index]],
            }
            for index in range(2, len(seasons))
        ]
        return {
            "average": round(float(totals[2:].mean()), 1),
            "minimum": int(round(float(totals[2:].min()))),
            "targetHits": sum(row["margin"] >= 0 for row in evaluation),
            "averageMargin": round(float(np.mean([row["margin"] for row in evaluation])), 1),
            "evaluation": evaluation,
        }
    structural = data["component_xpts"].to_numpy(float)
    raw, training_audit = causal_predictions(data)
    mapped = quantile_map(data, raw, structural)
    forecasts = {"stable": structural}
    forecasts.update({f"frontier{int(blend * 100)}": (1 - blend) * structural + blend * mapped for blend in BLENDS})
    recursive: dict[str, dict] = {}
    totals_by_name: dict[str, np.ndarray] = {}
    stats_by_name: dict[str, list[dict]] = {}
    for name, score in forecasts.items():
        totals, stats = lens.simulate_candidate(data, score, STRATEGY, plan_scores=plan_scores(data, score))
        totals_by_name[name] = totals
        stats_by_name[name] = stats
        recursive[name] = recursive_summary(totals)
        print(name, recursive[name])
    best_name = max(recursive, key=lambda name: recursive[name]["average"])
    best_score = forecasts[best_name]
    fresh = lens.precompute_fresh_squads(data, plan_scores(data, best_score))
    free_hits = lens.precompute_fresh_squads(data, best_score)
    chip_totals, _ = lens.simulate_candidate(
        data,
        best_score,
        STRATEGY,
        chip_policy=CHIP_POLICY,
        fresh_squads=fresh,
        free_hit_squads=free_hits,
        plan_scores=plan_scores(data, best_score),
    )
    recursive[f"{best_name}WithFrozenChips"] = recursive_summary(chip_totals)
    evaluation = data["season_order"].to_numpy(int) >= 2
    result = {
        "schemaVersion": 1,
        "status": "challenger-only",
        "promotionRule": "Promote only after a frozen prospective sample; historical exposure cannot qualify a model.",
        "method": "Position-specific XGBoost trained causally on the deadline-selectable frontier, with structural quantile mapping before recursive policy evaluation.",
        "frontierRows": int(selectable_frontier(data).sum()),
        "metrics": {name: asdict(frontier_metrics(data.loc[evaluation].reset_index(drop=True), values[evaluation])) for name, values in forecasts.items()},
        "recursive": recursive,
        "historicalBest": best_name,
        "trainingAudit": training_audit,
    }
    output = lens.ROOT / "analysis" / "data" / "frontier_ranker_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    current = json.loads((lens.ROOT / "app" / "data" / "current-players.json").read_text(encoding="utf-8"))
    current_scores, final_audit = fit_current(data, current)
    app_result = {
        "schemaVersion": 1,
        "status": "shadow challenger",
        "model": "selectable-frontier XGBoost + structural quantile mapping",
        "historicalBest": best_name,
        "historicalValidation": recursive,
        "frontierMetrics": result["metrics"],
        "promotionRule": result["promotionRule"],
        "trainingAudit": final_audit,
        "players": current_scores,
    }
    (lens.ROOT / "app" / "data" / "frontier-scores.json").write_text(
        json.dumps(app_result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"best": best_name, "recursive": recursive}, indent=2))


if __name__ == "__main__":
    main()
