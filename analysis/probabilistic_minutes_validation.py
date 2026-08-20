"""Causal probabilistic minutes challenger and recursive policy validation.

The challenger predicts four related quantities at each historical deadline:
appearance, start, 60-minute and expected-minute rates.  Every season is scored
by models trained only on earlier seasons.  Short-vs-long lineup features let
the trees react to role changes without reading the target Gameweek.
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

import calibrate_model as lens
from frontier_ranker_validation import STRATEGY
from multiscale_horizon_validation import add_targets
from wildcard_freehit_ablation import champion_forecasts


CACHE_VERSION = 1
PROBABILITY_TARGETS = {
    "play": "appearance_rate_target",
    "start": "start_rate_target",
    "sixty": "sixty_rate_target",
}
FEATURES = [
    "position_id",
    "GW",
    "price",
    "log_selected",
    "log_transfer_balance",
    "age",
    "observations",
    "past_minutes",
    "start_probability",
    "play_probability",
    "sixty_probability",
    "expected_minutes",
    "minutes_std",
    "minutes_model_confidence",
    "competition_pressure",
    "rotation_volatility",
    "team_rotation_rate",
    "team_rest_days",
    "fixture_count",
    "price_rise_probability",
    "price_fall_probability",
    "recent_minutes_2",
    "recent_minutes_3",
    "recent_minutes_6",
    "recent_start_2",
    "recent_start_3",
    "recent_start_6",
    "recent_play_2",
    "recent_play_3",
    "recent_play_6",
    "recent_sixty_3",
    "recent_sixty_6",
    "minutes_change_2v6",
    "start_change_2v6",
    "play_change_2v6",
    "recent_points_3",
    "recent_ict_3",
]


def feature_frame(data: pd.DataFrame) -> pd.DataFrame:
    work = data.copy()
    fixture = work["fixture_count"].clip(lower=1)
    work["appearance_rate_target"] = work["appearances_observed"] / fixture
    work["start_rate_target"] = work["starts_observed"] / fixture
    work["sixty_rate_target"] = work["sixty_observed"] / fixture
    work["minute_rate_target"] = (work["minutes"] / fixture).clip(0, 90)
    work["log_selected"] = np.log1p(work["selected"].clip(lower=0))
    work["log_transfer_balance"] = np.sign(work["transfers_balance"]) * np.log1p(
        work["transfers_balance"].abs()
    )
    ordered = work.sort_values(["player_key", "season_order", "GW"], kind="stable")
    player = ordered.groupby("player_key", sort=False)
    minute_observed = (ordered["minutes"] / ordered["fixture_count"].clip(lower=1)).where(
        ordered["fixture_count"] > 0
    )
    start_observed = (ordered["starts_observed"] / ordered["fixture_count"].clip(lower=1)).where(
        ordered["fixture_count"] > 0
    )
    play_observed = (ordered["appearances_observed"] / ordered["fixture_count"].clip(lower=1)).where(
        ordered["fixture_count"] > 0
    )
    sixty_observed = (ordered["sixty_observed"] / ordered["fixture_count"].clip(lower=1)).where(
        ordered["fixture_count"] > 0
    )
    temporary = {
        "_minute": minute_observed,
        "_start": start_observed,
        "_play": play_observed,
        "_sixty": sixty_observed,
        "_points": ordered["points"].where(ordered["fixture_count"] > 0),
        "_ict": ordered["ict"].where(ordered["fixture_count"] > 0),
    }
    for name, values in temporary.items():
        ordered[name] = values
    player = ordered.groupby("player_key", sort=False)
    for window in [2, 3, 6]:
        ordered[f"recent_minutes_{window}"] = player["_minute"].transform(
            lambda values, n=window: values.rolling(n, min_periods=1).mean().shift(1)
        )
        ordered[f"recent_start_{window}"] = player["_start"].transform(
            lambda values, n=window: values.rolling(n, min_periods=1).mean().shift(1)
        )
        ordered[f"recent_play_{window}"] = player["_play"].transform(
            lambda values, n=window: values.rolling(n, min_periods=1).mean().shift(1)
        )
    for window in [3, 6]:
        ordered[f"recent_sixty_{window}"] = player["_sixty"].transform(
            lambda values, n=window: values.rolling(n, min_periods=1).mean().shift(1)
        )
    ordered["recent_points_3"] = player["_points"].transform(
        lambda values: values.rolling(3, min_periods=1).mean().shift(1)
    )
    ordered["recent_ict_3"] = player["_ict"].transform(
        lambda values: values.rolling(3, min_periods=1).mean().shift(1)
    )
    ordered["minutes_change_2v6"] = ordered["recent_minutes_2"] - ordered["recent_minutes_6"]
    ordered["start_change_2v6"] = ordered["recent_start_2"] - ordered["recent_start_6"]
    ordered["play_change_2v6"] = ordered["recent_play_2"] - ordered["recent_play_6"]
    ordered.drop(columns=list(temporary), inplace=True)
    return ordered.sort_index()


def matrix(frame: pd.DataFrame, medians: pd.Series | None = None) -> tuple[np.ndarray, pd.Series]:
    complete = frame.copy()
    for feature in FEATURES:
        if feature not in complete:
            complete[feature] = np.nan
    values = complete[FEATURES].replace([np.inf, -np.inf], np.nan).copy()
    values["position_id"] = values["position_id"].astype(float)
    values["price"] = values["price"] / 10.0
    values["GW"] = values["GW"] / 38.0
    values["age"] = values["age"] / 35.0
    values["observations"] = np.log1p(values["observations"].clip(lower=0))
    if medians is None:
        medians = values.median().fillna(0)
    return values.fillna(medians).to_numpy(np.float32), medians


def model(seed: int, minutes: bool = False) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=180,
        max_depth=4,
        learning_rate=0.045,
        min_child_weight=18,
        subsample=0.78,
        colsample_bytree=0.78,
        reg_alpha=0.20,
        reg_lambda=3.0,
        objective="reg:pseudohubererror" if minutes else "reg:squarederror",
        eval_metric="mae" if minutes else "rmse",
        tree_method="hist",
        n_jobs=-1,
        random_state=seed,
    )


def causal_predictions(data: pd.DataFrame) -> dict[str, np.ndarray]:
    cache_path = lens.CACHE / f"probabilistic-minutes-v{CACHE_VERSION}.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        if len(cached["play"]) == len(data):
            return {name: cached[name] for name in ["play", "start", "sixty", "minutes"]}
    work = feature_frame(data)
    output = {
        "play": work["play_probability"].to_numpy(float).copy(),
        "start": work["start_probability"].to_numpy(float).copy(),
        "sixty": work["sixty_probability"].to_numpy(float).copy(),
        "minutes": work["expected_minutes"].to_numpy(float).copy(),
    }
    orders = work["season_order"].to_numpy(int)
    observed = work["fixture_count"].to_numpy(int) > 0
    seasons = list(dict.fromkeys(work["season"].tolist()))
    for season_order in range(1, len(seasons)):
        train_mask = (orders < season_order) & observed
        test_mask = orders == season_order
        train = work.loc[train_mask]
        test = work.loc[test_mask]
        train_x, medians = matrix(train)
        test_x, _ = matrix(test, medians)
        age = season_order - train["season_order"].to_numpy(int)
        sample_weight = np.power(0.84, np.maximum(age - 1, 0))
        sample_weight *= np.where(train["observations"].to_numpy(float) >= 3, 1.15, 0.80)
        sample_weight /= sample_weight.mean()
        for offset, (name, target) in enumerate(
            [*PROBABILITY_TARGETS.items(), ("minutes", "minute_rate_target")]
        ):
            fitted = model(260814 + 10 * season_order + offset, minutes=name == "minutes")
            fitted.fit(train_x, train[target].to_numpy(float), sample_weight=sample_weight)
            prediction = fitted.predict(test_x)
            if name == "minutes":
                prediction = np.clip(prediction, 0, 90)
            else:
                prediction = np.clip(prediction, 0.005, 0.995)
            output[name][test_mask] = prediction
        print(f"Minutes challenger predicted {seasons[season_order]}")
    # Enforce probability coherence.  P(60) <= P(start) <= P(play).
    output["start"] = np.minimum(output["start"], output["play"])
    output["sixty"] = np.minimum(output["sixty"], output["start"])
    output["minutes"] = np.minimum(output["minutes"], 90 * output["play"] + 8 * (1 - output["play"]))
    np.savez_compressed(cache_path, **output)
    return output


def terminal_live_predictions(
    historical: pd.DataFrame,
    live: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Fit the frozen causal minutes challenger through the latest completed data.

    This is the prospective counterpart of ``causal_predictions``. Historical
    targets are completed fixtures only; the current live rows supply inputs and
    never supply outcomes.
    """
    train = feature_frame(historical)
    observed = train["fixture_count"].to_numpy(int) > 0
    fitted = train.loc[observed]
    current = live.copy()
    current["log_selected"] = np.log1p(current.get("selected", 0).clip(lower=0))
    transfer_balance = current.get(
        "transfers_balance", pd.Series(0.0, index=current.index)
    )
    current["log_transfer_balance"] = np.sign(transfer_balance) * np.log1p(
        np.abs(transfer_balance)
    )
    train_x, medians = matrix(fitted)
    live_x, _ = matrix(current, medians)
    output: dict[str, np.ndarray] = {}
    terminal_order = int(train["season_order"].max()) + 1
    sample_age = terminal_order - fitted["season_order"].to_numpy(int)
    sample_weight = np.power(0.84, np.maximum(sample_age - 1, 0))
    sample_weight *= np.where(
        fitted["observations"].to_numpy(float) >= 3, 1.15, 0.80
    )
    sample_weight /= sample_weight.mean()
    for offset, (name, target) in enumerate(
        [*PROBABILITY_TARGETS.items(), ("minutes", "minute_rate_target")]
    ):
        estimator = model(260814 + 10 * terminal_order + offset, minutes=name == "minutes")
        estimator.fit(
            train_x,
            fitted[target].to_numpy(float),
            sample_weight=sample_weight,
        )
        prediction = estimator.predict(live_x)
        output[name] = np.clip(
            prediction,
            0 if name == "minutes" else 0.005,
            90 if name == "minutes" else 0.995,
        )
    output["start"] = np.minimum(output["start"], output["play"])
    output["sixty"] = np.minimum(output["sixty"], output["start"])
    output["minutes"] = np.minimum(
        output["minutes"], 90 * output["play"] + 8 * (1 - output["play"])
    )
    return output


def exposure_weights(data: pd.DataFrame, scores: np.ndarray, plan: np.ndarray, captain: np.ndarray) -> np.ndarray:
    _, stats = lens.simulate_candidate(
        data, scores, STRATEGY, plan_scores=plan, captain_scores=captain, audit_selections=True
    )
    weights = np.zeros(len(data), dtype=float)
    rank = pd.Series(scores).groupby([data["season"], data["GW"], data["position_id"]]).rank(
        method="first", ascending=False
    )
    weights[rank.to_numpy(float) <= 20] += 0.20
    context = lens.simulation_context(data)
    for season_index, season_context in enumerate(context["seasons"]):
        if season_context["season"] not in lens.EVALUATION_SEASONS:
            continue
        selection_by_gw = {int(row["gw"]): row for row in stats[season_index]["selectionLog"]}
        transfer_by_gw = {int(row["gw"]): row for row in stats[season_index]["transferLog"]}
        for gw in season_context["weeks"]:
            indices = np.asarray(season_context["weekIndices"][gw], dtype=int)
            elements = data.loc[indices, "element"].to_numpy(int)
            selection = selection_by_gw[int(gw)]
            weights[indices] += 0.5 * np.isin(elements, selection["squad"])
            weights[indices] += 1.0 * np.isin(elements, selection["xi"])
            weights[indices] += 1.5 * (elements == int(selection["captain"]))
            transfer = transfer_by_gw.get(int(gw), {})
            transfer_elements = transfer.get("inElements", []) + transfer.get("outElements", [])
            weights[indices] += 1.25 * np.isin(elements, transfer_elements)
    return weights


def probability_metrics(actual: np.ndarray, predicted: np.ndarray, weight: np.ndarray) -> dict:
    mask = weight > 0
    local_weight = weight[mask]
    error = predicted[mask] - actual[mask]
    return {
        "brier": round(float(np.average(error**2, weights=local_weight)), 5),
        "mae": round(float(np.average(np.abs(error), weights=local_weight)), 5),
        "bias": round(float(np.average(error, weights=local_weight)), 5),
    }


def adjusted_forecasts(
    data: pd.DataFrame,
    immediate: np.ndarray,
    plan: np.ndarray,
    captain: np.ndarray,
    prediction: dict[str, np.ndarray],
    strength: float,
    mode: str = "symmetric",
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    old_play = data["play_probability"].to_numpy(float)
    old_sixty = data["sixty_probability"].to_numpy(float)
    old_minutes = data["expected_minutes"].to_numpy(float)
    fixture = data["fixture_count"].clip(lower=1).to_numpy(float)
    new_play = (1 - strength) * old_play + strength * prediction["play"]
    new_start = (1 - strength) * data["start_probability"].to_numpy(float) + strength * prediction["start"]
    new_sixty = (1 - strength) * old_sixty + strength * prediction["sixty"]
    new_minutes = (1 - strength) * old_minutes + strength * prediction["minutes"]
    old_appearance = (old_play + old_sixty) * fixture
    new_appearance = (new_play + new_sixty) * fixture
    nonappearance = np.maximum(immediate - old_appearance, 0)
    minute_ratio = np.clip(new_minutes / np.maximum(old_minutes, 12), 0.35, 1.65)
    full_immediate = new_appearance + nonappearance * minute_ratio
    # The champion is an ensemble, so a half-strength structural correction is
    # safer than pretending every point route scales perfectly with minutes.
    correction = 0.55 * (full_immediate - immediate)
    if mode in {"downside", "lineup-downside"}:
        correction = np.minimum(correction, 0)
    new_immediate = immediate + correction
    relative = np.clip(new_immediate / np.maximum(immediate, 0.75), 0.65, 1.35)
    new_plan = (
        plan.copy()
        if mode.startswith("lineup-")
        else plan * (1 + 0.45 * (relative - 1))
    )
    new_captain = captain + 0.75 * (new_immediate - immediate)
    updated = data.copy()
    updated["play_probability"] = new_play
    updated["start_probability"] = np.minimum(new_start, new_play)
    updated["sixty_probability"] = np.minimum(new_sixty, updated["start_probability"])
    updated["expected_minutes"] = new_minutes
    updated["minutes_security_raw"] = 0.65 * updated["sixty_probability"] + 0.35 * updated["play_probability"]
    updated["minutes_security"] = updated["minutes_security_raw"].groupby(
        [updated["season"], updated["GW"], updated["position_id"]]
    ).rank(pct=True)
    structural_blank = updated["fixture_count"].eq(0).to_numpy(bool)
    new_immediate[structural_blank] = 0
    new_plan[structural_blank] = 0
    new_captain[structural_blank] = 0
    return updated, new_immediate, new_plan, new_captain


def season_summary(totals: np.ndarray, seasons: list[str]) -> dict:
    rows = [
        {"season": season.replace("-", "/"), "points": int(round(float(totals[index])))}
        for index, season in enumerate(seasons)
        if season in lens.EVALUATION_SEASONS
    ]
    development = rows[:-2]
    holdout = rows[-2:]
    return {
        "average": round(float(np.mean([row["points"] for row in rows])), 1),
        "minimum": min(row["points"] for row in rows),
        "developmentAverage": round(float(np.mean([row["points"] for row in development])), 1),
        "holdoutAverage": round(float(np.mean([row["points"] for row in holdout])), 1),
        "seasons": rows,
    }


def main() -> None:
    original, _ = lens.load_or_build_prepared_history()
    data = add_targets(original.reset_index(drop=True))
    immediate, plan, captain = champion_forecasts(data)
    prediction = causal_predictions(data)
    weights = exposure_weights(data, immediate, plan, captain)
    valid = data["season"].isin(lens.EVALUATION_SEASONS).to_numpy(bool) & data["fixture_count"].gt(0).to_numpy(bool)
    weights = weights * valid
    fixture = data["fixture_count"].clip(lower=1).to_numpy(float)
    actual = {
        "play": data["appearances_observed"].to_numpy(float) / fixture,
        "start": data["starts_observed"].to_numpy(float) / fixture,
        "sixty": data["sixty_observed"].to_numpy(float) / fixture,
        "minutes": data["minutes"].to_numpy(float) / fixture / 90.0,
    }
    baseline = {
        "play": data["play_probability"].to_numpy(float),
        "start": data["start_probability"].to_numpy(float),
        "sixty": data["sixty_probability"].to_numpy(float),
        "minutes": data["expected_minutes"].to_numpy(float) / 90.0,
    }
    candidate = {**prediction, "minutes": prediction["minutes"] / 90.0}
    metric_rows = {
        name: {
            "baseline": probability_metrics(actual[name], baseline[name], weights),
            "challenger": probability_metrics(actual[name], candidate[name], weights),
        }
        for name in actual
    }
    seasons = list(dict.fromkeys(data["season"].tolist()))
    base_totals, _ = lens.simulate_candidate(
        data, immediate, STRATEGY, plan_scores=plan, captain_scores=captain
    )
    base = season_summary(base_totals, seasons)
    variants = []
    applications = [
        ("symmetric", 0.20), ("symmetric", 0.35), ("symmetric", 0.50),
        ("symmetric", 0.75), ("symmetric", 1.00),
        ("lineup-symmetric", 0.20), ("lineup-symmetric", 0.50),
        ("lineup-downside", 0.25), ("lineup-downside", 0.50),
        ("lineup-downside", 0.75), ("lineup-downside", 1.00),
    ]
    for mode, strength in applications:
        updated, score, plan_score, captain_score = adjusted_forecasts(
            data, immediate.copy(), plan.copy(), captain.copy(), prediction, strength, mode
        )
        totals, _ = lens.simulate_candidate(
            updated, score, STRATEGY, plan_scores=plan_score, captain_scores=captain_score
        )
        summary = season_summary(totals, seasons)
        deltas = [
            row["points"] - base_row["points"]
            for row, base_row in zip(summary["seasons"], base["seasons"])
        ]
        variants.append(
            {
                "name": f"{mode}-{strength:.2f}",
                "mode": mode,
                "strength": strength,
                **summary,
                "averageDelta": round(summary["average"] - base["average"], 1),
                "developmentDelta": round(summary["developmentAverage"] - base["developmentAverage"], 1),
                "holdoutDelta": round(summary["holdoutAverage"] - base["holdoutAverage"], 1),
                "improvedSeasons": int(sum(delta > 0 for delta in deltas)),
                "worseSeasons": int(sum(delta < 0 for delta in deltas)),
                "worstSeasonDelta": int(min(deltas)),
            }
        )
        print("minutes blend", strength, variants[-1]["average"], deltas)
    eligible = [
        row for row in variants
        if row["developmentDelta"] > 0
        and row["holdoutDelta"] >= 5
        and row["worstSeasonDelta"] >= 0
        and row["improvedSeasons"] >= 5
    ]
    selected = max(eligible, key=lambda row: (row["holdoutDelta"], row["developmentDelta"])) if eligible else None
    result = {
        "status": "promoted" if selected else "research-only; robust promotion gate failed",
        "method": (
            "Season-ahead causal gradient trees trained only on earlier seasons. "
            "Short/long lineup features are shifted before the target GW. Recursive policy variants "
            "are compared with the frozen champion; the last two seasons are untouched holdout."
        ),
        "features": FEATURES,
        "decisionWeightedMetrics": metric_rows,
        "baseline": base,
        "variants": variants,
        "promotionGate": {
            "developmentDelta": "> 0",
            "holdoutDelta": ">= 5 points/season",
            "worstSeasonDelta": ">= 0",
            "improvedSeasons": ">= 5 of 8",
        },
        "selected": selected,
    }
    output = lens.ROOT / "analysis" / "data" / "probabilistic_minutes_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": metric_rows, "baseline": base, "selected": selected}, indent=2))


if __name__ == "__main__":
    main()
