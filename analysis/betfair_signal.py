"""Does the anytime-scorer market know something the forecast does not?

Run after `betfair_scorers.py` has built its prices. On every single-fixture
player-week with a pre-deadline price, this compares three probabilities that
the player scores:

* the model's, from the same goal route the forecast prices
  (goal rate x opponent vulnerability x expected minutes x team attack);
* the market's, 1 / last traded price before the deadline;
* a logistic blend of the two, fitted on earlier seasons only and scored on the
  next, so the blend's gain is out of sample.

The blend's weight on the market, and its log-loss gain over the model alone,
are the answer. A market coefficient near zero means the forecast already has
what the price knows, and there is nothing to build.

    python analysis/betfair_signal.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import betfair_scorers
import calibrate_model as lens


def model_goal_probability(data: pd.DataFrame) -> pd.Series:
    """P(at least one goal) from the forecast's own attacking goal route."""
    group = ["season", "GW", "position_id"]
    vulnerability = (
        data["opponent_goal_vulnerability"]
        / data.groupby(group)["opponent_goal_vulnerability"].transform("median").clip(lower=0.01)
    ).clip(0.68, 1.42)
    team_attack = (
        data["team_expected_goals_for"] / data["league_goal_rate"].clip(lower=0.9)
    ).pow(0.45).clip(0.70, 1.38)
    expected_goals = data["goal_rate"] * vulnerability * data["expected_minutes"] / 90 * team_attack
    return 1.0 - np.exp(-expected_goals.clip(lower=0.0))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _fit_logistic(x: np.ndarray, y: np.ndarray, steps: int = 200) -> np.ndarray:
    """Newton-Raphson logistic regression with an intercept."""
    design = np.column_stack([np.ones(len(x)), x])
    beta = np.zeros(design.shape[1])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-design @ beta))
        gradient = design.T @ (y - p)
        hessian = design.T @ (design * (p * (1 - p))[:, None]) + 1e-6 * np.eye(len(beta))
        step = np.linalg.solve(hessian, gradient)
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    return beta


def evaluate(data: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    single = data[data["fixture_count"] == 1].copy()
    single["p_model"] = model_goal_probability(single).to_numpy()
    single["scored"] = (single["goals"] > 0).astype(float)
    per_week = prices.groupby(["season", "GW", "element"], as_index=False)["market_score_probability"].mean()
    joined = single.merge(per_week, on=["season", "GW", "element"], how="inner")
    rows = []
    seasons = [season for season in lens.SEASONS if season in set(joined["season"])]
    for index, season in enumerate(seasons):
        test = joined[joined["season"] == season]
        train = joined[joined["season"].isin(seasons[:index])]
        y = test["scored"].to_numpy(float)
        p_model = test["p_model"].to_numpy(float)
        p_market = test["market_score_probability"].to_numpy(float)
        row = {
            "season": season,
            "rows": len(test),
            "scorers": int(y.sum()),
            "loss_model": _log_loss(y, p_model),
            "loss_market": _log_loss(y, p_market),
        }
        if len(train) >= 500:
            x_train = np.column_stack([_logit(train["p_model"].to_numpy(float)), _logit(train["market_score_probability"].to_numpy(float))])
            beta = _fit_logistic(x_train, train["scored"].to_numpy(float))
            model_only = _fit_logistic(x_train[:, :1], train["scored"].to_numpy(float))
            x_test = np.column_stack([_logit(p_model), _logit(p_market)])
            blend = 1 / (1 + np.exp(-(beta[0] + x_test @ beta[1:])))
            recalibrated = 1 / (1 + np.exp(-(model_only[0] + x_test[:, :1] @ model_only[1:])))
            row.update(
                loss_model_recalibrated=_log_loss(y, recalibrated),
                loss_blend=_log_loss(y, blend),
                weight_model=float(beta[1]),
                weight_market=float(beta[2]),
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    data, _ = lens.load_or_build_prepared_history()
    prices = betfair_scorers.build(list(lens.SEASONS))
    if prices.empty:
        raise SystemExit(
            "No Betfair prices found. Download the free BASIC Soccer TO_SCORE files "
            f"into {betfair_scorers.BETFAIR_ROOT} first (see betfair_scorers.py)."
        )
    print(betfair_scorers.coverage_report(prices, list(lens.SEASONS)).round(3).to_string(index=False))
    result = evaluate(data, prices)
    print(result.round(4).to_string(index=False))
    scored = result.dropna(subset=["loss_blend"])
    if not scored.empty:
        weights = scored["rows"] / scored["rows"].sum()
        gain = float(((scored["loss_model_recalibrated"] - scored["loss_blend"]) * weights).sum())
        print(f"out-of-sample log-loss gain from adding the market: {gain:+.4f} per player-week")


if __name__ == "__main__":
    main()
