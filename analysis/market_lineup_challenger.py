"""Pre-closing market anchor combined with the causal lineup model.

The historical source exposes two sets of odds from 2019/20 onward.  This
challenger deliberately reads only the first, non-``C`` columns and rejects any
closing column.  Exact archive capture times relative to each historical FPL
deadline are unavailable, so even a strong result remains research-only until
the prospective deadline snapshot pipeline reproduces it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import calibrate_model as lens
from captain_fixture_history_validation import add_fixture_history
from captain_route_consensus_validation import selected_consensus_metric
from forecast_routes import route_components
from frontier_ranker_validation import STRATEGY
from multiscale_horizon_validation import add_targets
from probabilistic_minutes_validation import causal_predictions as minute_predictions
from probabilistic_minutes_validation import season_summary
from wildcard_freehit_ablation import champion_forecasts


ARCHIVE_SEASONS = tuple(
    f"{start:04d}-{(start + 1) % 100:02d}" for start in range(2012, 2026)
)
SOURCE_TEMPLATE = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
MARKET_STRENGTHS = (0.15, 0.30, 0.50, 0.70, 1.00)
MINUTE_DOWNSIDE_STRENGTHS = (0.00, 0.15, 0.30)
TEAM_ALIASES = {
    "man utd": "man united",
    "spurs": "tottenham",
    "sheffield utd": "sheffield united",
}


def season_code(season: str) -> str:
    start, end = season.split("-")
    return start[-2:] + end[-2:]


def normalize_team(name: str) -> str:
    normalized = " ".join(str(name).strip().lower().replace(".", "").split())
    return TEAM_ALIASES.get(normalized, normalized)


def first_available(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    for column in columns:
        if "C" in column and column not in {"B365<2.5", "Avg<2.5", "BbAv<2.5"}:
            raise ValueError(f"Closing market column is forbidden: {column}")
        if column in frame:
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.notna().any():
                return values
    raise KeyError(f"None of the required market columns exist: {columns}")


def no_vig_probabilities(*odds: np.ndarray) -> tuple[np.ndarray, ...]:
    inverse = np.column_stack([1.0 / np.clip(np.asarray(value, float), 1.001, None) for value in odds])
    probability = inverse / inverse.sum(axis=1, keepdims=True)
    return tuple(probability[:, index] for index in range(probability.shape[1]))


def implied_total_goals(over_probability: np.ndarray) -> np.ndarray:
    """Invert P(Poisson(lambda) >= 3) with a stable interpolation grid."""
    grid = np.linspace(0.20, 6.50, 20_000)
    under_three = np.exp(-grid) * (1 + grid + 0.5 * grid**2)
    over = 1 - under_three
    return np.interp(np.clip(over_probability, over[0], over[-1]), over, grid)


def load_market_matches() -> tuple[pd.DataFrame, list[dict]]:
    folder = lens.CACHE / "market-archive"
    folder.mkdir(parents=True, exist_ok=True)
    frames = []
    audit = []
    for order, season in enumerate(ARCHIVE_SEASONS):
        code = season_code(season)
        path = folder / f"{code}-E0.csv"
        if not path.exists():
            urlretrieve(SOURCE_TEMPLATE.format(code=code), path)
        raw = pd.read_csv(path)
        home = first_available(raw, ("AvgH", "BbAvH", "B365H"))
        draw = first_available(raw, ("AvgD", "BbAvD", "B365D"))
        away = first_available(raw, ("AvgA", "BbAvA", "B365A"))
        over = first_available(raw, ("Avg>2.5", "BbAv>2.5", "B365>2.5"))
        under = first_available(raw, ("Avg<2.5", "BbAv<2.5", "B365<2.5"))
        home_p, draw_p, away_p = no_vig_probabilities(home, draw, away)
        over_p, _ = no_vig_probabilities(over, under)
        frame = pd.DataFrame(
            {
                "season": season,
                "marketSeasonOrder": order,
                "homeTeam": raw["HomeTeam"].map(normalize_team),
                "awayTeam": raw["AwayTeam"].map(normalize_team),
                "homeGoals": pd.to_numeric(raw["FTHG"], errors="coerce"),
                "awayGoals": pd.to_numeric(raw["FTAG"], errors="coerce"),
                "homeProbability": home_p,
                "drawProbability": draw_p,
                "awayProbability": away_p,
                "over25Probability": over_p,
                "marketTotalGoals": implied_total_goals(over_p),
            }
        ).dropna()
        frames.append(frame)
        audit.append(
            {
                "season": season,
                "matches": int(len(frame)),
                "source": SOURCE_TEMPLATE.format(code=code),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "oddsTiming": "first non-closing market set only",
            }
        )
    return pd.concat(frames, ignore_index=True), audit


def team_perspective(matches: pd.DataFrame) -> pd.DataFrame:
    common = matches[
        [
            "season",
            "marketSeasonOrder",
            "drawProbability",
            "over25Probability",
            "marketTotalGoals",
        ]
    ]
    home = common.assign(
        team=matches["homeTeam"],
        opponent=matches["awayTeam"],
        wasHome=1.0,
        winProbability=matches["homeProbability"],
        lossProbability=matches["awayProbability"],
        goalsFor=matches["homeGoals"],
        goalsAgainst=matches["awayGoals"],
    )
    away = common.assign(
        team=matches["awayTeam"],
        opponent=matches["homeTeam"],
        wasHome=0.0,
        winProbability=matches["awayProbability"],
        lossProbability=matches["homeProbability"],
        goalsFor=matches["awayGoals"],
        goalsAgainst=matches["homeGoals"],
    )
    return pd.concat([home, away], ignore_index=True)


FEATURES = (
    "winProbability",
    "drawProbability",
    "lossProbability",
    "over25Probability",
    "marketTotalGoals",
    "logWinLoss",
    "wasHome",
    "winOverInteraction",
    "lossOverInteraction",
)


def model_matrix(frame: pd.DataFrame) -> np.ndarray:
    values = frame.copy()
    values["logWinLoss"] = np.log(
        values["winProbability"].clip(lower=0.01)
        / values["lossProbability"].clip(lower=0.01)
    )
    values["winOverInteraction"] = (
        values["winProbability"] * values["over25Probability"]
    )
    values["lossOverInteraction"] = (
        values["lossProbability"] * values["over25Probability"]
    )
    return values[list(FEATURES)].to_numpy(float)


def causal_market_predictions(matches: pd.DataFrame) -> pd.DataFrame:
    sides = team_perspective(matches)
    sides["expectedGoalsFor"] = np.nan
    sides["cleanProbability"] = np.nan
    orders = sides["marketSeasonOrder"].to_numpy(int)
    first_evaluation_order = ARCHIVE_SEASONS.index("2018-19")
    for test_order in range(first_evaluation_order, len(ARCHIVE_SEASONS)):
        train_mask = orders < test_order
        test_mask = orders == test_order
        train = sides.loc[train_mask]
        test = sides.loc[test_mask]
        train_x = model_matrix(train)
        test_x = model_matrix(test)
        age = test_order - train["marketSeasonOrder"].to_numpy(int)
        weight = np.power(0.90, np.maximum(age - 1, 0))
        goal_model = make_pipeline(
            StandardScaler(),
            PoissonRegressor(alpha=0.35, max_iter=1_000),
        )
        clean_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.45, max_iter=1_000),
        )
        goal_model.fit(train_x, train["goalsFor"].to_numpy(float), poissonregressor__sample_weight=weight)
        clean_model.fit(
            train_x,
            train["goalsAgainst"].eq(0).astype(int).to_numpy(),
            logisticregression__sample_weight=weight,
        )
        expected_for = np.clip(goal_model.predict(test_x), 0.20, 4.20)
        logistic_clean = clean_model.predict_proba(test_x)[:, 1]
        # The opponent's independently predicted attack supplies a coherent
        # Poisson clean-sheet prior; logistic calibration absorbs dependence.
        reverse = test.copy()
        reverse[["winProbability", "lossProbability"]] = reverse[
            ["lossProbability", "winProbability"]
        ].to_numpy()
        reverse["wasHome"] = 1.0 - reverse["wasHome"]
        expected_against = np.clip(goal_model.predict(model_matrix(reverse)), 0.20, 4.20)
        clean = np.clip(0.55 * np.exp(-expected_against) + 0.45 * logistic_clean, 0.02, 0.80)
        sides.loc[test_mask, "expectedGoalsFor"] = expected_for
        sides.loc[test_mask, "expectedGoalsAgainst"] = expected_against
        sides.loc[test_mask, "cleanProbability"] = clean
        print(f"Market model predicted {ARCHIVE_SEASONS[test_order]}", flush=True)
    return sides


def attach_market_predictions(
    data: pd.DataFrame, market: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    lookup = {
        (str(row.season), str(row.team), str(row.opponent), bool(row.wasHome)): (
            float(row.expectedGoalsFor),
            float(row.expectedGoalsAgainst),
            float(row.cleanProbability),
        )
        for row in market.dropna(subset=["expectedGoalsFor"]).itertuples()
    }
    output = data.copy()
    keys = zip(
        output["season"].astype(str),
        output["team_name"].fillna("").map(normalize_team),
        output["opponent_name"].fillna("").map(normalize_team),
        output["was_home"].fillna(False).astype(bool),
    )
    values = [lookup.get(key) for key in keys]
    covered = np.asarray([value is not None for value in values], dtype=bool)
    single = output["fixture_count"].eq(1).to_numpy(bool)
    covered &= single
    baseline_for = output["team_expected_goals_for"].to_numpy(float)
    baseline_against = output["team_expected_goals_against"].to_numpy(float)
    baseline_clean = output["team_clean_probability"].to_numpy(float)
    output["market_expected_goals_for"] = [
        value[0] if value is not None and use else fallback
        for value, use, fallback in zip(values, covered, baseline_for)
    ]
    output["market_expected_goals_against"] = [
        value[1] if value is not None and use else fallback
        for value, use, fallback in zip(values, covered, baseline_against)
    ]
    output["market_clean_probability"] = [
        value[2] if value is not None and use else fallback
        for value, use, fallback in zip(values, covered, baseline_clean)
    ]
    output["market_covered"] = covered
    evaluation_single = (
        output["season"].isin(lens.EVALUATION_SEASONS).to_numpy(bool) & single
    )
    unique = output.loc[evaluation_single].drop_duplicates(
        ["season", "GW", "team_id"]
    )
    coverage = float(unique["market_covered"].mean()) if len(unique) else 0.0
    return output, {
        "evaluationSingleTeamWeeks": int(len(unique)),
        "coveredTeamWeeks": int(unique["market_covered"].sum()),
        "coverage": round(coverage, 4),
    }


def team_forecast_metrics(data: pd.DataFrame) -> dict:
    mask = (
        data["season"].isin(lens.EVALUATION_SEASONS)
        & data["fixture_count"].eq(1)
        & data["market_covered"]
    )
    teams = data.loc[mask].drop_duplicates(["season", "GW", "team_id"])

    def mae(column: str, actual: str) -> float:
        return float(np.mean(np.abs(teams[column] - teams[actual])))

    clean_actual = teams["team_clean_sheets"].clip(0, 1).to_numpy(float)
    return {
        "rows": int(len(teams)),
        "goalsForMae": {
            "structural": round(mae("team_expected_goals_for", "team_goals"), 4),
            "market": round(mae("market_expected_goals_for", "team_goals"), 4),
        },
        "goalsAgainstMae": {
            "structural": round(mae("team_expected_goals_against", "team_goals_against"), 4),
            "market": round(mae("market_expected_goals_against", "team_goals_against"), 4),
        },
        "cleanSheetBrier": {
            "structural": round(
                float(np.mean((teams["team_clean_probability"].to_numpy(float) - clean_actual) ** 2)),
                5,
            ),
            "market": round(
                float(np.mean((teams["market_clean_probability"].to_numpy(float) - clean_actual) ** 2)),
                5,
            ),
        },
    }


def adjusted_forecasts(
    data: pd.DataFrame,
    immediate: np.ndarray,
    plan: np.ndarray,
    minute: dict[str, np.ndarray],
    market_strength: float,
    minute_downside_strength: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    routes = route_components(data, immediate)
    covered = data["market_covered"].to_numpy(bool)
    attack_ratio = np.divide(
        data["market_expected_goals_for"].to_numpy(float),
        data["team_expected_goals_for"].to_numpy(float).clip(0.25),
    )
    attack_delta = routes["attack"] * (np.clip(attack_ratio, 0.55, 1.65) - 1)
    clean_delta = (
        0.82
        * (
            data["market_clean_probability"].to_numpy(float)
            - data["team_clean_probability"].to_numpy(float)
        )
        * routes["cleanPoints"]
        * data["sixty_probability"].to_numpy(float)
        * routes["fixture"]
    )
    defence = np.isin(data["position_id"].to_numpy(int), [1, 2])
    conceded_delta = -0.5 * (
        data["market_expected_goals_against"].to_numpy(float)
        - data["team_expected_goals_against"].to_numpy(float)
    ) * (data["expected_minutes"].to_numpy(float) / 90.0) * defence
    market_delta = np.where(
        covered, attack_delta + clean_delta + conceded_delta, 0.0
    )
    score = immediate + market_strength * market_delta
    plan_score = plan + 0.90 * market_strength * market_delta

    old_minutes = data["expected_minutes"].to_numpy(float)
    minute_ratio = np.divide(
        minute["minutes"],
        np.maximum(old_minutes, 12.0),
        out=np.ones_like(old_minutes),
        where=old_minutes > 0,
    )
    downside = np.minimum(np.clip(minute_ratio, 0.45, 1.35) - 1.0, 0.0)
    score *= 1.0 + 0.55 * minute_downside_strength * downside
    blank = data["fixture_count"].to_numpy(int) == 0
    score[blank] = 0.0

    updated = data.copy()
    if minute_downside_strength > 0:
        for column, key in (
            ("play_probability", "play"),
            ("start_probability", "start"),
            ("sixty_probability", "sixty"),
        ):
            old = updated[column].to_numpy(float)
            candidate = np.minimum(minute[key], old)
            updated[column] = (
                old + minute_downside_strength * (candidate - old)
            )
        updated["expected_minutes"] = np.minimum(
            old_minutes,
            old_minutes
            + minute_downside_strength * (minute["minutes"] - old_minutes),
        )
    return updated, score, plan_score


def recursive_summary(
    totals: np.ndarray,
    seasons: list[str],
    baseline: dict,
) -> dict:
    summary = season_summary(totals, seasons)
    deltas = [
        row["points"] - old["points"]
        for row, old in zip(summary["seasons"], baseline["seasons"])
    ]
    development = np.asarray(deltas[:-2], dtype=float)
    holdout = np.asarray(deltas[-2:], dtype=float)
    return {
        **summary,
        "averageDelta": round(float(np.mean(deltas)), 1),
        "developmentDelta": round(float(development.mean()), 1),
        "developmentStability": round(
            float(development.mean() - 0.20 * development.std()), 3
        ),
        "holdoutDelta": round(float(holdout.mean()), 1),
        "worstSeasonDelta": int(min(deltas)),
        "positiveSeasons": int(sum(delta > 0 for delta in deltas)),
        "negativeSeasons": int(sum(delta < 0 for delta in deltas)),
        "seasonDeltas": deltas,
    }


def main() -> None:
    matches, source_audit = load_market_matches()
    market = causal_market_predictions(matches)
    original, _ = lens.load_or_build_prepared_history()
    data = add_fixture_history(add_targets(original.reset_index(drop=True)))
    data, coverage = attach_market_predictions(data, market)
    if coverage["coverage"] < 0.95:
        raise AssertionError(f"Market mapping coverage is only {coverage['coverage']:.1%}")

    immediate, plan, frozen_captain = champion_forecasts(data)
    captain = selected_consensus_metric(data, immediate, frozen_captain)
    minute = minute_predictions(data)
    seasons = list(dict.fromkeys(data["season"].tolist()))
    baseline_totals, _ = lens.simulate_candidate(
        data,
        immediate,
        STRATEGY,
        plan_scores=plan,
        captain_scores=captain,
    )
    baseline = season_summary(baseline_totals, seasons)

    rows = []
    for market_strength in MARKET_STRENGTHS:
        for minute_strength in MINUTE_DOWNSIDE_STRENGTHS:
            print(
                f"Market-lineup replay market={market_strength:.2f} minutes={minute_strength:.2f}",
                flush=True,
            )
            updated, score, plan_score = adjusted_forecasts(
                data,
                immediate.copy(),
                plan.copy(),
                minute,
                market_strength,
                minute_strength,
            )
            totals, _ = lens.simulate_candidate(
                updated,
                score,
                STRATEGY,
                plan_scores=plan_score,
                captain_scores=captain,
            )
            rows.append(
                {
                    "marketStrength": market_strength,
                    "minuteDownsideStrength": minute_strength,
                    **recursive_summary(totals, seasons, baseline),
                }
            )

    # Configuration selection sees development seasons only.  The final two
    # seasons are reported once after this choice.
    selected = max(
        rows,
        key=lambda row: (
            row["developmentStability"],
            row["developmentDelta"],
            -row["minuteDownsideStrength"],
        ),
    )
    selected_data, selected_score, selected_plan = adjusted_forecasts(
        data,
        immediate.copy(),
        plan.copy(),
        minute,
        selected["marketStrength"],
        selected["minuteDownsideStrength"],
    )
    baseline_chip_totals, _ = lens.simulate_candidate(
        data,
        immediate,
        STRATEGY,
        chip_policy=lens.AUDITED_CHAMPION_CHIP_POLICY,
        plan_scores=plan,
        captain_scores=captain,
    )
    selected_chip_totals, _ = lens.simulate_candidate(
        selected_data,
        selected_score,
        STRATEGY,
        chip_policy=lens.AUDITED_CHAMPION_CHIP_POLICY,
        plan_scores=selected_plan,
        captain_scores=captain,
    )
    baseline_chips = season_summary(baseline_chip_totals, seasons)
    selected_chips = recursive_summary(
        selected_chip_totals, seasons, baseline_chips
    )
    promotion_gate = {
        "developmentDeltaAtLeastFive": selected["developmentDelta"] >= 5,
        "holdoutNonNegative": selected["holdoutDelta"] >= 0,
        "worstSeasonAtLeastMinusFifteen": selected["worstSeasonDelta"] >= -15,
        "positiveSeasonsAtLeastFive": selected["positiveSeasons"] >= 5,
        "deadlineTimestampAuditable": False,
    }
    promoted = all(promotion_gate.values())
    result = {
        "status": (
            "promoted"
            if promoted
            else "research-only; historical odds lack exact FPL-deadline capture timestamps"
        ),
        "method": (
            "Prior-season-only Poisson and logistic models transform no-vig first-set "
            "1X2 and over/under prices into team attack, defence and clean-sheet priors. "
            "A downside-only causal minutes model is combined in a small development-only grid."
        ),
        "sourcePolicy": {
            "provider": "Football-Data.co.uk",
            "usedColumns": "Avg/BbAv/B365 first-set columns without C",
            "closingColumnsForbidden": True,
            "timingLimitation": (
                "The archive identifies first-set rather than closing prices, but does not "
                "prove the exact timestamp preceded every historical FPL deadline."
            ),
            "files": source_audit,
        },
        "mapping": coverage,
        "teamForecastMetrics": team_forecast_metrics(data),
        "baselineNoChips": baseline,
        "experiments": rows,
        "selectedByDevelopmentOnly": selected,
        "auditedChipInteraction": {
            "baseline": baseline_chips,
            "challenger": selected_chips,
        },
        "promotionGate": promotion_gate,
        "productionPromotion": promoted,
    }
    output = lens.ROOT / "analysis" / "data" / "market_lineup_challenger.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "mapping": coverage,
                "teamForecastMetrics": result["teamForecastMetrics"],
                "selected": selected,
                "auditedChips": selected_chips,
                "promotionGate": promotion_gate,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
