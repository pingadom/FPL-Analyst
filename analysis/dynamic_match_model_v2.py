"""Causal dynamic team-strength ensemble for FPL forecast routes.

The structural Lens ratings react to team form and opponent strength.  The
market model supplies an independent, information-rich prior.  This module
combines the two with weights selected strictly from earlier seasons and keeps
the resulting probabilities separate from player-selection policy.

Historical market files have unknown capture times.  They are therefore valid
for challenger research but not, by themselves, for production promotion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

import calibrate_model as lens
from market_lineup_challenger import (
    attach_market_predictions,
    causal_market_predictions,
    load_market_matches,
)


WEIGHT_GRID = np.linspace(0.0, 1.0, 21)
DEFAULT_MARKET_WEIGHT = 0.50
SHRINK_MATCHES = 380.0


@dataclass(frozen=True)
class BlendWeights:
    attack: float
    defence: float
    clean: float
    prior_team_matches: int


def logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, float), 0.01, 0.99)
    return np.log(clipped / (1.0 - clipped))


def logistic(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(values, float)))


def geometric_blend(
    structural: np.ndarray, market: np.ndarray, weight: float
) -> np.ndarray:
    structural = np.clip(np.asarray(structural, float), 0.05, 6.0)
    market = np.clip(np.asarray(market, float), 0.05, 6.0)
    return np.exp((1.0 - weight) * np.log(structural) + weight * np.log(market))


def probability_blend(
    structural: np.ndarray, market: np.ndarray, weight: float
) -> np.ndarray:
    return np.clip(
        logistic((1.0 - weight) * logit(structural) + weight * logit(market)),
        0.01,
        0.90,
    )


def poisson_deviance(actual: np.ndarray, prediction: np.ndarray) -> float:
    actual = np.asarray(actual, float)
    prediction = np.clip(np.asarray(prediction, float), 0.02, None)
    term = np.where(
        actual > 0,
        actual * np.log(np.clip(actual, 1e-12, None) / prediction),
        0.0,
    )
    return float(np.mean(2.0 * (term - (actual - prediction))))


def _selected_weight(
    structural: np.ndarray,
    market: np.ndarray,
    actual: np.ndarray,
    probability: bool = False,
) -> float:
    if len(actual) == 0:
        return DEFAULT_MARKET_WEIGHT
    loss = []
    for weight in WEIGHT_GRID:
        prediction = (
            probability_blend(structural, market, float(weight))
            if probability
            else geometric_blend(structural, market, float(weight))
        )
        score = (
            float(np.mean((prediction - actual) ** 2))
            if probability
            else poisson_deviance(actual, prediction)
        )
        loss.append(score)
    empirical = float(WEIGHT_GRID[int(np.argmin(loss))])
    # Early estimates are deliberately pulled to a neutral predeclared prior.
    reliability = len(actual) / (len(actual) + SHRINK_MATCHES)
    return float(reliability * empirical + (1.0 - reliability) * DEFAULT_MARKET_WEIGHT)


def _unique_team_weeks(data: pd.DataFrame) -> pd.DataFrame:
    mask = data["fixture_count"].eq(1) & data["market_covered"]
    return data.loc[mask].drop_duplicates(["season", "GW", "team_id"]).copy()


def attach_dynamic_predictions(data: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Attach walk-forward dynamic forecasts to a market-enriched player frame."""
    output = data.copy()
    output["dynamic_expected_goals_for"] = output["team_expected_goals_for"]
    output["dynamic_expected_goals_against"] = output["team_expected_goals_against"]
    output["dynamic_clean_probability"] = output["team_clean_probability"]
    teams = _unique_team_weeks(output)
    audits: list[dict] = []
    for season_order in sorted(teams["season_order"].unique()):
        test = teams["season_order"].eq(season_order)
        train = teams["season_order"].lt(season_order)
        prior = teams.loc[train]
        weights = BlendWeights(
            attack=_selected_weight(
                prior["team_expected_goals_for"].to_numpy(float),
                prior["market_expected_goals_for"].to_numpy(float),
                prior["team_goals"].to_numpy(float),
            ),
            defence=_selected_weight(
                prior["team_expected_goals_against"].to_numpy(float),
                prior["market_expected_goals_against"].to_numpy(float),
                prior["team_goals_against"].to_numpy(float),
            ),
            clean=_selected_weight(
                prior["team_clean_probability"].to_numpy(float),
                prior["market_clean_probability"].to_numpy(float),
                prior["team_clean_sheets"].clip(0, 1).to_numpy(float),
                probability=True,
            ),
            prior_team_matches=int(train.sum()),
        )
        player_test = output["season_order"].eq(season_order) & output["market_covered"]
        output.loc[player_test, "dynamic_expected_goals_for"] = geometric_blend(
            output.loc[player_test, "team_expected_goals_for"].to_numpy(float),
            output.loc[player_test, "market_expected_goals_for"].to_numpy(float),
            weights.attack,
        )
        output.loc[player_test, "dynamic_expected_goals_against"] = geometric_blend(
            output.loc[player_test, "team_expected_goals_against"].to_numpy(float),
            output.loc[player_test, "market_expected_goals_against"].to_numpy(float),
            weights.defence,
        )
        output.loc[player_test, "dynamic_clean_probability"] = probability_blend(
            output.loc[player_test, "team_clean_probability"].to_numpy(float),
            output.loc[player_test, "market_clean_probability"].to_numpy(float),
            weights.clean,
        )
        season_name = str(teams.loc[test, "season"].iloc[0])
        audits.append(
            {
                "season": season_name,
                "priorTeamMatches": weights.prior_team_matches,
                "attackMarketWeight": round(weights.attack, 4),
                "defenceMarketWeight": round(weights.defence, 4),
                "cleanMarketWeight": round(weights.clean, 4),
                "testTeamMatches": int(test.sum()),
            }
        )
    return output, audits


def forecast_metrics(data: pd.DataFrame) -> dict:
    teams = _unique_team_weeks(data)
    actual_clean = teams["team_clean_sheets"].clip(0, 1).to_numpy(float)

    def goal_metrics(prefix: str) -> dict:
        forecast_for = teams[f"{prefix}_expected_goals_for"].to_numpy(float)
        forecast_against = teams[f"{prefix}_expected_goals_against"].to_numpy(float)
        return {
            "goalsForMae": round(
                float(np.mean(np.abs(forecast_for - teams["team_goals"].to_numpy(float)))), 4
            ),
            "goalsForPoissonDeviance": round(
                poisson_deviance(teams["team_goals"], forecast_for), 4
            ),
            "goalsAgainstMae": round(
                float(
                    np.mean(
                        np.abs(
                            forecast_against
                            - teams["team_goals_against"].to_numpy(float)
                        )
                    )
                ),
                4,
            ),
            "cleanSheetBrier": round(
                float(
                    np.mean(
                        (teams[f"{prefix}_clean_probability"].to_numpy(float) - actual_clean)
                        ** 2
                    )
                ),
                5,
            ),
        }

    renamed = data.copy()
    renamed["structural_expected_goals_for"] = renamed["team_expected_goals_for"]
    renamed["structural_expected_goals_against"] = renamed["team_expected_goals_against"]
    renamed["structural_clean_probability"] = renamed["team_clean_probability"]
    renamed["market_expected_goals_for"] = renamed["market_expected_goals_for"]
    renamed["market_expected_goals_against"] = renamed["market_expected_goals_against"]
    renamed["market_clean_probability"] = renamed["market_clean_probability"]
    # Rebind the closure after adding the structural aliases.
    teams = _unique_team_weeks(renamed)
    return {
        "teamMatches": int(len(teams)),
        "structural": goal_metrics("structural"),
        "market": goal_metrics("market"),
        "dynamic": goal_metrics("dynamic"),
    }


def build_dynamic_history() -> tuple[pd.DataFrame, dict]:
    matches, source_audit = load_market_matches()
    market = causal_market_predictions(matches)
    original, _ = lens.load_or_build_prepared_history()
    enriched, coverage = attach_market_predictions(original.reset_index(drop=True), market)
    dynamic, weight_audit = attach_dynamic_predictions(enriched)
    result = {
        "status": "research-only causal dynamic match ensemble",
        "provenance": (
            "Structural inputs are deadline-causal. Historical first-market odds "
            "have unknown capture times and cannot qualify production promotion."
        ),
        "coverage": coverage,
        "metrics": forecast_metrics(dynamic),
        "weights": weight_audit,
        "marketSources": source_audit,
    }
    return dynamic, result


def main() -> None:
    _, result = build_dynamic_history()
    output = lens.ROOT / "analysis" / "data" / "dynamic_match_model_v2.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"coverage": result["coverage"], "metrics": result["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
