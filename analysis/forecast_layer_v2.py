"""Player-level forecast corrections built on causal match and minutes models.

This layer is intentionally policy-free.  It emits alternative scores and
uncertainty diagnostics; the recursive tournament decides which surfaces, if
any, are allowed to influence captains, line-ups, transfers or chips.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecast_routes import route_components


FORBIDDEN_INPUTS = frozenset(
    {
        "points",
        "minutes",
        "goals_scored",
        "assists",
        "clean_sheets",
        "bonus",
        "bps",
        "expected_goals",
        "expected_assists",
        "expected_goals_conceded",
    }
)


@dataclass(frozen=True)
class MinutesMixture:
    no_show: np.ndarray
    cameo: np.ndarray
    start_under_sixty: np.ndarray
    sixty_plus: np.ndarray
    expected_minutes: np.ndarray
    entropy: np.ndarray


def minutes_mixture(prediction: dict[str, np.ndarray]) -> MinutesMixture:
    play = np.clip(np.asarray(prediction["play"], float), 0, 1)
    start = np.minimum(np.clip(np.asarray(prediction["start"], float), 0, 1), play)
    sixty = np.minimum(np.clip(np.asarray(prediction["sixty"], float), 0, 1), start)
    no_show = 1.0 - play
    cameo = play - start
    start_under = start - sixty
    probabilities = np.column_stack([no_show, cameo, start_under, sixty])
    safe_probabilities = np.clip(probabilities, 1e-12, 1.0)
    entropy = -np.sum(probabilities * np.log(safe_probabilities), axis=1) / np.log(4.0)
    return MinutesMixture(
        no_show=no_show,
        cameo=cameo,
        start_under_sixty=start_under,
        sixty_plus=sixty,
        expected_minutes=np.clip(np.asarray(prediction["minutes"], float), 0, 90),
        entropy=np.clip(entropy, 0, 1),
    )


def dynamic_route_score(
    data: pd.DataFrame,
    immediate: np.ndarray,
    strength: float = 1.0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Translate match probabilities only through identifiable scoring routes."""
    routes = route_components(data, immediate)
    structural_for = data["team_expected_goals_for"].to_numpy(float)
    dynamic_for = data["dynamic_expected_goals_for"].to_numpy(float)
    attack_ratio = np.clip(
        np.divide(dynamic_for, np.maximum(structural_for, 0.25)), 0.60, 1.55
    )
    attack_delta = routes["attack"] * (attack_ratio - 1.0)

    clean_delta = (
        0.82
        * (
            data["dynamic_clean_probability"].to_numpy(float)
            - data["team_clean_probability"].to_numpy(float)
        )
        * routes["cleanPoints"]
        * data["sixty_probability"].to_numpy(float)
        * routes["fixture"]
    )
    defence = np.isin(data["position_id"].to_numpy(int), [1, 2])
    conceded_delta = (
        -0.5
        * (
            data["dynamic_expected_goals_against"].to_numpy(float)
            - data["team_expected_goals_against"].to_numpy(float)
        )
        * (data["expected_minutes"].to_numpy(float) / 90.0)
        * defence
    )
    covered = data["market_covered"].to_numpy(bool)
    delta = np.where(covered, attack_delta + clean_delta + conceded_delta, 0.0)
    score = np.asarray(immediate, float) + strength * delta
    score[data["fixture_count"].to_numpy(int) == 0] = 0.0
    return score, {
        "attackDelta": attack_delta,
        "cleanDelta": clean_delta,
        "concededDelta": conceded_delta,
        "totalDelta": delta,
    }


def captain_availability_score(
    data: pd.DataFrame,
    score: np.ndarray,
    prediction: dict[str, np.ndarray],
    downside_strength: float,
) -> tuple[np.ndarray, MinutesMixture]:
    """Penalise only new downside evidence; never manufacture an upside boost."""
    mixture = minutes_mixture(prediction)
    old_play = np.clip(data["play_probability"].to_numpy(float), 0, 1)
    old_sixty = np.clip(data["sixty_probability"].to_numpy(float), 0, 1)
    new_reliability = 0.45 * (1.0 - mixture.no_show) + 0.55 * mixture.sixty_plus
    old_reliability = 0.45 * old_play + 0.55 * old_sixty
    downside = np.minimum(new_reliability - old_reliability, 0.0)
    # Entropy raises the penalty only when the challenger sees downside.  A
    # stable starter and a volatile rotation candidate with the same mean are
    # therefore no longer treated as equivalent armband options.
    multiplier = 1.0 + downside_strength * downside * (0.75 + 0.25 * mixture.entropy)
    adjusted = np.asarray(score, float) * np.clip(multiplier, 0.55, 1.0)
    adjusted[data["fixture_count"].to_numpy(int) == 0] = 0.0
    return adjusted, mixture


def validate_forecast_inputs(columns: list[str] | tuple[str, ...]) -> None:
    leaked = sorted(FORBIDDEN_INPUTS.intersection(columns))
    if leaked:
        raise ValueError(f"Forecast layer received realised fields: {leaked}")
