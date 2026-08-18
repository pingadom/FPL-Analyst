"""Shared forecast-route decomposition with no realised-outcome features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def route_components(data: pd.DataFrame, scores: np.ndarray) -> dict[str, np.ndarray]:
    """Reconstruct four identifiable forecast routes and the ensemble residual."""
    fixture = data["fixture_count"].clip(lower=1).to_numpy(float)
    minutes_factor = data["expected_minutes"].to_numpy(float) / 90.0
    fixture_multiplier = 0.72 + 0.56 * data["fixture_now"].fillna(0.5).to_numpy(float)
    position = data["position_id"].to_numpy(int)
    goal_points = np.choose(position - 1, [6.0, 6.0, 5.0, 4.0])
    clean_points = np.choose(position - 1, [4.0, 4.0, 1.0, 0.0])
    group = [data["season"], data["GW"], data["position_id"]]
    goal_vulnerability = (
        data["opponent_goal_vulnerability"]
        / data["opponent_goal_vulnerability"]
        .groupby(group)
        .transform("median")
        .clip(lower=0.01)
    ).clip(0.68, 1.42).to_numpy(float)
    assist_vulnerability = (
        data["opponent_assist_vulnerability"]
        / data["opponent_assist_vulnerability"]
        .groupby(group)
        .transform("median")
        .clip(lower=0.01)
    ).clip(0.72, 1.35).to_numpy(float)
    appearance = (
        data["play_probability"].to_numpy(float)
        + data["sixty_probability"].to_numpy(float)
    ) * fixture
    attack = (
        data["goal_rate"].to_numpy(float) * goal_points * goal_vulnerability
        + 3.0 * data["assist_rate"].to_numpy(float) * assist_vulnerability
    ) * minutes_factor * fixture_multiplier * fixture
    blended_clean = (
        0.82 * data["team_clean_probability"] + 0.18 * data["clean_sheet_rate"]
    ).clip(0.03, 0.78).to_numpy(float)
    clean = (
        blended_clean
        * clean_points
        * data["sixty_probability"].to_numpy(float)
        * fixture
    )
    bonus = (
        data["bonus_rate"].to_numpy(float)
        * minutes_factor
        * fixture_multiplier
        * fixture
    )
    known = appearance + attack + clean + bonus
    return {
        "appearance": appearance,
        "attack": attack,
        "clean": clean,
        "bonus": bonus,
        "residual": scores - known,
        "goalPoints": goal_points,
        "cleanPoints": clean_points,
        "fixture": fixture,
    }
