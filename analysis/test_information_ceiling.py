"""Integrity tests for the hindsight-only information ceiling tournament."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from forecast_routes import route_components
from information_ceiling_tournament import OracleSpec, oracle_scores


class InformationCeilingTests(unittest.TestCase):
    def fixture(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "fixture_count": [1, 1, 0],
                "expected_minutes": [75.0, 72.0, 60.0],
                "minutes": [90.0, 0.0, 0.0],
                "position_id": [3, 2, 4],
                "fixture_now": [0.6, 0.4, 0.5],
                "season": ["x", "x", "x"],
                "GW": [1, 1, 1],
                "opponent_goal_vulnerability": [1.0, 1.0, 1.0],
                "opponent_assist_vulnerability": [1.0, 1.0, 1.0],
                "play_probability": [0.9, 0.9, 0.8],
                "sixty_probability": [0.8, 0.75, 0.6],
                "goal_rate": [0.25, 0.05, 0.3],
                "assist_rate": [0.2, 0.08, 0.1],
                "team_clean_probability": [0.35, 0.45, 0.3],
                "clean_sheet_rate": [0.25, 0.4, 0.0],
                "bonus_rate": [0.3, 0.2, 0.25],
                "appearances_observed": [1, 0, 0],
                "sixty_observed": [1, 0, 0],
                "team_expected_goals_for": [1.5, 1.2, 0.0],
                "team_goals": [2, 0, 0],
                "team_clean_sheets": [0, 1, 0],
                "team_games": [1, 1, 0],
                "goals": [1, 0, 0],
                "assists": [0, 0, 0],
                "points": [10.0, 0.0, 0.0],
            }
        )

    def test_route_components_reconstruct_baseline(self) -> None:
        data = self.fixture()
        baseline = np.array([5.5, 4.2, 0.0])
        routes = route_components(data, baseline)
        reconstructed = sum(
            routes[name]
            for name in ("appearance", "attack", "clean", "bonus", "residual")
        )
        np.testing.assert_allclose(reconstructed, baseline)

    def test_perfect_total_points_is_exact(self) -> None:
        data = self.fixture()
        scores = oracle_scores(
            data,
            np.array([5.5, 4.2, 0.0]),
            OracleSpec("total", "test", perfect_total_points=True),
        )
        np.testing.assert_allclose(scores, data["points"].to_numpy(float))

    def test_perfect_minutes_penalises_a_known_no_show(self) -> None:
        data = self.fixture()
        baseline = np.array([5.5, 4.2, 0.0])
        scores = oracle_scores(
            data,
            baseline,
            OracleSpec("minutes", "test", perfect_minutes=True),
        )
        self.assertLess(scores[1], baseline[1])
        self.assertEqual(scores[2], 0.0)

    def test_player_involvement_raises_known_scorer(self) -> None:
        data = self.fixture()
        baseline = np.array([5.5, 4.2, 0.0])
        scores = oracle_scores(
            data,
            baseline,
            OracleSpec("involvement", "test", perfect_player_involvement=True),
        )
        self.assertGreater(scores[0], baseline[0])


if __name__ == "__main__":
    unittest.main()
