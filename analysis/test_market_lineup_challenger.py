"""Unit tests for market normalisation and safe score adjustment."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from market_lineup_challenger import (
    first_available,
    implied_total_goals,
    no_vig_probabilities,
    normalize_team,
)


class MarketLineupTests(unittest.TestCase):
    def test_no_vig_probabilities_sum_to_one(self) -> None:
        home, draw, away = no_vig_probabilities(
            np.array([1.8, 2.4]),
            np.array([3.5, 3.2]),
            np.array([4.8, 2.9]),
        )
        np.testing.assert_allclose(home + draw + away, 1.0)

    def test_total_goal_inversion_is_monotonic(self) -> None:
        values = implied_total_goals(np.array([0.35, 0.50, 0.70]))
        self.assertTrue(np.all(np.diff(values) > 0))

    def test_fpl_team_aliases_match_market_names(self) -> None:
        self.assertEqual(normalize_team("Man Utd"), "man united")
        self.assertEqual(normalize_team("Spurs"), "tottenham")
        self.assertEqual(normalize_team("Sheffield Utd"), "sheffield united")

    def test_closing_columns_are_rejected(self) -> None:
        frame = pd.DataFrame({"AvgCH": [1.8]})
        with self.assertRaises(ValueError):
            first_available(frame, ("AvgCH",))


if __name__ == "__main__":
    unittest.main()
