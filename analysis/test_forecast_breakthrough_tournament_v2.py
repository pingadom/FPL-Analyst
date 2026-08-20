from __future__ import annotations

import unittest

import numpy as np

from forecast_breakthrough_tournament_v2 import blend, summary_with_delta


class ForecastTournamentTests(unittest.TestCase):
    def test_blend_endpoints(self) -> None:
        base = np.array([1.0, 2.0])
        challenger = np.array([3.0, 4.0])
        np.testing.assert_allclose(blend(base, challenger, 0), base)
        np.testing.assert_allclose(blend(base, challenger, 1), challenger)


if __name__ == "__main__":
    unittest.main()
