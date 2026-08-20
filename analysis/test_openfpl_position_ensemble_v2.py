from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import openfpl_position_ensemble_v2 as ensemble


class OpenFplPositionEnsembleTests(unittest.TestCase):
    def test_grouped_percentile_stays_with_position_and_week(self) -> None:
        data = pd.DataFrame(
            {
                "season": ["a"] * 4,
                "GW": [1] * 4,
                "position_id": [1, 1, 2, 2],
            }
        )
        result = ensemble.grouped_percentile(data, np.array([1, 2, 100, 200]))
        np.testing.assert_allclose(result, [0.5, 1.0, 0.5, 1.0])

    def test_negative_coefficients_are_removed(self) -> None:
        fitted = type("Model", (), {"coef_": np.array([-2.0, 1.0, 3.0, 0.0])})()
        result = ensemble._normalised_positive_coefficients(fitted)
        np.testing.assert_allclose(result, [0.0, 0.25, 0.75, 0.0])


if __name__ == "__main__":
    unittest.main()
