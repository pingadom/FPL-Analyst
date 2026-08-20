from __future__ import annotations

import unittest

import numpy as np

import forecast_layer_v2 as layer


class ForecastLayerTests(unittest.TestCase):
    def test_minutes_mixture_sums_to_one(self) -> None:
        prediction = {
            "play": np.array([0.9]),
            "start": np.array([0.7]),
            "sixty": np.array([0.5]),
            "minutes": np.array([58.0]),
        }
        mixture = layer.minutes_mixture(prediction)
        total = mixture.no_show + mixture.cameo + mixture.start_under_sixty + mixture.sixty_plus
        np.testing.assert_allclose(total, 1.0)

    def test_probability_coherence_is_enforced(self) -> None:
        prediction = {
            "play": np.array([0.4]),
            "start": np.array([0.8]),
            "sixty": np.array([0.9]),
            "minutes": np.array([100.0]),
        }
        mixture = layer.minutes_mixture(prediction)
        self.assertAlmostEqual(float(mixture.sixty_plus[0]), 0.4)
        self.assertAlmostEqual(float(mixture.expected_minutes[0]), 90.0)

    def test_realised_fields_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            layer.validate_forecast_inputs(("price", "points"))


if __name__ == "__main__":
    unittest.main()
