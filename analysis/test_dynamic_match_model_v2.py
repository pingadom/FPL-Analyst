from __future__ import annotations

import unittest

import numpy as np

import dynamic_match_model_v2 as model


class DynamicMatchModelTests(unittest.TestCase):
    def test_geometric_blend_respects_endpoints(self) -> None:
        structural = np.array([1.0, 2.0])
        market = np.array([2.0, 1.0])
        np.testing.assert_allclose(model.geometric_blend(structural, market, 0), structural)
        np.testing.assert_allclose(model.geometric_blend(structural, market, 1), market)

    def test_weight_selection_prefers_more_accurate_source(self) -> None:
        actual = np.array([0, 1, 2, 3, 1, 2] * 100, dtype=float)
        structural = np.full_like(actual, 1.4)
        market = np.clip(actual + 0.05, 0.05, None)
        self.assertGreater(model._selected_weight(structural, market, actual), 0.7)

    def test_probability_blend_is_bounded(self) -> None:
        result = model.probability_blend(np.array([0.001, 0.999]), np.array([0.8, 0.2]), 0.5)
        self.assertTrue(np.all((result >= 0.01) & (result <= 0.90)))


if __name__ == "__main__":
    unittest.main()
