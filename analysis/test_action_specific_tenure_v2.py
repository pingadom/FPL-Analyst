from __future__ import annotations

import unittest

import numpy as np

from action_specific_tenure_v2 import tenure_probabilities


class ActionSpecificTenureTests(unittest.TestCase):
    def test_weights_sum_to_one(self) -> None:
        result = tenure_probabilities(np.array([1.0, 2.0, 4.5, 8.0, 10.0]))
        np.testing.assert_allclose(result.sum(axis=1), 1.0)

    def test_exact_knots_are_one_hot(self) -> None:
        result = tenure_probabilities(np.array([1.0, 3.0, 6.0, 10.0]))
        np.testing.assert_allclose(result, np.eye(4))


if __name__ == "__main__":
    unittest.main()
