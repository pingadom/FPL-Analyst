from __future__ import annotations

import unittest

import numpy as np

from safe_policy_evaluation_v2 import doubly_robust_value


class SafePolicyEvaluationTests(unittest.TestCase):
    def test_perfect_q_model_returns_reward_mean_on_policy(self) -> None:
        reward = np.array([1.0, 3.0, 5.0])
        estimate, _ = doubly_robust_value(
            np.array([0, 0, 0]), reward, np.ones(3), np.ones(3), reward, reward
        )
        self.assertAlmostEqual(estimate, 3.0)

    def test_zero_propensity_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            doubly_robust_value(
                np.array([0]), np.array([1.0]), np.array([0.0]), np.array([1.0]), np.array([0.0]), np.array([0.0])
            )


if __name__ == "__main__":
    unittest.main()
