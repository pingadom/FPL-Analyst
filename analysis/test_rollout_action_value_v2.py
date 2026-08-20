from __future__ import annotations

import unittest

from rollout_action_value_v2 import interpolate_rollout


class RolloutActionValueTests(unittest.TestCase):
    def test_interpolates_between_horizons(self) -> None:
        values = {1: 1.0, 3: 3.0, 6: 6.0, 10: 10.0}
        self.assertAlmostEqual(interpolate_rollout(4.5, values), 4.5)

    def test_clamps_outside_horizon(self) -> None:
        values = {1: 2.0, 3: 3.0, 6: 6.0, 10: 8.0}
        self.assertEqual(interpolate_rollout(0.0, values), 2.0)
        self.assertEqual(interpolate_rollout(12.0, values), 8.0)


if __name__ == "__main__":
    unittest.main()
