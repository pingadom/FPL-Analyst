import unittest

import pandas as pd

from calibrate_model import (
    WEEKLY_CHASE_STRATEGY,
    official_availability_chance,
)


class AvailabilityRepairTests(unittest.TestCase):
    def test_nullable_chance_respects_official_status(self):
        chance = pd.Series([None, None, None, None, None, None, 50])
        status = pd.Series(["a", "d", "i", "s", "u", "n", "i"])
        self.assertEqual(
            official_availability_chance(chance, status).tolist(),
            [100.0, 75.0, 0.0, 0.0, 0.0, 0.0, 50.0],
        )

    def test_selected_strategy_keeps_rejected_overcorrections_off(self):
        self.assertFalse(WEEKLY_CHASE_STRATEGY.enforce_weekly_xi_floor)
        self.assertFalse(WEEKLY_CHASE_STRATEGY.consistent_transfer_objective)
        self.assertTrue(WEEKLY_CHASE_STRATEGY.enforce_fieldability)


if __name__ == "__main__":
    unittest.main()
