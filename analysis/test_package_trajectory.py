from __future__ import annotations

import unittest

from package_trajectory_validation import restore_squad, serialise_squad, state_signature


class PackageTrajectoryTests(unittest.TestCase):
    def test_squad_state_round_trip_preserves_prices_and_nationality(self) -> None:
        squad = {
            9: {
                "position": 3,
                "team": 4,
                "purchase": 78,
                "last_price": 82,
                "nationality": "Egypt",
            },
            2: {
                "position": 1,
                "team": 7,
                "purchase": 45,
                "last_price": 44,
                "nationality": "England",
            },
        }
        self.assertEqual(restore_squad(serialise_squad(squad)), squad)

    def test_state_signature_changes_with_purchase_price(self) -> None:
        first = {
            1: {
                "position": 3,
                "team": 2,
                "purchase": 70,
                "last_price": 80,
                "nationality": "",
            }
        }
        second = {1: {**first[1], "purchase": 75}}
        self.assertNotEqual(state_signature(first), state_signature(second))


if __name__ == "__main__":
    unittest.main()
