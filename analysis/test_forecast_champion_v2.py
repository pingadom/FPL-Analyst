from __future__ import annotations

import unittest

import forecast_champion_v2 as champion


class ForecastChampionTests(unittest.TestCase):
    def test_selected_configuration_is_frozen(self) -> None:
        self.assertEqual(champion.MODEL_ID, "forecast-v2-dynamic-captain-030-share-010")
        self.assertEqual(champion.DYNAMIC_ROUTE_STRENGTH, 0.30)
        self.assertEqual(champion.CAPTAIN_SHARE, 0.10)


if __name__ == "__main__":
    unittest.main()
