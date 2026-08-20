from __future__ import annotations

import unittest

import forecast_champion_v2 as champion


class ForecastChampionTests(unittest.TestCase):
    def test_selected_configuration_is_frozen(self) -> None:
        self.assertEqual(
            champion.MODEL_ID,
            "forecast-v2-dynamic-captain-070-share-020-minutes-050",
        )
        self.assertEqual(champion.DYNAMIC_ROUTE_STRENGTH, 0.70)
        self.assertEqual(champion.CAPTAIN_SHARE, 0.20)
        self.assertEqual(champion.MINUTE_DOWNSIDE, 0.50)
        self.assertEqual(champion.BENCH_BOOST_THRESHOLD, 9.0)
        self.assertEqual(champion.TRIPLE_CAPTAIN_THRESHOLD, 10.0)
        self.assertEqual(champion.FREE_HIT_THRESHOLD, 3.0)


if __name__ == "__main__":
    unittest.main()
