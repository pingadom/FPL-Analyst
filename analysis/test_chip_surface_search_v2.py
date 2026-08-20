from __future__ import annotations

import unittest

import pandas as pd

from chip_surface_search_v2 import sequential_chip_gain


class ChipSurfaceSearchTests(unittest.TestCase):
    def test_first_crossing_does_not_look_ahead(self) -> None:
        frame = pd.DataFrame(
            [
                {"season": "2024-25", "gw": 1, "bbSignal": 10, "tcSignal": 0, "fhSignal": 0, "actualBenchBoostGain": 1, "actualTripleCaptainGain": 0, "actualFreeHitGain": 0},
                {"season": "2024-25", "gw": 2, "bbSignal": 20, "tcSignal": 0, "fhSignal": 0, "actualBenchBoostGain": 9, "actualTripleCaptainGain": 0, "actualFreeHitGain": 0},
            ]
        )
        gain, choices = sequential_chip_gain(
            frame,
            {"Bench Boost": 10, "Triple Captain": 99, "Free Hit": 99},
        )
        self.assertEqual(gain, 1)
        self.assertEqual(choices[0]["gw"], 1)


if __name__ == "__main__":
    unittest.main()
