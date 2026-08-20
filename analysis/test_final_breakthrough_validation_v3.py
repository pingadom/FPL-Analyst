from __future__ import annotations

import unittest

import numpy as np

import calibrate_model as lens
from final_breakthrough_validation_v3 import stage_summary


class FinalBreakthroughValidationTests(unittest.TestCase):
    def test_stage_summary_uses_evaluation_seasons_only(self) -> None:
        totals = np.asarray([1, 2, 10, 20, 30, 40, 50, 60, 70, 80], dtype=float)
        seasons = ["train-a", "train-b", *lens.EVALUATION_SEASONS]
        targets = {
            season: target
            for season, target in zip(
                lens.EVALUATION_SEASONS,
                [8, 25, 30, 40, 50, 60, 70, 80],
            )
        }
        result = stage_summary(
            totals,
            seasons,
            targets,
        )
        self.assertEqual(result["top500Hits"], 7)
        self.assertEqual(result["top500SeasonMargins"], [2, -5, 0, 0, 0, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
