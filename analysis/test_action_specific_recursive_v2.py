from __future__ import annotations

import unittest

import numpy as np

from action_specific_recursive_v2 import paired_summary


class ActionSpecificRecursiveTests(unittest.TestCase):
    def test_paired_summary_uses_evaluation_seasons(self) -> None:
        seasons = ["2016-17", "2017-18", "2018-19", "2019-20", "2020-21", "2021-22"]
        baseline = np.array([1, 1, 10, 20, 30, 40], dtype=float)
        totals = baseline + np.array([0, 0, 1, -1, 2, 0], dtype=float)
        result = paired_summary(totals, baseline, seasons)
        self.assertEqual(result["seasonDeltas"], [1, -1, 2, 0])


if __name__ == "__main__":
    unittest.main()
