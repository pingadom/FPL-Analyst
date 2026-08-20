from __future__ import annotations

import unittest

from captain_surface_search_v2 import development_stability


class CaptainSurfaceSearchTests(unittest.TestCase):
    def test_holdout_is_excluded_from_stability(self) -> None:
        stable = development_stability({"seasonDeltas": [1, 1, 1, 1, 1, 1, -100, -100]})
        self.assertAlmostEqual(stable, 1.0)


if __name__ == "__main__":
    unittest.main()
