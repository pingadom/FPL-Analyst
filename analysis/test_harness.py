"""Regression tests for the controlled-experiment harness."""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import calibrate_model as lens
import harness


class HarnessTests(unittest.TestCase):
    def _config(self) -> harness.Config:
        return harness.Config(
            candidate=lens.Candidate(
                0.30, 0.06, 0.00, 0.13, 0.17, 0.03, 0.18, 0.13, 0.78
            ),
            strategy=lens.WEEKLY_CHASE_STRATEGY,
            chip_policy=None,
        )

    def test_with_field_changes_exactly_one_thing(self) -> None:
        base = self._config()
        changed = base.with_field("transfer_hurdle", 1.5)
        self.assertEqual(changed.strategy.transfer_hurdle, 1.5)
        # Everything else must be untouched — that is the guarantee the whole
        # module exists to provide.
        self.assertEqual(
            replace(changed.strategy, transfer_hurdle=base.strategy.transfer_hurdle),
            base.strategy,
        )
        self.assertEqual(changed.candidate, base.candidate)
        self.assertEqual(changed.chip_policy, base.chip_policy)

    def test_with_field_routes_to_the_owning_object(self) -> None:
        base = replace(
            self._config(),
            chip_policy=lens.ChipPolicy(44.0, 13.0, 19.0, 9.6, 0.2, 10, 20),
        )
        self.assertEqual(base.with_field("wildcard_gap", 20.0).chip_policy.wildcard_gap, 20.0)
        self.assertEqual(base.with_field("max_hits", 2).strategy.max_hits, 2)
        self.assertEqual(base.with_field("age", 0.5).candidate.age, 0.5)
        self.assertTrue(base.with_field("robust_planning", True).robust_planning)

    def test_unknown_field_is_rejected_rather_than_ignored(self) -> None:
        # Silently accepting a typo would produce two "different" configs that
        # are identical, and a confident zero-effect result.
        with self.assertRaises(KeyError):
            self._config().with_field("not_a_real_field", 1)

    def test_verdict_calls_a_sub_noise_effect_noise(self) -> None:
        small = harness.Comparison("x", 4.0, 4.0, 4.0, standard_error=40.0, confidence=0.6)
        self.assertEqual(small.verdict, "indistinguishable from noise")
        unresolved = harness.Comparison(
            "x", 50.0, 50.0, 50.0, standard_error=20.0, confidence=0.6
        )
        self.assertEqual(unresolved.verdict, "unresolved")
        clear = harness.Comparison(
            "x", 50.0, 50.0, 50.0, standard_error=10.0, confidence=0.95
        )
        self.assertEqual(clear.verdict, "better")

    def test_outcome_splits_training_from_evaluation(self) -> None:
        totals = np.arange(len(lens.SEASONS), dtype=float)
        outcome = harness.Outcome("x", totals, [], [])
        training = len(lens.TRAINING_SEASONS)
        self.assertAlmostEqual(outcome.training, totals[:training].mean())
        self.assertAlmostEqual(outcome.evaluation, totals[training:].mean())
        self.assertAlmostEqual(outcome.overall, totals.mean())


if __name__ == "__main__":
    unittest.main()
