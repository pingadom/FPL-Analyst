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

    def test_zero_uncertainty_is_not_reported_as_unresolved(self) -> None:
        # Two configurations that never diverge produce identical bootstrap
        # draws. Calling that "unresolved" invites someone to go looking for a
        # bigger sample when the honest answer is that nothing happened.
        none = harness.Comparison("x", 0.0, 0.0, 0.0, standard_error=0.0, confidence=0.0)
        self.assertEqual(none.verdict, "no effect on the selecting seasons")
        deterministic = harness.Comparison(
            "x", -6.0, -6.0, -6.0, standard_error=0.0, confidence=0.0
        )
        self.assertEqual(deterministic.verdict, "worse")

    def test_confidently_negative_is_worse_not_unresolved(self) -> None:
        losing = harness.Comparison(
            "x", -50.0, -50.0, -50.0, standard_error=10.0, confidence=0.02
        )
        self.assertEqual(losing.verdict, "worse")

    def test_outcome_splits_training_from_evaluation(self) -> None:
        totals = np.arange(len(lens.SEASONS), dtype=float)
        outcome = harness.Outcome("x", totals, [], [])
        training = len(lens.TRAINING_SEASONS)
        self.assertAlmostEqual(outcome.training, totals[:training].mean())
        self.assertAlmostEqual(outcome.evaluation, totals[training:].mean())
        self.assertAlmostEqual(outcome.overall, totals.mean())


class VerdictDisagreementTests(unittest.TestCase):
    def test_confident_training_gain_contradicted_by_evaluation(self):
        """A training win the evaluation seasons contradict is not a win.

        Measured case: relaxing the bench premium penalty scored +89.0 on the two
        selecting seasons and -29.8 on the eight reported ones, at confidence
        0.983. Labelled "better" it would have been shipped.
        """
        comparison = harness.Comparison(
            label="bench_premium_penalty=0.0",
            delta_training=89.0,
            delta_evaluation=-29.8,
            delta_overall=-6.0,
            standard_error=43.9,
            confidence=0.983,
        )
        self.assertEqual(comparison.verdict, "training-only; evaluation disagrees")

    def test_agreeing_directions_still_read_plainly(self):
        agreeing = harness.Comparison(
            label="good", delta_training=50.0, delta_evaluation=20.0,
            delta_overall=26.0, standard_error=10.0, confidence=0.9,
        )
        self.assertEqual(agreeing.verdict, "better")
        losing = harness.Comparison(
            label="bad", delta_training=-50.0, delta_evaluation=-20.0,
            delta_overall=-26.0, standard_error=10.0, confidence=0.1,
        )
        self.assertEqual(losing.verdict, "worse")


def synthetic(label: str, weekly_edge: list[float]) -> harness.Outcome:
    """Ten seasons of 38 weeks; each season's weeks sit `edge` above a noisy base."""
    rng = np.random.default_rng(7)
    base = rng.normal(55, 15, size=(10, 38))
    weekly = [list(base[s] + weekly_edge[s]) for s in range(10)]
    return harness.Outcome(
        label=label,
        totals=np.array([sum(week) for week in weekly]),
        weekly=weekly,
        stats=[],
    )


class WalkForwardTests(unittest.TestCase):
    def test_a_rival_that_is_better_everywhere_is_adopted_and_scored(self) -> None:
        baseline = synthetic("base", [0.0] * 10)
        better = synthetic("better", [3.0] * 10)
        result = harness.walk_forward(baseline, [better])
        self.assertEqual(result.chosen, ["better"] * 8)
        self.assertAlmostEqual(result.mean_delta, 3.0 * 38, places=6)

    def test_a_training_only_winner_is_dropped_once_later_seasons_disagree(self) -> None:
        """The failure mode the old protocol could not see: good on the first
        two seasons, bad afterwards. Walk-forward adopts it, pays for it, then
        lets it go as evidence accumulates, and the score reflects all of that."""
        baseline = synthetic("base", [0.0] * 10)
        mirage = synthetic("mirage", [4.0, 4.0] + [-4.0] * 8)
        result = harness.walk_forward(baseline, [mirage])
        self.assertEqual(result.chosen[0], "mirage")
        self.assertEqual(result.chosen[-1], "base")
        self.assertLess(result.mean_delta, 0.0)

    def test_no_evidence_keeps_the_incumbent(self) -> None:
        baseline = synthetic("base", [0.0] * 10)
        same = synthetic("same", [0.0] * 10)
        result = harness.walk_forward(baseline, [same])
        self.assertEqual(result.chosen, ["base"] * 8)
        self.assertEqual(result.mean_delta, 0.0)


if __name__ == "__main__":
    unittest.main()
