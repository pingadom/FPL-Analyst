from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from breakthrough_engine import (
    ActionRiskPolicy,
    ChipState,
    ChipTransition,
    FieldabilityPolicy,
    ScenarioConfig,
    choose_conservative_action,
    fieldability_audit,
    hard_unavailable,
    optimise_chip_sequence,
    premium_access_diagnostic,
    regime_change_probability,
    sample_correlated_player_scenarios,
)


class BreakthroughEngineTests(unittest.TestCase):
    def test_hard_unavailable_covers_blank_and_confirmed_out(self) -> None:
        self.assertTrue(
            hard_unavailable(
                {
                    "opponent": "-",
                    "researchFeatures": {"fixture_count": 0},
                    "minutesModel": {
                        "availabilityEvidence": {"status": "a", "chance": 100}
                    },
                }
            )
        )
        self.assertTrue(
            hard_unavailable(
                {
                    "opponent": "ARS",
                    "researchFeatures": {"fixture_count": 1},
                    "minutesModel": {
                        "availabilityEvidence": {"status": "i", "chance": 0}
                    },
                }
            )
        )
        self.assertFalse(
            hard_unavailable(
                {
                    "opponent": "ARS",
                    "researchFeatures": {"fixture_count": 1},
                    "minutesModel": {
                        "playProbability": 75,
                        "availabilityEvidence": {"status": "d", "chance": 75},
                    },
                }
            )
        )

    def test_fieldability_audit_reports_known_zero(self) -> None:
        players = [
            {
                "opponent": "ARS",
                "position": "GK" if index == 0 else "MID",
                "researchFeatures": {"fixture_count": 1, "play_probability": 0.9},
            }
            for index in range(15)
        ]
        players[3]["researchFeatures"]["fixture_count"] = 0
        audit = fieldability_audit(players, list(range(15)), list(range(11)))
        self.assertEqual(audit["hardUnavailableXi"], [3])
        self.assertEqual(audit["xiWithFixture"], 10)

    def test_scenarios_share_clean_sheet_outcomes(self) -> None:
        frame = pd.DataFrame(
            {
                "season": ["2025-26", "2025-26"],
                "GW": [1, 1],
                "team_id": [1, 1],
                "position_id": [2, 2],
                "fixture_count": [1, 1],
                "expected_minutes": [90, 90],
                "fixture_now": [0.5, 0.5],
                "play_probability": [1.0, 1.0],
                "sixty_probability": [1.0, 1.0],
                "goal_rate": [0.0, 0.0],
                "assist_rate": [0.0, 0.0],
                "team_clean_probability": [0.5, 0.5],
                "clean_sheet_rate": [0.5, 0.5],
                "bonus_rate": [0.0, 0.0],
                "opponent_goal_vulnerability": [1.0, 1.0],
                "opponent_assist_vulnerability": [1.0, 1.0],
                "prediction_uncertainty": [0.1, 0.1],
            }
        )
        bundle = sample_correlated_player_scenarios(
            frame, np.asarray([4.0, 4.0]), ScenarioConfig(draws=256, seed=7)
        )
        self.assertEqual(bundle.points.shape, (256, 2))
        self.assertGreater(np.corrcoef(bundle.points.T)[0, 1], 0.6)

    def test_conservative_action_abstains_on_weak_edge(self) -> None:
        hold = np.zeros(512)
        rng = np.random.default_rng(5)
        weak = rng.normal(0.1, 2.0, 512)
        selected, rows = choose_conservative_action(
            hold,
            [{"id": "Transfer", "points": weak}],
            ActionRiskPolicy(minimum_cvar10=-10),
        )
        self.assertEqual(selected, "Hold")
        self.assertFalse(rows[0].passes_gate)

    def test_conservative_action_selects_strong_paired_edge(self) -> None:
        hold = np.zeros(512)
        selected, rows = choose_conservative_action(
            hold,
            [{"id": "Transfer", "points": np.full(512, 2.0)}],
            ActionRiskPolicy(minimum_cvar10=-1),
        )
        self.assertEqual(selected, "Transfer")
        self.assertTrue(rows[0].passes_gate)

    def test_premium_access_values_captaincy(self) -> None:
        result = premium_access_diagnostic(
            premium_id=99,
            current_package_value=40.0,
            premium_package_value=38.0,
            future_captain_probabilities=[0.8, 0.7],
            future_captain_edges=[2.0, 2.0],
            liquidity_value=0.5,
            model_disagreement=0.2,
        )
        self.assertTrue(result.access_failure)
        self.assertGreater(result.robust_advantage, 0)

    def test_regime_signal_responds_to_role_and_minutes(self) -> None:
        stable = regime_change_probability(0.4, 0.4, 65, 65, 0.0, False, 12)
        changed = regime_change_probability(0.8, 0.4, 85, 55, 0.6, True, 12)
        self.assertGreater(changed, stable)

    def test_chip_sequence_respects_inventory_and_fh_non_persistence(self) -> None:
        def transitions(state: ChipState):
            result = [ChipTransition("Hold", 0.0, state.permanent_state)]
            if state.week == 1:
                result.append(
                    ChipTransition(
                        "Free Hit",
                        8.0,
                        "temporary",
                        consumes_chip="FH",
                        preserves_permanent_state=True,
                    )
                )
            if state.week == 2:
                result.append(
                    ChipTransition(
                        "Wildcard", 6.0, "wildcarded", consumes_chip="WC"
                    )
                )
            return result

        plan = optimise_chip_sequence(
            ChipState(1, 2, frozenset({"FH", "WC"}), permanent_state="base"),
            transitions,
            discount=1.0,
        )
        self.assertEqual(plan.actions, ((1, "Free Hit"), (2, "Wildcard")))
        self.assertEqual(plan.terminal_state, "wildcarded")
        self.assertEqual(plan.total_value, 14.0)


if __name__ == "__main__":
    unittest.main()

