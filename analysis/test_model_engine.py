"""Focused regression tests for the audited Python model engine."""

from __future__ import annotations

import math
import unittest
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import calibrate_model as lens
from dataclasses import replace
from multiscale_horizon_validation import (
    add_targets,
    expected_tenure,
    remaining_events,
)

import pandas as pd


class ModelEngineTests(unittest.TestCase):
    def test_multiscale_labels_have_declared_maturity(self) -> None:
        frame = pd.DataFrame(
            {
                "season": ["toy"] * 5,
                "player_key": ["p"] * 5,
                "GW": [1, 2, 3, 4, 5],
                "points": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        labelled = add_targets(frame)
        self.assertAlmostEqual(
            labelled.loc[0, "target_h3"],
            1.0 + 0.86 * 2.0 + 0.86**2 * 3.0,
        )
        self.assertEqual(int(labelled.loc[0, "target_h3_end_gw"]), 3)
        self.assertEqual(int(labelled.loc[3, "target_h10_end_gw"]), 5)

    def test_remaining_events_uses_event_order_not_gw_subtraction(self) -> None:
        frame = pd.DataFrame(
            {
                "season": ["restart"] * 4,
                "GW": [28, 29, 39, 40],
            }
        )
        np.testing.assert_array_equal(remaining_events(frame), [4, 3, 2, 1])

    def test_expected_tenure_is_player_specific_and_bounded(self) -> None:
        frame = pd.DataFrame(
            {
                "observations": [2, 100],
                "minutes_model_confidence": [0.20, 0.62],
                "play_probability": [0.25, 0.95],
                "start_probability": [0.12, 0.90],
                "sixty_probability": [0.08, 0.88],
                "rotation_volatility": [0.90, 0.02],
                "team_rating_confidence": [0.10, 0.92],
                "team_regime_shift": [0.60, 0.02],
            }
        )
        tenure = expected_tenure(frame)
        self.assertTrue(np.all((tenure >= 2) & (tenure <= 10)))
        self.assertLess(tenure[0], tenure[1])

    def test_triple_captain_signal_uses_points_not_rank_percentile(self) -> None:
        # The projection is already a Gameweek total, so the signal must not
        # multiply a Double Gameweek in a second time.
        self.assertAlmostEqual(lens.triple_captain_signal(15.2, 2), 15.2)
        self.assertAlmostEqual(lens.triple_captain_signal(7.6, 1), 7.6)
        self.assertAlmostEqual(lens.triple_captain_signal(7.6, 0), 0.0)

    def test_chip_allow_list_round_trips_to_audit_payload(self) -> None:
        policy = lens.ChipPolicy(
            1e6,
            1e6,
            11,
            15,
            0,
            enabled_chips=("Bench Boost", "Triple Captain"),
        )
        self.assertEqual(
            policy.as_dict()["enabledChips"],
            ["Bench Boost", "Triple Captain"],
        )

    def test_captain_utility_can_be_separate_from_lineup_scores(self) -> None:
        squad = {
            1: {"position": 1, "team": 1},
            2: {"position": 1, "team": 2},
            **{
                element: {"position": 2, "team": element}
                for element in range(3, 8)
            },
            **{
                element: {"position": 3, "team": element}
                for element in range(8, 13)
            },
            **{
                element: {"position": 4, "team": element}
                for element in range(13, 16)
            },
        }
        rows = {element: element - 1 for element in squad}
        scores = np.arange(1, 16, dtype=float)
        captain_scores = np.zeros(15, dtype=float)
        captain_scores[11] = 50.0
        ordinary = lens.squad_decision_utility(
            squad, rows, scores, captain_weight=1.0
        )
        separated = lens.squad_decision_utility(
            squad,
            rows,
            scores,
            captain_weight=1.0,
            captain_scores=captain_scores,
        )
        self.assertGreater(separated, ordinary)

    def test_disabled_chip_windows_are_actually_absent(self) -> None:
        enabled = {"Bench Boost", "Triple Captain"}
        windows = [
            row
            for row in lens.chip_windows("2024-25", 1, 38, 10, 28)
            if row["chip"] in enabled
        ]
        self.assertTrue(windows)
        self.assertEqual({row["chip"] for row in windows}, enabled)

    def test_package_search_preserves_a_funding_move_for_a_premium(self) -> None:
        positions = np.array(
            [1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 3, 4]
        )
        elements = np.arange(1, 18)
        teams = np.arange(1, 18)
        prices = np.full(17, 50)
        prices[15] = 35  # funding midfielder
        prices[16] = 65  # currently unaffordable premium forward
        scores = np.full(17, 5.0)
        scores[15] = 4.5
        scores[16] = 15.0
        squad = {
            element: {
                "position": int(positions[element - 1]),
                "team": int(teams[element - 1]),
                "purchase": int(prices[element - 1]),
                "last_price": int(prices[element - 1]),
                "nationality": "",
            }
            for element in range(1, 16)
        }
        row_by_element = {element: element - 1 for element in elements}
        incoming = {
            1: np.array([0, 1]),
            2: np.array([2, 3, 4, 5, 6]),
            3: np.array([7, 8, 9, 10, 11, 15]),
            4: np.array([16, 12, 13, 14]),
        }
        base_strategy = lens.SimulationStrategy(
            name="toy",
            transfer_hurdle=16.0,
            bank_limit=5,
            force_weekly_review=False,
            safe_captain=False,
            max_hits=0,
            joint_squad_optimiser=True,
        )

        def run(strategy: lens.SimulationStrategy) -> tuple[dict, int]:
            result, _, moves, _, _ = lens.joint_transfer_plan(
                squad={key: value.copy() for key, value in squad.items()},
                bank=0,
                free_transfers=1,
                row_by_element=row_by_element,
                incoming_by_position=incoming,
                element_values=elements,
                position_values=positions,
                team_values=teams,
                price_values=prices,
                plan_scores=scores,
                bench_scores=None,
                captain_utility_scores=None,
                price_rise_values=np.zeros(17),
                price_fall_values=np.zeros(17),
                uncertainty_values=np.ones(17),
                risk_scores=None,
                excluded_elements=set(),
                team_option_score={},
                strategy=strategy,
                gw=8,
            )
            return result, moves

        ordinary_squad, ordinary_moves = run(base_strategy)
        routed_squad, routed_moves = run(
            replace(base_strategy, package_route_search=True)
        )
        self.assertEqual(ordinary_moves, 0)
        self.assertEqual(set(ordinary_squad), set(squad))
        self.assertEqual(routed_moves, 1)
        self.assertIn(16, routed_squad)
        self.assertNotIn(8, routed_squad)

    def test_defensive_stack_has_higher_correlated_downside(self) -> None:
        squad = {
            1: {"position": 1, "team": 1},
            2: {"position": 1, "team": 2},
            3: {"position": 2, "team": 3},
            4: {"position": 2, "team": 3},
            5: {"position": 2, "team": 4},
            6: {"position": 2, "team": 5},
            7: {"position": 2, "team": 6},
            **{element: {"position": 3, "team": element} for element in range(8, 13)},
            **{element: {"position": 4, "team": element} for element in range(13, 16)},
        }
        distributed = {element: state.copy() for element, state in squad.items()}
        distributed[4]["team"] = 7
        rows = {element: element - 1 for element in squad}
        scores = np.arange(1, 16, dtype=float)
        scores[2:7] = 20.0
        risk = np.ones(15, dtype=float)
        stacked_value = lens.squad_decision_utility(
            squad,
            rows,
            scores,
            risk_scores=risk,
            risk_aversion=1.0,
            defence_correlation=0.28,
        )
        distributed_value = lens.squad_decision_utility(
            distributed,
            rows,
            scores,
            risk_scores=risk,
            risk_aversion=1.0,
            defence_correlation=0.28,
        )
        self.assertLess(stacked_value, distributed_value)

    def test_package_action_adjustment_can_change_joint_choice(self) -> None:
        elements = np.arange(1, 17, dtype=int)
        positions = np.array([1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 3])
        teams = np.arange(1, 17, dtype=int)
        prices = np.full(16, 50, dtype=int)
        scores = np.arange(16, 0, -1, dtype=float)
        scores[15] = 0.0
        squad = {
            element: {
                "position": int(positions[element - 1]),
                "team": int(teams[element - 1]),
                "purchase": 50,
                "last_price": 50,
                "nationality": "",
            }
            for element in range(1, 16)
        }
        rows = {element: element - 1 for element in elements}
        incoming = {
            1: np.array([0, 1]),
            2: np.array([2, 3, 4, 5, 6]),
            3: np.array([7, 8, 9, 10, 11, 15]),
            4: np.array([12, 13, 14]),
        }
        strategy = lens.SimulationStrategy(
            name="package-hook",
            transfer_hurdle=1.0,
            bank_limit=5,
            force_weekly_review=False,
            safe_captain=False,
            max_hits=0,
            joint_squad_optimiser=True,
        )

        def favour_sixteen(context: dict) -> float:
            return 20.0 if 16 in context["incomingElements"] else 0.0

        result, _, moves, _, _ = lens.joint_transfer_plan(
            squad={key: value.copy() for key, value in squad.items()},
            bank=0,
            free_transfers=1,
            row_by_element=rows,
            incoming_by_position=incoming,
            element_values=elements,
            position_values=positions,
            team_values=teams,
            price_values=prices,
            plan_scores=scores,
            bench_scores=None,
            captain_utility_scores=None,
            price_rise_values=np.zeros(16),
            price_fall_values=np.zeros(16),
            uncertainty_values=np.ones(16),
            risk_scores=None,
            excluded_elements=set(),
            team_option_score={},
            strategy=strategy,
            gw=8,
            package_action_adjustment=favour_sixteen,
        )
        self.assertEqual(moves, 1)
        self.assertIn(16, result)

    def test_additional_move_hurdle_defaults_to_legacy_value(self) -> None:
        strategy = lens.SimulationStrategy(
            name="default-extra-move-cost",
            transfer_hurdle=16.0,
            bank_limit=5,
            force_weekly_review=False,
            safe_captain=False,
        )
        self.assertEqual(strategy.additional_move_hurdle, 1.15)

    def test_team_ratings_ignore_events_the_club_did_not_play(self) -> None:
        # A blank Gameweek used to enter the attack/defence EWM as a goalless
        # match, depressing attack and flattering defence for weeks afterwards.
        rows = []
        for gw in range(1, 9):
            blank = gw == 5
            rows.append(
                {
                    "season": "toy",
                    "season_order": 0,
                    "GW": gw,
                    "team_id": 1,
                    "team_name": "Scorers",
                    "opponent_team": 2,
                    "was_home": True,
                    "team_games": 0 if blank else 1,
                    "team_goals": 0 if blank else 3,
                    "team_xg": 0 if blank else 3.0,
                    "team_goals_against": 0,
                    "team_xga": 0.0,
                    "team_clean_sheets": 0 if blank else 1,
                    "team_result_points": 0 if blank else 3,
                }
            )
        opponents = [dict(row, team_id=2, team_name="Conceders") for row in rows]
        frame = pd.DataFrame(rows + opponents)
        rated = lens.add_causal_team_strength(frame)
        scorers = rated[rated["team_id"] == 1].set_index("GW")
        # The rating carried into GW6 must reflect the four scoring matches
        # played, not a phantom goalless fifth.
        self.assertGreater(float(scorers.loc[6, "team_attack_rating"]), 2.0)
        self.assertAlmostEqual(
            float(scorers.loc[6, "team_attack_rating"]),
            float(scorers.loc[5, "team_attack_rating"]),
            places=6,
        )

    def test_expiring_chip_threshold_ramps_down_to_a_token_check(self) -> None:
        def threshold(chip: str, remaining: int) -> float:
            floor = lens.CHIP_EXPIRY_THRESHOLD_SHARE[chip]
            share = 1.0 - math.exp(-remaining / lens.CHIP_HOLD_DECAY_GWS)
            return floor + (1.0 + lens.CHIP_HOLD_VALUE - floor) * share

        for chip in lens.CHIP_EXPIRY_THRESHOLD_SHARE:
            ladder = [threshold(chip, remaining) for remaining in range(0, 20)]
            self.assertAlmostEqual(
                ladder[0], lens.CHIP_EXPIRY_THRESHOLD_SHARE[chip]
            )
            self.assertTrue(all(a < b for a, b in zip(ladder, ladder[1:])))
            # Patient while the window is open.
            self.assertGreater(ladder[-1], 1.0)

    def test_wildcard_expires_dearer_than_the_one_week_chips(self) -> None:
        """A Wildcard's cost is the squad trajectory it leaves behind.

        Free Hit, Bench Boost and Triple Captain settle inside their own
        Gameweek, so an unplayed one really is worth nothing and dumping it is
        close to free. Treating the Wildcard the same way fired it 1.75 times a
        season out of a possible 2.
        """
        floors = lens.CHIP_EXPIRY_THRESHOLD_SHARE
        for chip in ("Free Hit", "Bench Boost", "Triple Captain"):
            self.assertLess(floors[chip], floors["Wildcard"])

    def test_monotone_map_honours_custom_bins_and_bounds(self) -> None:
        counts = np.full(20, 50.0)
        successes = np.linspace(0, 50, 20)
        mapped = lens.monotone_probability_map(
            successes, counts, 0.5, bounds=(2.0, 88.0)
        )
        self.assertEqual(len(mapped), 20)
        self.assertTrue(np.all(np.diff(mapped) >= -1e-9))
        self.assertTrue(np.all((mapped >= 2.0) & (mapped <= 88.0)))

    def test_calibration_tier_tracks_price_within_the_group(self) -> None:
        frame = pd.DataFrame(
            {
                "position_id": [3] * 100,
                "price": list(range(40, 140)),
            }
        )
        tiers = lens.minutes_calibration_tier(frame, ["position_id"])
        self.assertEqual(tiers[0], 0)
        self.assertEqual(tiers[-1], 2)
        # Monotone in price: a dearer player is never placed in a lower band.
        self.assertTrue(np.all(np.diff(tiers) >= 0))

    def test_minutes_calibration_undoes_the_compression(self) -> None:
        """A compressed predictor must be pulled apart by realised outcomes.

        Every player is predicted to start half the time. The cheap ones actually
        start one week in ten and the expensive ones nine, which is exactly the
        residual the raw beta prior cannot express.
        """
        rows = []
        for gw in range(1, 21):
            for index in range(200):
                rows.append((gw, 40, 0.1, f"cheap{index}"))
            for index in range(30):
                rows.append((gw, 100, 0.9, f"prem{index}"))
        frame = pd.DataFrame(rows, columns=["GW", "price", "true_rate", "who"])
        rng = np.random.default_rng(7)
        started = (rng.random(len(frame)) < frame["true_rate"]).astype(float)
        frame = frame.assign(
            season="toy",
            season_order=0,
            position_id=3,
            fixture_count=1.0,
            starts_observed=started,
            appearances_observed=started,
            sixty_observed=started,
            start_minutes_total=started * 85.0,
            bench_minutes_total=0.0,
            bench_appearances_observed=0.0,
            start_probability=0.5,
            play_probability=0.5,
            sixty_probability=0.5,
            minutes_if_start=80.0,
            minutes_if_bench=20.0,
        )
        calibrated = lens.causal_calibrate_minutes(frame)
        final = calibrated[calibrated["GW"] == 20]
        cheap = final.loc[final["price"] == 40, "start_probability"].mean()
        premium = final.loc[final["price"] == 100, "start_probability"].mean()
        self.assertLess(cheap, 0.25)
        self.assertGreater(premium, 0.70)
        # The uncalibrated series is retained so the live deadline can fit its
        # own map on the raw predictor rather than on a corrected one.
        self.assertTrue(
            np.allclose(calibrated["start_probability_uncalibrated"], 0.5)
        )

    def test_gate_prefers_regime_comparable_seasons(self) -> None:
        """Once an xG-era season exists the gate must stop using pre-xG ones.

        The decision policies rank oppositely across the two data regimes, so
        selecting on the oldest seasons picks a policy for a game that no longer
        exists.
        """
        weeks = [1.0] * 38
        gate = {
            "central:Six-GW planner + adaptive banking": (
                np.zeros(len(lens.SEASONS)),
                lens.WEEKLY_CHASE_STRATEGY,
                None,
                [{"weeklyPoints": weeks} for _ in lens.SEASONS],
            ),
            "central:Joint transfer-chip tree + hold value": (
                np.zeros(len(lens.SEASONS)),
                lens.JOINT_OPTION_STRATEGY,
                None,
                [{"weeklyPoints": weeks} for _ in lens.SEASONS],
            ),
        }
        modern = lens.SEASONS.index(lens.XG_ERA_FIRST_SEASON)

        _, early = lens.select_gate_option(gate, modern)
        self.assertFalse(early["regimeMatched"])
        self.assertIn(lens.SEASONS[0], early["seasonsUsed"])

        _, late = lens.select_gate_option(gate, len(lens.SEASONS))
        self.assertTrue(late["regimeMatched"])
        self.assertNotIn(lens.SEASONS[0], late["seasonsUsed"])
        self.assertTrue(
            all(
                lens.SEASONS.index(season) >= modern
                for season in late["seasonsUsed"]
            )
        )


if __name__ == "__main__":
    unittest.main()
