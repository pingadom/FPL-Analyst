"""Focused regression tests for the audited Python model engine."""

from __future__ import annotations

import math
import unittest
from unittest import mock
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

    def test_placeholder_seasons_have_real_club_names(self):
        """The two nameless seasons must resolve to twenty real clubs each.

        Placeholder names are not cosmetic. `add_causal_team_strength` scopes its
        `team_key` per season for anything starting "Team ", so a placeholder
        restarts every club's rating history at the 2017 to 2018 boundary, and no
        external source can be joined by club.
        """
        import historical_odds as odds
        import team_identity as identity

        for season in identity.PLACEHOLDER_SEASONS:
            mapping = identity.PLACEHOLDER_TEAM_NAMES[season]
            self.assertEqual(len(mapping), 20)
            self.assertEqual(sorted(mapping), list(range(1, 21)))
            self.assertFalse(
                any(name.startswith("Team ") for name in mapping.values())
            )
            keys = {odds.normalise_team(name) for name in mapping.values()}
            self.assertEqual(len(keys), 20)
            # The reconstructed clubs must be exactly the twenty an independent
            # record says played that season — not merely twenty distinct names.
            self.assertEqual(keys, set(identity.season_club_names(season)))

    def test_recovered_names_share_team_keys_with_named_seasons(self):
        """A club in both eras must key identically, or its rating still restarts."""
        import historical_odds as odds
        import team_identity as identity

        earlier = {
            odds.normalise_team(name)
            for name in identity.PLACEHOLDER_TEAM_NAMES["2017-18"].values()
        }
        later = set(identity.season_club_names("2018-19"))
        # Sixteen clubs survived 2017/18 into 2018/19; the exact figure matters
        # less than that the overlap is substantial rather than empty, which is
        # what a spelling mismatch would produce.
        self.assertGreaterEqual(len(earlier & later), 15)

    def test_market_uses_opening_odds_never_closing(self):
        """The opening line is priced before the deadline; the closing line is not.

        football-data ships both, and they are not interchangeable — they differ on
        the large majority of matches. Reading a `C`-infixed column would turn this
        forecast into a leak, silently, because a closing line is strictly the
        better predictor and would look like an improvement.
        """
        import historical_odds as odds

        selected = [
            name
            for group in (*odds.PRICE_COLUMNS, *odds.OVER_COLUMNS)
            for name in group
        ]
        self.assertTrue(selected)
        for name in selected:
            # Closing columns are the opening name with a C before the outcome:
            # PSH -> PSCH, Avg>2.5 -> AvgC>2.5, B365H -> B365CH.
            self.assertNotIn("C>", name)
            self.assertNotIn("C<", name)
            self.assertFalse(
                name.endswith(("CH", "CD", "CA")),
                f"{name} is a closing-odds column",
            )

    def test_market_blend_has_its_own_cache_when_enabled(self):
        """A blended frame must never be reachable under an unblended name.

        The blend is off by default: it measured +12.8 with the gate pinned and
        -5.5 without, and production does not pin. What must not change either way
        is the namespacing — a frame built at one weight being served to a run at
        another would silently mix two experiments, which is the contamination the
        version bump exists to stop.
        """
        if lens.MARKET_BLEND_WEIGHT > 0.0:
            self.assertIn("market", lens.PREPARED_HISTORY_CACHE.name)
        else:
            self.assertNotIn("market", lens.PREPARED_HISTORY_CACHE.name)

    def test_diagnostic_chip_policy_is_not_chosen_on_the_reported_seasons(self):
        """The *published* best-policy figure must not be picked on its own test.

        This governs `chipStrategy.policy` and `diagnosticBestPolicyAverageGain`
        only. The reported backtest comes from the walk-forward loop, which
        already slices `chip_gains[:, :season_id]`, so runs under "training" and
        "all" produce byte-identical season totals — the fix is to a published
        figure, not to the score.
        """
        self.assertEqual(lens.CHIP_POLICY_SELECTION, "training")
        # The training seasons must be a strict prefix of SEASONS, or a
        # `[:training_count]` slice would not mean "the selecting seasons".
        self.assertEqual(
            lens.SEASONS[: len(lens.TRAINING_SEASONS)], list(lens.TRAINING_SEASONS)
        )

    def test_european_proximity_finds_the_right_ties(self):
        """Days to and from the nearest European match, against known fixtures.

        Manchester City played Real Madrid in the 2023/24 quarter-final on the
        Tuesday and the following Wednesday, with a league game between them.
        """
        import european_fixtures as euro

        rows = pd.DataFrame(
            {
                "team_id": [43, 43, 43, 43, 1],
                "kickoff_time": pd.to_datetime(
                    [
                        "2024-04-13T11:30:00Z",  # between both quarter-final legs
                        "2023-11-25T15:00:00Z",  # three days before a group tie
                        "2024-02-10T15:00:00Z",  # three days before a last-16 tie
                        "2024-04-20T14:00:00Z",  # three days after the second leg
                        "2024-04-13T14:00:00Z",  # a club with no European football
                    ],
                    utc=True,
                ),
            }
        )
        out = euro.attach_european_proximity(
            rows, "2023-24", {43: "Man City", 1: "Burnley"}
        )
        self.assertAlmostEqual(out["european_days_since"].iloc[0], 4.479, places=2)
        self.assertAlmostEqual(out["european_days_to"].iloc[0], 3.521, places=2)
        # A knockout tie within four days counts; a group tie the same distance
        # away does not, which is the entire distinction the penalty rests on.
        self.assertEqual(out["european_knockout_soon"].iloc[0], 1.0)
        self.assertEqual(out["european_knockout_soon"].iloc[1], 0.0)
        self.assertEqual(out["european_knockout_soon"].iloc[2], 1.0)
        self.assertEqual(out["european_knockout_soon"].iloc[3], 0.0)
        # No European football must be "far away", never zero days.
        self.assertEqual(out["european_days_to"].iloc[4], 99.0)
        self.assertEqual(out["european_knockout_soon"].iloc[4], 0.0)

    def test_european_season_dates_resolve_across_the_new_year(self):
        """The source states the year once per file; everything else is inferred.

        Two ways to get this wrong were found and fixed: tracking the year while
        reading drifts forward on every out-of-order section, and a plain
        July-to-June rule dates the COVID-delayed 2019/20 finals to August 2019 —
        on top of that season's own opening Gameweeks.
        """
        import european_fixtures as euro

        for season in ("2018-19", "2019-20", "2023-24"):
            matches = euro.load_european_matches([season], ("cl",))
            self.assertFalse(matches.empty)
            start = int(season.split("-")[0])
            # Every date must sit inside the season it belongs to, never a year out.
            self.assertGreaterEqual(matches["date"].min(), pd.Timestamp(f"{start}-06-01"))
            self.assertLessEqual(matches["date"].max(), pd.Timestamp(f"{start + 1}-09-30"))

        # 2019/20 ran to the Lisbon final tournament in August 2020.
        covid = euro.load_european_matches(["2019-20"], ("cl",))
        self.assertGreater(covid["date"].max(), pd.Timestamp("2020-07-01"))

    def test_absence_run_treats_a_blank_gameweek_as_neither(self):
        """A blank week is not an absence and not a return.

        There was no fixture to miss, so the run must carry through it. Resetting
        on a blank would tell the model an injured player had recovered because
        his club happened to have a free week.
        """
        codes = np.array(["a"] * 6 + ["b"] * 3)
        minutes = np.array([90.0, 0.0, 0.0, 0.0, 90.0, 0.0, 0.0, 90.0, 0.0])
        had_fixture = np.array(
            [True, True, False, True, True, True, True, True, True]
        )
        runs = lens.absence_run_lengths(codes, minutes, had_fixture)
        # index 2 is the blank: the run holds at 1 rather than resetting to 0
        # or counting up to 2.
        self.assertEqual(list(runs[:6]), [0.0, 0.0, 1.0, 1.0, 2.0, 0.0])
        # The count is of *earlier* weeks only, so the first row of any player is
        # always zero — it can never see its own outcome.
        self.assertEqual(runs[0], 0.0)
        self.assertEqual(runs[6], 0.0)
        self.assertEqual(list(runs[6:]), [0.0, 1.0, 0.0])

    def test_calibration_tier_separates_price_and_absence(self):
        """Nine cells: three price bands crossed with three absence bands.

        Price says who is first choice; absence says who is available, and the
        predicted-to-realised map differs completely between them.
        """
        frame = pd.DataFrame(
            {
                "price": [40, 40, 70, 70, 130, 130],
                "position_id": [3] * 6,
                "absence_run": [0.0, 4.0, 0.0, 4.0, 0.0, 1.0],
            }
        )
        tiers = lens.minutes_calibration_tier(frame, ["position_id"])
        # Same price, different absence, must not share a cell.
        self.assertNotEqual(tiers[0], tiers[1])
        self.assertNotEqual(tiers[2], tiers[3])
        # Same absence, different price, must not share a cell either.
        self.assertNotEqual(tiers[0], tiers[2])
        self.assertTrue(all(0 <= tier < len(lens.MINUTES_CALIBRATION_TIERS) for tier in tiers))
        # Without the column the tier falls back to price alone, so a frame built
        # before this axis existed still calibrates rather than crashing.
        legacy = lens.minutes_calibration_tier(
            frame.drop(columns=["absence_run"]), ["position_id"]
        )
        self.assertTrue(all(0 <= tier <= 2 for tier in legacy))

    def test_live_absence_run_uses_only_current_season_events(self):
        """A GW1 starter cannot inherit an absence from last season's GW38."""
        current = pd.DataFrame(
            {
                "id": [411, 999, 1000],
                "team": [1, 1, 2],
                "minutes": [90, 0, 180],
            }
        )
        fixtures = pd.DataFrame(
            {
                "event": [1, 2],
                "team_h": [1, 1],
                "team_a": [2, 3],
            }
        )
        live = {
            1: {
                "elements": [
                    {"id": 411, "stats": {"minutes": 90}},
                    {"id": 999, "stats": {"minutes": 0}},
                    {"id": 1000, "stats": {"minutes": 90}},
                ]
            },
            2: {
                "elements": [
                    {"id": 411, "stats": {"minutes": 0}},
                    {"id": 1000, "stats": {"minutes": 90}},
                ]
            },
        }
        runs = lens.current_absence_runs_from_events(current, fixtures, live, [1, 2])
        self.assertEqual(runs[411], 1.0)
        self.assertEqual(runs[1000], 0.0)
        # No season appearance means neutral, not "missed every week".
        self.assertEqual(runs[999], 0.0)
        self.assertEqual(
            lens.season_label_from_deadline("2026-08-14T17:30:00Z"), "2026-27"
        )

    def test_live_recommendation_skips_a_locked_unfinished_event(self):
        events = pd.DataFrame(
            [
                {
                    "id": 2,
                    "deadline_time": "2026-08-28T17:30:00Z",
                    "finished": False,
                },
                {
                    "id": 3,
                    "deadline_time": "2026-09-04T17:30:00Z",
                    "finished": False,
                },
            ]
        )
        selected = lens.next_recommendation_event(
            events, as_of="2026-08-29T12:00:00Z"
        )
        self.assertEqual(int(selected["id"]), 3)

    def test_gate_does_not_count_candidate_variants_as_extra_seasons(self):
        """Duplicating a correlated candidate path must not shrink gate error."""
        incumbent = lens.GATE_INCUMBENT
        challenger = "central:Joint transfer-chip tree + hold value"
        base = [float((index % 5) - 2) for index in range(38)]
        better = [value + (0.4 if index % 3 else -0.2) for index, value in enumerate(base)]

        def payload(paths: int) -> dict:
            incumbent_stats = [
                {"weeklyPoints": base} for _ in range(paths) for _ in lens.SEASONS
            ]
            challenger_stats = [
                {"weeklyPoints": better} for _ in range(paths) for _ in lens.SEASONS
            ]
            zeros = np.zeros(len(lens.SEASONS))
            return {
                incumbent: (zeros, lens.WEEKLY_CHASE_STRATEGY, None, incumbent_stats),
                challenger: (zeros, lens.JOINT_OPTION_STRATEGY, None, challenger_stats),
            }

        _, single = lens.select_gate_option(payload(1), len(lens.SEASONS))
        _, tripled = lens.select_gate_option(payload(3), len(lens.SEASONS))
        single_result = single["options"][challenger]
        tripled_result = tripled["options"][challenger]
        self.assertEqual(single_result["standardError"], tripled_result["standardError"])
        self.assertEqual(single_result["confidenceVsIncumbent"], tripled_result["confidenceVsIncumbent"])
        self.assertEqual(tripled["candidateVariantsAveraged"], 3)
        self.assertIn("season", tripled["evidenceUnit"])

    def test_gate_pin_holds_the_selection_and_rejects_unknown_names(self):
        """Pinning the gate is the only way to vary the data on its own.

        The gate chooses between options differing by up to 200 points a season,
        and it has moved on changes unrelated to strategy. When it moves at the
        same time as the thing being measured the comparison is confounded, and
        no care elsewhere in the run recovers it.
        """
        options = {
            "central:Six-GW planner + adaptive banking": None,
            "central:Joint transfer-chip tree + hold value": None,
        }
        with mock.patch.object(lens, "GATE_PIN", ""):
            # Unpinned, the incumbent is defended rather than pinned.
            self.assertFalse(lens.GATE_PIN)
        with mock.patch.object(
            lens, "GATE_PIN", "central:Joint transfer-chip tree + hold value"
        ):
            name, detail = lens.select_gate_option(options, len(lens.SEASONS))
            self.assertEqual(name, "central:Joint transfer-chip tree + hold value")
            self.assertTrue(detail["pinned"])
            # The pinned report must carry every key the normal path returns.
            # The first version of this omitted "switched" and "incumbent", which
            # the caller reads to print the decision — so the run died on a
            # KeyError after the expensive replays had already finished, and the
            # test passed anyway because it only checked "pinned".
            for key in (
                "selected",
                "incumbent",
                "switched",
                "seasonsAvailable",
                "seasonsUsed",
                "regimeMatched",
            ):
                self.assertIn(key, detail)
            self.assertFalse(detail["switched"])
        # A typo must fail loudly. Silently falling back to the incumbent would
        # produce a run that looks pinned, is not, and is reported as controlled.
        with mock.patch.object(lens, "GATE_PIN", "central:No such strategy"):
            with self.assertRaises(KeyError):
                lens.select_gate_option(options, len(lens.SEASONS))

    def test_live_path_hardcodes_no_season_label(self):
        """The live deadline must derive its seasons, never name one.

        A hardcoded season is invisible to every behaviour test, because it stays
        correct until the calendar rolls over and then quietly describes the wrong
        year. That is exactly how the live absence run came to read a finished
        season's injury streaks as if they described fit players: one literal was
        copied from the line above it and nothing failed until a new season began.

        Season-specific *rules* elsewhere in the module are legitimate — chip
        allowances and scoring changes really do differ by year. This guards only
        the live recommendation path, which should always be talking about now.
        """
        import inspect
        import re

        source = inspect.getsource(lens.current_recommendation)
        self.assertEqual(re.findall(r'"20\d\d-\d\d"', source), [])
        # And the archive reference must track the data rather than a literal.
        self.assertIn("SEASONS[-1]", source)

    def test_gate_defends_the_standing_choice_not_a_constant(self):
        """The walk-forward gate must have memory, or it oscillates.

        Deciding every season independently against the same fixed constant makes
        an option near the confidence bar flip back and forth: the run reported
        five strategy changes across ten seasons, cycling through four options.
        Defending the previous season's choice means a challenger must beat what
        is actually in force, and reverting must clear the bar again the other
        way.
        """
        weeks = [1.0] * 38
        options = {
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
        challenger = "central:Joint transfer-chip tree + hold value"
        with mock.patch.object(lens, "GATE_PIN", ""):
            # With identical evidence nothing can clear the bar, so whichever
            # option is standing must survive — including one that is not the
            # module-level incumbent.
            held, report = lens.select_gate_option(
                options, len(lens.SEASONS), incumbent_override=challenger
            )
            self.assertEqual(held, challenger)
            self.assertEqual(report["incumbent"], challenger)
            self.assertFalse(report["switched"])
            # An unknown standing name must fall back to the default incumbent
            # rather than crash or silently accept it.
            fallback, _ = lens.select_gate_option(
                options, len(lens.SEASONS), incumbent_override="central:Nonexistent"
            )
            self.assertEqual(fallback, lens.GATE_INCUMBENT)

    def test_one_match_does_not_define_a_rate(self) -> None:
        """The winner's curse: a single haul must move a rate, not replace it."""
        frame = pd.DataFrame(
            {
                "player_key": ["a"] * 4,
                "rate_game": [3.0, np.nan, 0.0, 1.0],
            }
        )
        prior = pd.Series(0.5, index=frame.index)
        shrunk = lens.shrunk_player_rate(frame, "rate_game", prior, 12, 4.0)
        # No history yet: exactly the prior.
        self.assertAlmostEqual(shrunk.iloc[0], 0.5)
        # One appearance of 3.0 against four pseudo-games at 0.5, not 3.0.
        self.assertAlmostEqual(shrunk.iloc[1], (3.0 + 4 * 0.5) / 5)
        # A censored week (no appearance) adds neither a game nor a zero.
        self.assertAlmostEqual(shrunk.iloc[2], shrunk.iloc[1])
        self.assertAlmostEqual(shrunk.iloc[3], (3.0 + 0.0 + 4 * 0.5) / 6)
        # Only earlier rows count, so the value is known at the deadline.
        expanding = lens.shrunk_player_rate(frame, "rate_game", prior, None, 4.0)
        self.assertAlmostEqual(expanding.iloc[3], shrunk.iloc[3])

    def test_decayed_history_weights_recent_matches_more(self) -> None:
        frame = pd.DataFrame(
            {"player_key": ["a"] * 4, "rate_game": [10.0, np.nan, 0.0, 5.0]}
        )
        prior = pd.Series(1.0, index=frame.index)
        shrunk = lens.shrunk_player_rate(frame, "rate_game", prior, None, 2.0, halflife=1.0)
        # Half-life of one row: weights halve per row of age, censored rows
        # still age the history but add no evidence.
        self.assertAlmostEqual(shrunk.iloc[0], 1.0)
        self.assertAlmostEqual(shrunk.iloc[1], (10.0 + 2.0) / (1.0 + 2.0))
        self.assertAlmostEqual(shrunk.iloc[2], (5.0 + 2.0) / (0.5 + 2.0))
        self.assertAlmostEqual(shrunk.iloc[3], (2.5 + 0.0 + 2.0) / (0.25 + 1.0 + 2.0))
        # With no half-life the path is the undecayed mean, unchanged.
        flat = lens.shrunk_player_rate(frame, "rate_game", prior, None, 2.0)
        self.assertAlmostEqual(flat.iloc[3], (10.0 + 0.0 + 2.0) / (2.0 + 2.0))

    def test_price_prior_is_fitted_on_training_seasons_only(self) -> None:
        training = lens.TRAINING_SEASONS[0]
        rows = 300
        price = np.linspace(40, 130, rows)
        frame = pd.DataFrame(
            {
                "season": [training] * rows + ["2025-26"] * rows,
                "position_id": [4] * (2 * rows),
                "price": np.concatenate([price, price]),
            }
        )
        # The evaluation season carries a wildly different relation; if it
        # leaked into the fit the slope would not come back as 0.01.
        target = pd.Series(np.concatenate([0.01 * price, 50 - 0.3 * price]))
        prior = lens.price_informed_prior(frame, target, {4: 0.28})
        self.assertAlmostEqual(prior.iloc[rows + 10], 0.01 * price[10], places=6)
        # Positions without enough training rows keep the flat fallback.
        sparse = frame.iloc[rows - 100 :].assign(position_id=2)
        flat = lens.price_informed_prior(sparse, target.iloc[rows - 100 :], {2: 0.04})
        self.assertAlmostEqual(flat.iloc[0], 0.04)

    def test_live_last_match_minutes_skips_blanks_and_splits_doubles(self) -> None:
        current = pd.DataFrame({"id": [1, 2, 3], "team": [10, 10, 20]})
        fixtures = pd.DataFrame(
            {
                "event": [1, 2, 2, 1],
                "team_h": [10, 10, 10, 20],
                "team_a": [11, 12, 13, 21],
            }
        )
        live = {
            1: {"elements": [{"id": 1, "stats": {"minutes": 90}},
                             {"id": 3, "stats": {"minutes": 12}}]},
            2: {"elements": [{"id": 1, "stats": {"minutes": 150}},
                             {"id": 2, "stats": {"minutes": 0}}]},
        }
        last = lens.current_last_match_minutes(current, fixtures, live, [1, 2])
        # Team 10 played twice in event 2: 150 minutes is 75 a match.
        self.assertAlmostEqual(last[1], 75.0)
        # Scheduled and did not play is a zero, not "no match".
        self.assertAlmostEqual(last[2], 0.0)
        # Team 20 blanked in event 2, so its last match is event 1.
        self.assertAlmostEqual(last[3], 12.0)

    def test_last_match_tier_separates_cameos_from_full_matches(self) -> None:
        frame = pd.DataFrame(
            {
                "position_id": [3, 3, 3, 3],
                "price": [60.0, 60.0, 60.0, 60.0],
                "absence_run": [0.0, 0.0, 0.0, 0.0],
                "last_match_minutes": [90.0, 70.0, 10.0, -1.0],
            }
        )
        with mock.patch.object(lens, "USE_LAST_MATCH_TIER", True),              mock.patch.object(lens, "LAST_MATCH_TIER_VERSION", 1):
            tiers = lens.minutes_calibration_tier(frame, ["position_id"])
        availability = tiers // 3
        self.assertEqual(list(availability), [0, 4, 3, 0])
        with mock.patch.object(lens, "USE_LAST_MATCH_TIER", False):
            legacy = lens.minutes_calibration_tier(frame, ["position_id"])
        self.assertEqual(list(legacy // 3), [0, 0, 0, 0])

    def test_live_form_equals_the_backtest_computation(self) -> None:
        """Appending this season's weeks live must give the batch frame's value."""
        rng = np.random.default_rng(3)
        rows = []
        for key, position, price in (("11", 3, 80.0), ("22", 2, 45.0)):
            for season_order, season in ((0, "s0"), (1, "s1")):
                for gw in range(1, 7):
                    rows.append(
                        {
                            "player_key": key, "season": season, "season_order": season_order,
                            "GW": gw, "position_id": position, "price": price,
                            "points": float(rng.integers(0, 12)), "minutes": 90.0,
                            "ict": float(rng.uniform(0, 15)), "fixture_count": 1,
                        }
                    )
        full = pd.DataFrame(rows)
        full["performance_points"] = full["points"]
        full["underlying_game"] = (full["ict"] / 90 * 90).clip(0, 35)
        # The batch answer: the season-1 GW6 row of a frame that contains it.
        batch = full.copy()
        batch.loc[(batch.season == "s1") & (batch.GW == 6), ["performance_points", "underlying_game"]] = np.nan
        batch = batch.sort_values(["player_key", "season_order", "GW"]).reset_index(drop=True)
        prior = lens.price_informed_prior(batch, batch["performance_points"], lens.FORM_POINTS_PRIOR, bounds=(0.5, 9.0))
        long = lens.shrunk_player_rate(batch, "performance_points", prior, None, lens.PRICE_PRIOR_STRENGTH)
        target = (batch.season == "s1") & (batch.GW == 6)
        # The live answer: season 0 as history, season 1 weeks 1-5 as event rows.
        history = full[full.season == "s0"]
        season_rows = full[(full.season == "s1") & (full.GW <= 5)][
            ["player_key", "GW", "points", "minutes", "ict", "fixture_count"]
        ]
        deadline = pd.DataFrame({"player_key": ["11", "22"], "position_id": [3, 2], "price": [80.0, 45.0]})
        with mock.patch.object(lens, "USE_PRICE_PRIOR", True), mock.patch.object(lens, "HISTORY_HALFLIFE", 0.0):
            live = lens.live_form_history(history, season_rows, deadline).set_index("player_key")
        expected = dict(zip(batch.loc[target, "player_key"], long[target]))
        for key in ("11", "22"):
            self.assertAlmostEqual(live.loc[key, "long_raw"], expected[key], places=10)

    def test_persistent_team_makes_only_moves_that_clear_the_hurdle(self) -> None:
        pool = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "price": [60, 60, 55, 90],
                "position_id": [3, 3, 3, 3],
                "team_id": [10, 11, 12, 13],
                "risk_adjusted_horizon": [20.0, 29.0, 22.0, 40.0],
            }
        )
        held = [
            {"id": 1, "purchase": 60, "price": 60, "position": 3, "team": 10},
            {"id": 9, "purchase": 50, "price": 50, "position": 3, "team": 14},  # injured: not in pool
        ]
        squad, bank, moves = lens.persistent_team_transfers(held, 5, 2, pool, hurdle=5.0)
        # The unavailable player is sold first (valued at -0.30). Selling him
        # raises 50 + 5 in the bank, so the best affordable option is id 3.
        self.assertEqual((moves[0]["out"], moves[0]["in"]), (9, 3))
        # Then 1 -> 2 gains 29 - 20 = 9, which clears a hurdle of 5.
        self.assertEqual((moves[1]["out"], moves[1]["in"]), (1, 2))
        self.assertEqual(bank, 5 + 50 - 55 + 60 - 60)
        self.assertEqual(sorted(player["id"] for player in squad), [2, 3])
        # With a hurdle of 10 the second move is banked instead.
        _, _, cautious = lens.persistent_team_transfers(held, 5, 2, pool, hurdle=10.0)
        self.assertEqual(len(cautious), 1)

    def test_live_team_state_reads_only_passed_ledger_entries(self) -> None:
        import json
        import tempfile

        history = {
            "entries": [
                {"season": "2026/27", "gameweek": 3, "deadline": "2026-09-04T17:30:00Z",
                 "players": [{"id": 1, "price": 5.0}]},  # no ledger: ignored
                {"season": "2026/27", "gameweek": 6, "deadline": "2026-10-10T10:00:00Z", "bank": 7,
                 "freeTransfersNext": 1, "players": [{"id": 2, "price": 6.0, "purchasePrice": 58}]},
            ]
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "pick-history.json"
            path.write_text(json.dumps(history), encoding="utf-8")
            with mock.patch.object(lens, "PICK_HISTORY_PATH", path):
                self.assertIsNone(lens.live_team_state("2026-27", "2026-10-10T10:00:00Z"))
                state = lens.live_team_state("2026-27", "2026-10-17T10:00:00Z")
        self.assertEqual(state["gameweek"], 6)
        self.assertEqual(state["bank"], 7)
        self.assertEqual(state["players"], [{"id": 2, "purchase": 58}])

    def test_live_team_strength_equals_the_backtest_team_model(self) -> None:
        """Appending this season's matches live must give the batch panel's ratings."""
        rng = np.random.default_rng(11)
        rows = []
        for season_order, season in ((0, "s0"), (1, "s1")):
            for gw in range(1, 9):
                for team_id, name in ((1, "Alpha"), (2, "Beta")):
                    scored, conceded = int(rng.integers(0, 4)), int(rng.integers(0, 4))
                    rows.append(
                        {
                            "season": season, "season_order": season_order, "GW": gw,
                            "team_id": team_id, "team_name": name, "team_games": 1,
                            "team_goals": scored, "team_xg": float(rng.uniform(0.3, 2.5)),
                            "team_goals_against": conceded, "team_xga": float(rng.uniform(0.3, 2.5)),
                            "team_clean_sheets": int(conceded == 0),
                            "team_result_points": 3 if scored > conceded else int(scored == conceded),
                        }
                    )
        panel = pd.DataFrame(rows)
        deadline_gw = 6
        # Batch: the season-1 panel up to the deadline, deadline row blank.
        batch = panel[(panel.season == "s0") | (panel.GW <= deadline_gw)].copy()
        blank = (batch.season == "s1") & (batch.GW == deadline_gw)
        batch.loc[blank, ["team_games", "team_goals", "team_xg", "team_goals_against",
                          "team_xga", "team_clean_sheets", "team_result_points"]] = 0
        batch["opponent_team"] = np.nan
        batch["was_home"] = False
        rated = lens.add_causal_team_strength(batch)
        expected = rated[(rated.season == "s1") & (rated.GW == deadline_gw)].set_index("team_id")
        live = lens.live_team_strength(
            panel[panel.season == "s0"],
            panel[(panel.season == "s1") & (panel.GW < deadline_gw)],
            {1: "Alpha", 2: "Beta"},
        )
        for column in lens.LIVE_TEAM_RATINGS:
            for team_id in (1, 2):
                self.assertAlmostEqual(live.loc[team_id, column], expected.loc[team_id, column], places=10)

    def test_a_player_without_minutes_this_season_keeps_his_prior_rate(self) -> None:
        """Numerator and denominator must describe the same sample."""
        minutes = pd.Series([0.0, 450.0])
        prior = 0.44
        season_goals_signal = pd.Series([0.0, 3.0])
        rate = (season_goals_signal + 5 * prior) / lens.live_rate_denominator(minutes)
        # No minutes this season: the rate is the prior, not a fraction of it.
        self.assertAlmostEqual(rate.iloc[0], prior)
        # Five nineties this season: an even blend of the season and the prior.
        self.assertAlmostEqual(rate.iloc[1], (3.0 + 5 * prior) / 10)

    def test_current_season_form_rows_read_official_event_data(self) -> None:
        current = pd.DataFrame({"id": [1, 2], "code": [111, 222], "team": [10, 20]})
        fixtures = pd.DataFrame({"event": [1, 2, 2], "team_h": [10, 10, 10], "team_a": [20, 11, 12]})
        live = {
            1: {"elements": [{"id": 1, "stats": {"total_points": 6, "minutes": 90, "ict_index": "7.5"}},
                             {"id": 2, "stats": {"total_points": 1, "minutes": 20, "ict_index": "1.0"}}]},
            2: {"elements": [{"id": 1, "stats": {"total_points": 9, "minutes": 180, "ict_index": "12"}}]},
        }
        rows = lens.current_season_form_rows(current, fixtures, live).set_index(["player_key", "GW"])
        self.assertEqual(rows.loc[("111", 2), "fixture_count"], 2)   # double Gameweek
        self.assertEqual(rows.loc[("222", 1), "fixture_count"], 1)
        self.assertAlmostEqual(rows.loc[("111", 1), "ict"], 7.5)

    def test_version_two_gives_debutants_their_own_tier(self) -> None:
        frame = pd.DataFrame(
            {
                "position_id": [3] * 5,
                "price": [60.0, 60.0, 60.0, 60.0, 90.0],
                "absence_run": [0.0] * 5,
                "last_match_minutes": [90.0, 70.0, 45.0, 10.0, -1.0],
            }
        )
        with mock.patch.object(lens, "USE_LAST_MATCH_TIER", True), \
             mock.patch.object(lens, "LAST_MATCH_TIER_VERSION", 2):
            tiers = lens.minutes_calibration_tier(frame, ["position_id"])
        self.assertEqual(list(tiers // 3), [0, 4, 5, 3, 6])
        # Debutants are pooled across price: a GBP9.0m debutant shares a cell
        # with a GBP4.0m one rather than borrowing a nailed starter's map.
        self.assertEqual(int(tiers[4] % 3), 0)
        self.assertTrue(all(tier < 21 for tier in tiers))


if __name__ == "__main__":
    unittest.main()
