import unittest

import numpy as np
import pandas as pd

from calibrate_model import (
    assert_legal_squad,
    choose_xi,
    fixture_integrity_audit,
    initial_squad,
)


class HistoricalFreeHitTests(unittest.TestCase):
    def test_real_world_transfer_can_create_temporary_four_player_club(self) -> None:
        positions = [1, 1] + [2] * 5 + [3] * 5 + [4] * 3
        squad = {
            element: {
                "position": position,
                "team": 1 if element <= 4 else element,
            }
            for element, position in enumerate(positions, start=1)
        }

        assert_legal_squad(
            squad,
            bank=0,
            season="test",
            gw=21,
            stage="held after real-world transfer",
            allow_temporary_club_overload=True,
        )
        with self.assertRaises(AssertionError):
            assert_legal_squad(
                squad,
                bank=0,
                season="test",
                gw=21,
                stage="after manager transfer",
            )

    def test_sixty_percent_no_fixture_is_valid_for_eight_club_slate(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "season": "test",
                    "GW": 29,
                    "team_id": team,
                    "element": team * 10 + player,
                    "fixture_count": 1 if team <= 8 else 0,
                    "team_games": 1 if team <= 8 else 0,
                }
                for team in range(1, 21)
                for player in range(2)
            ]
        )

        audit = fixture_integrity_audit(frame)

        self.assertTrue(audit["passed"])
        self.assertEqual(audit["playerVsClubFixtureMismatches"], 0)
        self.assertEqual(audit["massBlankRounds"][0]["activeClubs"], 8)
        self.assertEqual(audit["massBlankRounds"][0]["noFixtureShare"], 60.0)

    def test_mass_blank_free_hit_fields_eleven_without_forced_spend(self) -> None:
        rows: list[dict] = []
        quotas = {1: 4, 2: 12, 3: 12, 4: 8}
        element = 1
        for position, count in quotas.items():
            for number in range(count):
                active = number < max(3, count // 2)
                rows.append(
                    {
                        "element": element,
                        "position_id": position,
                        "team_id": (number % 8) + 1 if active else 20 + number % 8,
                        "price": 40 + position * 2 + number % 4,
                        "play_probability": 0.90 if active else 0.88,
                        "component_xpts": 5.0 - number * 0.04,
                        "fixture_count": 1 if active else 0,
                        "season": "test",
                        "GW": 29,
                    }
                )
                element += 1
        frame = pd.DataFrame(rows)
        scores = frame["component_xpts"].to_numpy(float)
        excluded = set(
            frame.loc[frame["fixture_count"].eq(0), "element"].astype(int)
        )

        picked = initial_squad(
            frame,
            scores,
            budget_limit=1000,
            excluded_elements=excluded,
            captain_weight=1.0,
            bench_weight=0.08,
            minimum_spend_gap=None,
            bench_premium_limit=20,
            exact_optimiser=True,
            lineup_scores=scores,
            captain_utility_scores=scores,
        )
        squad = {
            int(frame.loc[index, "element"]): {
                "position": int(frame.loc[index, "position_id"])
            }
            for index in picked
        }
        row_by_element = {
            int(row.element): int(index) for index, row in frame.iterrows()
        }
        xi, _ = choose_xi(squad, row_by_element, scores, excluded)

        self.assertEqual(len(picked), 15)
        self.assertEqual(len(xi), 11)
        self.assertFalse(set(xi) & excluded)
        self.assertLess(sum(int(frame.loc[index, "price"]) for index in picked), 995)


if __name__ == "__main__":
    unittest.main()
