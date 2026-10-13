"""Tests for the hourly live job's decision rule."""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import live_job

DEADLINE = datetime(2026, 10, 17, 10, 0, tzinfo=timezone.utc)


def at(hours_before: float) -> datetime:
    return DEADLINE - timedelta(hours=hours_before)


class DecideTests(unittest.TestCase):
    def test_a_fresh_draft_far_from_the_deadline_is_left_alone(self):
        self.assertFalse(live_job.decide(at(48), DEADLINE, at(50)).refresh)

    def test_a_missing_or_day_old_draft_is_refreshed(self):
        self.assertTrue(live_job.decide(at(48), DEADLINE, None).refresh)
        self.assertTrue(live_job.decide(at(48), DEADLINE, at(70)).refresh)

    def test_the_deadline_pick_is_taken_once_inside_the_final_window(self):
        plan = live_job.decide(at(2.5), DEADLINE, at(20))
        self.assertTrue(plan.refresh)
        self.assertIn("deadline pick", plan.reason)
        # Already recorded inside the window: the next hourly run does nothing.
        self.assertFalse(live_job.decide(at(1.5), DEADLINE, at(2.5)).refresh)

    def test_nothing_is_recorded_after_the_deadline(self):
        self.assertFalse(live_job.decide(at(-0.5), DEADLINE, at(20)).refresh)
        self.assertFalse(live_job.decide(at(-0.5), None, None).refresh)


if __name__ == "__main__":
    unittest.main()


class ScorerOddsTests(unittest.TestCase):
    def test_only_the_yes_side_of_each_scorer_price_is_kept(self):
        import scorer_odds

        payload = {
            "matches": [
                {
                    "home_team": "Arsenal",
                    "away_team": "Chelsea",
                    "commence_time": "2026-10-17T14:00:00Z",
                    "bookmakers": [
                        {
                            "key": "draftkings",
                            "markets": [
                                {
                                    "key": "player_goal_scorer_anytime",
                                    "outcomes": [
                                        {"name": "Yes", "description": "Bukayo Saka", "price": 2.6},
                                        {"name": "No", "description": "Bukayo Saka", "price": 1.45},
                                        {"name": "Yes", "description": "Cole Palmer", "price": 2.9},
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ]
        }
        rows = scorer_odds.player_prices(payload)
        self.assertEqual([(r["player"], r["price"]) for r in rows], [("Bukayo Saka", 2.6), ("Cole Palmer", 2.9)])

    def test_the_job_never_fails_a_pick_over_odds(self):
        from unittest import mock

        with mock.patch.object(live_job.scorer_odds, "api_key", return_value="k"), \
             mock.patch.object(live_job.scorer_odds, "coming_gameweek", side_effect=OSError("offline")):
            live_job.capture_scorer_odds(deadline_pick=True)  # must not raise

    def test_without_a_key_nothing_is_called(self):
        from unittest import mock

        with mock.patch.object(live_job.scorer_odds, "api_key", return_value=None), \
             mock.patch.object(live_job.scorer_odds, "coming_gameweek") as coming:
            live_job.capture_scorer_odds(deadline_pick=True)
        coming.assert_not_called()
