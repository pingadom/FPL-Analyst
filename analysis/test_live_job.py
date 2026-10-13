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
