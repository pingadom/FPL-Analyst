"""Tests for the prospective pick log.

The scoring path only runs once a Gameweek finishes, so without these it would
sit unexercised until the first real score — and a bug there would corrupt the
one record in this project that is genuine out-of-sample evidence.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pick_history


def live_payload(points: dict[int, tuple[int, int]]) -> dict:
    """{element: (points, minutes)} in the shape the official endpoint returns."""
    return {
        "elements": [
            {"id": element, "stats": {"total_points": scored, "minutes": played}}
            for element, (scored, played) in points.items()
        ]
    }


def entry(players: list[dict], gameweek: int = 5) -> dict:
    return {
        "season": "2026/27",
        "gameweek": gameweek,
        "formation": "3-4-3",
        "captain": "Captain",
        "vice": "Vice",
        "projectedPoints": 50.0,
        "players": players,
        "scored": False,
    }


def player(element: int, starter: bool, **flags) -> dict:
    return {
        "id": element,
        "name": f"P{element}",
        "position": flags.get("position", "MID"),
        "price": 5.0,
        "starter": starter,
        "captain": flags.get("captain", False),
        "vice": flags.get("vice", False),
        "projected": flags.get("projected", 4.0),
    }


class PickHistoryScoringTests(unittest.TestCase):
    def _score(self, players, points, gameweek=5):
        history = {"schemaVersion": 1, "entries": [entry(players, gameweek)]}
        saved: dict = {}
        with mock.patch.object(pick_history, "load_history", return_value=history), \
             mock.patch.object(pick_history, "save_history", saved.update), \
             mock.patch.object(
                 pick_history,
                 "_get",
                 side_effect=lambda url: (
                     {"events": [{"id": gameweek, "finished": True}]}
                     if "bootstrap" in url
                     else live_payload(points)
                 ),
             ):
            pick_history.score()
        return history["entries"][0]["realised"]

    def test_captain_points_are_counted_twice(self):
        players = [
            player(1, True, captain=True),
            player(2, True),
            player(3, False),
        ]
        realised = self._score(players, {1: (10, 90), 2: (4, 90), 3: (7, 90)})
        # XI is 10 + 4; the captain contributes one extra copy of his own score.
        self.assertEqual(realised["xiPoints"], 14)
        self.assertEqual(realised["captainBonus"], 10)
        self.assertEqual(realised["total"], 24)
        # Bench is reported but never added to the total.
        self.assertEqual(realised["benchPoints"], 7)

    def test_armband_passes_to_the_vice_when_the_captain_does_not_play(self):
        players = [
            player(1, True, captain=True),
            player(2, True, vice=True),
            player(3, False),
        ]
        realised = self._score(players, {1: (0, 0), 2: (6, 90), 3: (2, 90)})
        # A captain with no minutes scores nothing and the armband moves, which is
        # the actual FPL rule rather than a silent zero.
        self.assertEqual(realised["captainBonus"], 6)
        # The absent captain is also autosubbed: the bench player's 2 counts.
        self.assertEqual(realised["total"], 6 + 6 + 2)

    def test_a_starter_who_did_not_play_is_replaced_from_the_bench(self):
        players = [player(1, True), player(2, True), player(3, False)]
        realised = self._score(players, {1: (0, 0), 2: (5, 90), 3: (9, 90)})
        self.assertEqual(realised["startersWhoDidNotPlay"], 1)
        # FPL brings the first bench player who played on, as the real game does.
        self.assertEqual(realised["total"], 5 + 9)
        self.assertEqual(realised["benchPoints"], 0)
        self.assertEqual(realised["autosubs"], [{"out": "P1", "in": "P3"}])

    def test_bench_players_who_did_not_play_are_skipped_in_order(self):
        players = [
            player(1, True),
            player(2, True),
            player(3, False, projected=6.0),
            player(4, False, projected=3.0),
        ]
        realised = self._score(players, {1: (0, 0), 2: (5, 90), 3: (0, 0), 4: (2, 30)})
        self.assertEqual(realised["autosubs"], [{"out": "P1", "in": "P4"}])
        self.assertEqual(realised["total"], 5 + 2)

    def test_a_substitution_cannot_break_the_minimum_formation(self):
        defenders = [player(10 + i, True, position="DEF") for i in range(3)]
        players = [
            player(1, True, position="GK"),
            *defenders,
            player(2, True),
            player(3, True),
            player(4, True, position="FWD"),
            player(5, False, position="MID", projected=6.0),  # first on the bench
            player(6, False, position="DEF", projected=2.0),
        ]
        points = {pid: (2, 90) for pid in (1, 2, 3, 4, 11, 12)}
        points.update({10: (0, 0), 5: (8, 90), 6: (1, 90)})
        realised = self._score(players, points)
        # Only three defenders started, so the defender must come on, not the midfielder.
        self.assertEqual(realised["autosubs"], [{"out": "P10", "in": "P6"}])

    def test_a_goalkeeper_is_only_replaced_by_a_goalkeeper(self):
        players = [
            player(1, True, position="GK"),
            player(2, True),
            player(3, False, position="GK", projected=1.0),
            player(4, False, projected=9.0),
        ]
        realised = self._score(players, {1: (0, 0), 2: (5, 90), 3: (3, 90), 4: (7, 90)})
        self.assertEqual(realised["autosubs"], [{"out": "P1", "in": "P3"}])
        self.assertEqual(realised["total"], 5 + 3)

    def test_a_gameweek_is_not_scored_until_fpl_confirms_the_points(self):
        players = [player(1, True, captain=True)]
        history = {"schemaVersion": 1, "entries": [entry(players, 5)]}
        with mock.patch.object(pick_history, "load_history", return_value=history),              mock.patch.object(pick_history, "save_history", lambda payload: None),              mock.patch.object(
                 pick_history,
                 "_get",
                 return_value={"events": [{"id": 5, "finished": True, "data_checked": False}]},
             ):
            pick_history.score()
        self.assertFalse(history["entries"][0]["scored"])

    def _score_with_chip(self, chip, players, points):
        history = {"schemaVersion": 1, "entries": [dict(entry(players, 5), chip=chip)]}
        with mock.patch.object(pick_history, "load_history", return_value=history), \
             mock.patch.object(pick_history, "save_history", lambda payload: None), \
             mock.patch.object(
                 pick_history,
                 "_get",
                 side_effect=lambda url: (
                     {"events": [{"id": 5, "finished": True}]}
                     if "bootstrap" in url
                     else live_payload(points)
                 ),
             ):
            pick_history.score()
        return history["entries"][0]["realised"]

    def test_bench_boost_counts_the_bench(self):
        players = [player(1, True, captain=True), player(2, False)]
        realised = self._score_with_chip("Bench Boost", players, {1: (5, 90), 2: (7, 90)})
        self.assertEqual(realised["total"], 5 + 5 + 7)

    def test_triple_captain_counts_the_captain_three_times(self):
        players = [player(1, True, captain=True), player(2, True)]
        realised = self._score_with_chip("Triple Captain", players, {1: (8, 90), 2: (2, 90)})
        self.assertEqual(realised["total"], 8 * 3 + 2)

    def test_an_unfinished_gameweek_is_left_alone(self):
        players = [player(1, True, captain=True)]
        history = {"schemaVersion": 1, "entries": [entry(players, 9)]}
        with mock.patch.object(pick_history, "load_history", return_value=history), \
             mock.patch.object(pick_history, "save_history", lambda payload: None), \
             mock.patch.object(
                 pick_history,
                 "_get",
                 return_value={"events": [{"id": 9, "finished": False}]},
             ):
            pick_history.score()
        self.assertFalse(history["entries"][0]["scored"])
        self.assertNotIn("realised", history["entries"][0])


class PickHistoryRecordTests(unittest.TestCase):
    ARTIFACT = {
        "headline": {
            "season": "2026/27",
            "gameweek": 6,
            "deadline": "2026-10-10T10:00:00Z",
            "formation": "3-5-2",
            "captain": "Captain",
        },
        "squad": [player(1, True, captain=True)],
    }

    def _record(self, now):
        saved: dict = {}
        with mock.patch.object(pick_history.ARTIFACT.__class__, "read_text", return_value=json.dumps(self.ARTIFACT)), \
             mock.patch.object(pick_history, "load_history", return_value={"schemaVersion": 1, "entries": []}), \
             mock.patch.object(pick_history, "save_history", saved.update):
            pick_history.record(now=now)
        return saved

    def test_a_pick_before_the_deadline_is_recorded(self):
        from datetime import datetime, timezone
        saved = self._record(datetime(2026, 10, 9, 18, 0, tzinfo=timezone.utc))
        self.assertEqual(saved["entries"][0]["gameweek"], 6)

    def test_a_pick_after_the_deadline_is_refused(self):
        """A squad chosen after kick-off knows how the week began; it is not evidence."""
        from datetime import datetime, timezone
        with self.assertRaises(RuntimeError) as caught:
            self._record(datetime(2026, 10, 10, 10, 0, tzinfo=timezone.utc))
        self.assertIn("has passed", str(caught.exception))


class PickHistoryProvenanceTests(unittest.TestCase):
    def test_backfill_refuses_an_artifact_written_after_its_deadline(self):
        """The rule that makes the log evidence rather than a scrapbook."""
        artifact = json.dumps(
            {
                "headline": {
                    "season": "2026/27",
                    "gameweek": 1,
                    "deadline": "2026-08-21T17:30:00Z",
                    "formation": "3-4-3",
                    "captain": "Someone",
                },
                "squad": [player(1, True)],
            }
        )
        with mock.patch.object(pick_history.subprocess, "run") as run:
            run.side_effect = [
                mock.Mock(stdout=artifact),
                mock.Mock(stdout="2026-08-22T15:19:10+01:00"),
            ]
            with self.assertRaises(RuntimeError) as caught:
                pick_history.backfill("deadbeef")
        self.assertIn("after the GW1 deadline", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
