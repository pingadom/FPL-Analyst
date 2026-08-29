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
        "position": "MID",
        "price": 5.0,
        "starter": starter,
        "captain": flags.get("captain", False),
        "vice": flags.get("vice", False),
        "projected": 4.0,
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
        self.assertEqual(realised["total"], 6 + 6)

    def test_a_blanking_starter_is_counted_not_substituted(self):
        players = [player(1, True), player(2, True), player(3, False)]
        realised = self._score(players, {1: (0, 0), 2: (5, 90), 3: (9, 90)})
        self.assertEqual(realised["startersWhoDidNotPlay"], 1)
        # Autosubs are not simulated, so the bench nine stays on the bench and the
        # total is a floor. Silently promoting it would invent a result.
        self.assertEqual(realised["total"], 5)
        self.assertEqual(realised["benchPoints"], 9)
        self.assertIn("autosub", realised["note"])

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
