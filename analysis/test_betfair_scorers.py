"""Tests for the Betfair anytime-scorer loader.

The data arrives from an account download and cannot be fetched in a test, so
these build stream files by hand in the documented BASIC format. What matters
most is the timestamp rule: a price after the deadline, or after the market
turns in-play, must never be used.
"""

from __future__ import annotations

import json
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import betfair_scorers as bf


def ms(text: str) -> int:
    return int(datetime.fromisoformat(text).replace(tzinfo=timezone.utc).timestamp() * 1000)


def definition(in_play: bool = False, market_type: str = "TO_SCORE") -> dict:
    return {
        "eventName": "Arsenal v Leicester",
        "marketType": market_type,
        "marketTime": "2017-08-11T18:45:00.000Z",
        "inPlay": in_play,
        "status": "OPEN",
        "runners": [
            {"id": 1, "name": "Alexandre Lacazette"},
            {"id": 2, "name": "Jamie Vardy"},
        ],
    }


def stream(*messages: dict) -> list[str]:
    return [json.dumps(message) for message in messages]


class ParseTests(unittest.TestCase):
    def test_prices_after_the_market_goes_in_play_are_dropped(self):
        lines = stream(
            {"pt": ms("2017-08-10T12:00:00"), "mc": [{"id": "1.1", "marketDefinition": definition(), "rc": [{"id": 1, "ltp": 2.5}]}]},
            {"pt": ms("2017-08-11T18:46:00"), "mc": [{"id": "1.1", "marketDefinition": definition(in_play=True)}]},
            {"pt": ms("2017-08-11T19:10:00"), "mc": [{"id": "1.1", "rc": [{"id": 1, "ltp": 1.01}]}]},
        )
        rows = bf.parse_market(lines)
        self.assertEqual([row["ltp"] for row in rows], [2.5])
        self.assertEqual(rows[0]["runner_name"], "Alexandre Lacazette")

    def test_other_market_types_are_ignored(self):
        lines = stream(
            {"pt": ms("2017-08-10T12:00:00"), "mc": [{"id": "1.2", "marketDefinition": definition(market_type="MATCH_ODDS"), "rc": [{"id": 1, "ltp": 1.5}]}]},
        )
        self.assertEqual(bf.parse_market(lines), [])


class NameTests(unittest.TestCase):
    def test_accents_and_underscores_normalise(self):
        self.assertEqual(bf.person_key("Martin Ødegaard"), "martinodegaard")
        self.assertEqual(bf.person_key("Mohamed_Salah"), "mohamedsalah")

    def test_betfair_club_spellings_meet_fpl_ones(self):
        self.assertEqual(bf.team_key("Tottenham"), bf.team_key("Spurs"))
        self.assertEqual(bf.team_key("Man Utd"), bf.team_key("Manchester United"))
        self.assertEqual(bf.team_key("Wolves"), bf.team_key("Wolverhampton Wanderers"))

    def test_runner_matching_prefers_unambiguous_names(self):
        players = pd.DataFrame(
            {
                "element": [10, 11, 12],
                "full_key": ["mohamedsalahhamedghaly", "jamesmilner", "andrewrobertson"],
                "web_key": ["salah", "milner", "robertson"],
                "surname_key": ["ghaly", "milner", "robertson"],
                "first_key": ["mohamed", "james", "andrew"],
            }
        )
        self.assertEqual(bf.match_runner("mohamedsalah", players), 10)  # via web name
        self.assertEqual(bf.match_runner("jamesmilner", players), 11)  # full name
        self.assertEqual(bf.match_runner("andyrobertson", players), 12)  # surname
        self.assertIsNone(bf.match_runner("someoneelse", players))


class DeadlineTests(unittest.TestCase):
    def _fixtures(self) -> pd.DataFrame:
        kickoff = pd.Timestamp("2017-08-11T18:45:00Z")
        return pd.DataFrame(
            {
                "season": ["2017-18"] * 2,
                "GW": [1, 1],
                "element": [100, 200],
                "fixture": [1, 1],
                "kickoff": [kickoff] * 2,
                "deadline": [kickoff - bf.DEADLINE_LEAD] * 2,
                "home_key": [bf.team_key("Arsenal")] * 2,
                "away_key": [bf.team_key("Leicester")] * 2,
                "full_key": ["alexandrelacazette", "jamievardy"],
                "web_key": ["lacazette", "vardy"],
                "surname_key": ["lacazette", "vardy"],
                "first_key": ["alexandre", "jamie"],
                "goals_scored": [1, 1],
                "minutes": [90, 90],
            }
        )

    def _updates(self, rows) -> pd.DataFrame:
        frame = pd.DataFrame(rows, columns=["runner_id", "runner_name", "published", "ltp"])
        frame["market_id"] = "1.1"
        frame["market_time"] = pd.Timestamp("2017-08-11T18:45:00Z")
        frame["published"] = pd.to_datetime(frame["published"], utc=True)
        frame["home_key"] = bf.team_key("Arsenal")
        frame["away_key"] = bf.team_key("Leicester City")
        frame["runner_key"] = frame["runner_name"].map(bf.person_key)
        return frame

    def test_the_last_price_before_the_deadline_is_used_not_the_kickoff_price(self):
        updates = self._updates(
            [
                (1, "Alexandre Lacazette", "2017-08-10T12:00:00Z", 2.5),
                (1, "Alexandre Lacazette", "2017-08-11T17:00:00Z", 2.2),  # after the 17:15 deadline? no: before
                (1, "Alexandre Lacazette", "2017-08-11T18:30:00Z", 1.8),  # after the deadline: unusable
            ]
        )
        prices = bf.deadline_prices(updates, self._fixtures())
        self.assertEqual(len(prices), 1)
        self.assertAlmostEqual(prices["market_score_probability"].iloc[0], 1 / 2.2)
        self.assertEqual(prices["element"].iloc[0], 100)

    def test_a_stale_price_is_no_price(self):
        updates = self._updates([(2, "Jamie Vardy", "2017-08-01T12:00:00Z", 3.0)])
        self.assertTrue(bf.deadline_prices(updates, self._fixtures()).empty)


if __name__ == "__main__":
    unittest.main()
