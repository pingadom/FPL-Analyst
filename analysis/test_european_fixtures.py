"""Tests for the European fixture calendar's fixturedownload.com source."""

from __future__ import annotations

import json
import sys
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import european_fixtures as ef

FEED = json.dumps(
    [
        {"DateUtc": "2026-09-16 19:00:00Z", "HomeTeam": "Arsenal", "AwayTeam": "Bayern"},
        {"DateUtc": "2026-09-17 16:45:00Z", "HomeTeam": "Porto", "AwayTeam": "Man Utd"},
        {"DateUtc": "2026-09-17 19:00:00Z", "HomeTeam": "Villarreal", "AwayTeam": "Celta"},
        {"DateUtc": "", "HomeTeam": "Arsenal", "AwayTeam": "TBC"},
    ]
)


class FixtureDownloadTests(unittest.TestCase):
    def test_only_english_clubs_are_kept_with_their_match_dates(self):
        rows = ef.parse_fixturedownload(FEED)
        self.assertEqual(rows, [(date(2026, 9, 16), "arsenal"), (date(2026, 9, 17), "manutd")])

    def test_a_club_whose_name_merely_contains_an_english_word_is_not_english(self):
        # "Villarreal" normalises to "villarreal", not Aston Villa.
        self.assertNotIn("villarreal", ef.ENGLISH_CLUB_KEYS)
        self.assertEqual(ef.club_key("Man City"), "mancity")

    def test_the_live_season_reads_from_fixturedownload(self):
        with mock.patch.object(ef, "_download_fixturedownload", return_value=FEED) as fetch, \
             mock.patch.object(ef, "_download") as openfootball:
            frame = ef.load_european_matches(["2026-27"], ("cl",))
        fetch.assert_called_once_with("2026-27", "cl")
        openfootball.assert_not_called()
        self.assertEqual(sorted(frame["club"]), ["arsenal", "manutd"])

    def test_a_failed_refresh_falls_back_to_the_cached_feed(self):
        with mock.patch.object(ef, "EUROPEAN_CACHE", Path(self._tmp())):
            target = ef.EUROPEAN_CACHE / "2026-27-cl.json"
            target.write_text(FEED, encoding="utf-8")
            # Make the cached copy stale so a refresh is attempted.
            import os
            os.utime(target, (0, 0))
            with mock.patch.object(ef.urllib.request, "urlopen", side_effect=OSError("offline")):
                self.assertEqual(ef._download_fixturedownload("2026-27", "cl"), FEED)

    def _tmp(self) -> str:
        import tempfile

        directory = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(directory, ignore_errors=True))
        return directory


if __name__ == "__main__":
    unittest.main()
