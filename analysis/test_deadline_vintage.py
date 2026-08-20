from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import deadline_vintage as vintage


class DeadlineVintageTests(unittest.TestCase):
    def test_exact_predeadline_source_is_eligible(self) -> None:
        source = vintage.VintageSource(
            source_id="test",
            source_type="official",
            captured_at="2026-08-20T12:00:00Z",
            deadline_at="2026-08-21T17:30:00Z",
            timing="exact-capture",
            path="unused",
            sha256="",
            fields=("fixtures", "prices"),
            decision_status="locked",
        )
        self.assertTrue(source.captured_before_deadline)
        self.assertTrue(source.promotion_eligible)

    def test_realised_and_closing_fields_are_rejected(self) -> None:
        source = vintage.VintageSource(
            source_id="test",
            source_type="market",
            captured_at="2026-08-20T12:00:00Z",
            deadline_at="2026-08-21T17:30:00Z",
            timing="exact-capture",
            path="unused",
            sha256="",
            fields=("AvgCH", "points"),
            decision_status="locked",
        )
        self.assertEqual(set(source.forbidden_fields), {"AvgCH", "points"})
        self.assertFalse(source.promotion_eligible)

    def test_unknown_archive_timing_is_research_only(self) -> None:
        source = vintage.VintageSource(
            source_id="test",
            source_type="market",
            captured_at=None,
            deadline_at=None,
            timing="archive-first-odds-unknown-capture",
            path="unused",
            sha256="",
            fields=("home_odds",),
            decision_status="research-only",
        )
        self.assertFalse(source.promotion_eligible)

    def test_provisional_exact_capture_can_shadow_but_not_promote(self) -> None:
        source = vintage.VintageSource(
            source_id="test",
            source_type="official",
            captured_at="2026-08-20T12:00:00Z",
            deadline_at="2026-08-21T17:30:00Z",
            timing="exact-capture",
            path="unused",
            sha256="",
            fields=("fixtures",),
            decision_status="provisional",
        )
        self.assertTrue(source.shadow_eligible)
        self.assertFalse(source.promotion_eligible)


if __name__ == "__main__":
    unittest.main()
