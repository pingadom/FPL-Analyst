"""Integrity tests for the prospective deadline-to-scoring pipeline."""

from __future__ import annotations

import json
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prospective_common import APP_DATA, ROOT, payload_hash


class ProspectivePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.deadline = json.loads((APP_DATA / "deadline-status.json").read_text(encoding="utf-8"))
        cls.snapshot = json.loads((ROOT / cls.deadline["snapshotPath"]).read_text(encoding="utf-8"))
        cls.shadow = json.loads((APP_DATA / "shadow-status.json").read_text(encoding="utf-8"))
        cls.decision = json.loads((ROOT / cls.shadow["decisionPath"]).read_text(encoding="utf-8"))
        cls.chips = json.loads((APP_DATA / "chip-scenarios.json").read_text(encoding="utf-8"))
        cls.frontier = json.loads((APP_DATA / "frontier-scores.json").read_text(encoding="utf-8"))
        cls.listwise = json.loads((APP_DATA / "listwise-scores.json").read_text(encoding="utf-8"))
        cls.performance = json.loads((APP_DATA / "performance-progress.json").read_text(encoding="utf-8"))

    def test_snapshot_hash_and_schedule_fingerprint(self) -> None:
        unhashed = {key: value for key, value in self.snapshot.items() if key != "snapshotHash"}
        self.assertEqual(payload_hash(unhashed), self.snapshot["snapshotHash"])
        self.assertEqual(self.snapshot["snapshotHash"], self.deadline["snapshotHash"])
        self.assertEqual(
            payload_hash(self.snapshot["official"]["fixtures"]),
            self.deadline["scheduleFingerprint"],
        )

    def test_decision_hash_and_common_snapshot(self) -> None:
        unhashed = {key: value for key, value in self.decision.items() if key != "decisionHash"}
        self.assertEqual(payload_hash(unhashed), self.decision["decisionHash"])
        self.assertEqual(self.decision["snapshotHash"], self.deadline["snapshotHash"])
        self.assertEqual(self.chips["snapshotHash"], self.deadline["snapshotHash"])

    def test_three_legal_shadow_squads(self) -> None:
        self.assertEqual(len(self.decision["managers"]), 3)
        for manager in self.decision["managers"]:
            squad = manager["squad"]
            self.assertEqual(len(squad), 15)
            self.assertEqual(len({row["id"] for row in squad}), 15)
            self.assertEqual(
                Counter(row["position"] for row in squad),
                Counter({"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}),
            )
            self.assertLessEqual(max(Counter(row["team"] for row in squad).values()), 3)
            self.assertLessEqual(sum(float(row["price"]) for row in squad), 100.0001)
            self.assertEqual(len(manager["xiIds"]), 11)
            self.assertIn(manager["captainId"], manager["xiIds"])
            self.assertIn(manager["viceId"], manager["xiIds"])

    def test_chip_recommendation_passes_its_gate(self) -> None:
        if self.chips["recommendation"] != "Hold":
            selected = next(
                row for row in self.chips["scenarios"]
                if row["chip"] == self.chips["recommendation"]
            )
            self.assertTrue(selected["gatePassed"])

    def test_provisional_cycle_cannot_claim_results(self) -> None:
        if self.deadline["status"] == "provisional":
            self.assertEqual(self.decision["decisionStatus"], "provisional")
            self.assertEqual(self.shadow["completedGameweeks"], 0)

    def test_challenger_remains_unpromoted(self) -> None:
        self.assertEqual(self.frontier["status"], "shadow challenger")
        self.assertIn("prospective", self.frontier["promotionRule"].lower())
        self.assertGreater(len(self.frontier["players"]), 300)
        self.assertLessEqual(len(self.frontier["players"]), self.deadline["playersTracked"])
        self.assertEqual(self.listwise["status"], "shadow challenger")
        self.assertEqual(len(self.listwise["players"]), len(self.frontier["players"]))
        self.assertEqual(self.performance["targetHits"], 0)
        self.assertGreater(self.performance["stackLift"], 0)
        self.assertIn("cannot promote", self.performance["governance"].lower())


if __name__ == "__main__":
    unittest.main()
