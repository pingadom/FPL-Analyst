"""Integrity tests for the prospective deadline-to-scoring pipeline."""

from __future__ import annotations

import json
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prospective_common import (
    APP_DATA,
    ROOT,
    available_squad_budget,
    chip_inventory_key,
    chip_set_for_gameweek,
    optimise_squad,
    payload_hash,
    used_chip_keys,
)
from chip_scenario_planner import currently_available_chips, manager_inventory
import calibrate_model as lens
from probabilistic_component_challenger import FEATURES as ROUTE_FEATURES


class ProspectivePipelineTests(unittest.TestCase):
    def test_two_chip_sets_have_distinct_inventory_keys(self) -> None:
        self.assertEqual(chip_set_for_gameweek(19), 1)
        self.assertEqual(chip_set_for_gameweek(20), 2)
        self.assertNotEqual(
            chip_inventory_key("Triple Captain", 19),
            chip_inventory_key("Triple Captain", 20),
        )
        legacy = {
            "chipsUsed": [
                {"chip": "Triple Captain", "gameweek": 6},
                {"chip": "Triple Captain", "gameweek": 26},
            ]
        }
        self.assertEqual(
            used_chip_keys(legacy),
            {"Triple Captain:H1", "Triple Captain:H2"},
        )

    def test_temporarily_unavailable_free_hit_remains_in_inventory(self) -> None:
        remaining = manager_inventory(None, 1)
        self.assertIn("Free Hit", remaining)
        self.assertNotIn("Free Hit", currently_available_chips(None, 1, remaining))
        self.assertIn("Wildcard", remaining)
        self.assertNotIn("Wildcard", currently_available_chips(None, 1, remaining))
        state = {"chipsUsed": [{"chip": "Free Hit", "gameweek": 19}]}
        refreshed = manager_inventory(state, 20)
        self.assertIn("Free Hit", refreshed)
        self.assertNotIn(
            "Free Hit", currently_available_chips(state, 20, refreshed)
        )

    def test_manager_rebuild_budget_uses_sale_value_and_bank(self) -> None:
        players = [
            {"id": 1, "price": 8.0},
            {"id": 2, "price": 6.5},
        ]
        state = {
            "squadIds": [1, 2],
            "purchasePrices": {"1": 7.0, "2": 7.0},
            "bank": 1.2,
        }
        # Player 1 rose by 1.0, so only half the profit is retained; player 2
        # fell and is sold at the current price.
        self.assertEqual(available_squad_budget(players, state), 15.2)

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
        cls.current = json.loads((APP_DATA / "current-players.json").read_text(encoding="utf-8"))
        cls.captain_route = json.loads(
            (ROOT / "analysis" / "data" / "captain_route_consensus_validation.json").read_text(
                encoding="utf-8"
            )
        )

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

    def test_four_legal_shadow_squads(self) -> None:
        self.assertEqual(len(self.decision["managers"]), 4)
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
        if self.chips["recommendation"] == "Wildcard":
            self.assertTrue(self.chips["forcedByExpiryCollision"])

    def test_live_free_hit_cross_check_is_versioned(self) -> None:
        calibration = self.chips["freeHitCalibration"]
        self.assertEqual(calibration["trainedThrough"], "2025/26")
        self.assertEqual(calibration["trainingRows"], 379)
        self.assertIn("learnedCorrelation", calibration["causalValidation"])

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
        frontier_ids = {int(row["id"]) for row in self.frontier["players"]}
        listwise_ids = {int(row["id"]) for row in self.listwise["players"]}
        self.assertTrue(frontier_ids.issubset(listwise_ids))
        self.assertLessEqual(len(listwise_ids - frontier_ids), 2)
        action = self.listwise["actionChallenger"]
        self.assertIn("invalidated", action["status"])
        self.assertEqual(action["activePlayerCount"], 0)
        self.assertTrue(
            all(not row["actionPolicyActive"] for row in self.listwise["players"])
        )
        captain_route = self.listwise["captainRouteChallenger"]
        self.assertEqual(captain_route["status"], "prospective captain shadow")
        self.assertIn(
            "captain-route-consensus",
            {manager["id"] for manager in self.decision["managers"]},
        )
        self.assertEqual(self.performance["targetHits"], 0)
        self.assertGreater(self.performance["stackLift"], 0)
        self.assertIn("cannot promote", self.performance["governance"].lower())

    def test_live_action_feature_bridge_is_complete(self) -> None:
        required = set(lens.LIVE_ACTION_FEATURES + lens.LIVE_ROUTE_FEATURES)
        self.assertGreater(len(self.current), 300)
        for row in self.current:
            research = row.get("researchFeatures", {})
            self.assertTrue(required.issubset(research))
            self.assertTrue(all(research[name] is not None for name in required))
        for row in self.listwise["players"]:
            self.assertGreaterEqual(float(row["actionVote"]), 0)
            self.assertLessEqual(float(row["actionVote"]), 1)
            self.assertGreaterEqual(int(row["actionAgreementSeeds"]), 0)
            self.assertLessEqual(int(row["actionAgreementSeeds"]), 5)

    def test_scoring_route_information_boundary(self) -> None:
        forbidden = {
            "expected_goals",
            "expected_assists",
            "expected_goals_conceded",
        }
        self.assertTrue(forbidden.isdisjoint(ROUTE_FEATURES))
        self.assertEqual(
            self.captain_route["informationBoundary"]["forbiddenFeaturesPresent"],
            [],
        )

    def test_route_captain_historical_gate_passed(self) -> None:
        self.assertTrue(self.captain_route["stability"]["passed"])
        recursive = self.captain_route["recursive"]["comparisons"]
        self.assertGreater(recursive["noChips"]["averageDelta"], 0)
        self.assertGreaterEqual(recursive["noChips"]["minimumDelta"], 0)
        self.assertGreater(recursive["auditedChips"]["averageDelta"], 0)

    def test_route_captain_manager_changes_only_captain_layer(self) -> None:
        managers = {manager["id"]: manager for manager in self.decision["managers"]}
        hybrid = {int(row["id"]) for row in managers["frontier-challenger"]["squad"]}
        route = {
            int(row["id"])
            for row in managers["captain-route-consensus"]["squad"]
        }
        self.assertEqual(route, hybrid)

    def test_joint_optimizer_does_not_park_a_premium_forward_on_the_bench(self) -> None:
        players: list[dict] = []
        identifier = 1
        for position, prices, scores in (
            ("GK", [5.0, 5.0], [6.0, 2.0]),
            ("DEF", [5.0] * 5, [8.0, 7.5, 7.0, 6.5, 6.0]),
            ("MID", [7.0] * 5, [9.0, 8.5, 8.0, 7.5, 7.0]),
            ("FWD", [14.0, 11.0, 4.5, 8.0], [10.0, 9.5, 3.0, 4.0]),
        ):
            for price, score in zip(prices, scores):
                players.append(
                    {
                        "id": identifier,
                        "team": f"T{identifier}",
                        "position": position,
                        "price": price,
                        "testScore": score,
                    }
                )
                identifier += 1

        expensive_bench_forward = players[-1]["id"]
        chosen, xi, _, _ = optimise_squad(players, "testScore")
        self.assertNotIn(expensive_bench_forward, [players[index]["id"] for index in chosen])
        self.assertGreaterEqual(sum(players[index]["price"] for index in chosen), 99.5)
        self.assertEqual(len(xi), 11)


if __name__ == "__main__":
    unittest.main()
