import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ModelAuditArtifactTests(unittest.TestCase):
    def test_public_audit_matches_reproducible_outputs(self) -> None:
        audit = json.loads(
            (ROOT / "app" / "data" / "model-audit.json").read_text(encoding="utf-8")
        )
        model = json.loads(
            (ROOT / "app" / "data" / "model-results.json").read_text(
                encoding="utf-8-sig"
            )
        )
        comparison = json.loads(
            (
                ROOT
                / "analysis"
                / "data"
                / "lens8_shadow_comparison.json"
            ).read_text(encoding="utf-8")
        )

        lens8_average = round(
            sum(float(row["points"]) for row in model["backtest"])
            / len(model["backtest"]),
            1,
        )
        self.assertEqual(audit["lens8"]["average"], lens8_average)
        self.assertEqual(audit["lens8"]["average"], comparison["lens8"]["average"])
        self.assertEqual(
            audit["causalChallenger"]["average"],
            comparison["causalRepairedShadow"]["average"],
        )
        self.assertEqual(
            audit["legacyBreakthrough"]["reproducedAverage"],
            comparison["repairedShadow"]["average"],
        )
        self.assertEqual(
            audit["legacyBreakthrough"]["overstatement"],
            -comparison["deltas"]["repairedShadowVsRecordedLegacy"],
        )
        self.assertAlmostEqual(
            audit["lens8"]["average"] - audit["lens8"]["deltaVsLens7"],
            audit["lens7"]["average"],
        )
        self.assertIn("Retired", audit["legacyBreakthrough"]["status"])
        self.assertTrue(model["fixtureIntegrity"]["passed"])


if __name__ == "__main__":
    unittest.main()
