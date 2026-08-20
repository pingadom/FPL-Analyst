import unittest

from live_external_signals import (
    implied_goal_rates,
    no_vig,
    parse_matchbook_event,
    poisson_outcomes,
)


def runner(name: str, back: float, lay: float) -> dict:
    return {
        "name": name,
        "prices": [
            {"side": "back", "decimal-odds": back, "available-amount": 250},
            {"side": "lay", "decimal-odds": lay, "available-amount": 250},
        ],
    }


class ExternalSignalTests(unittest.TestCase):
    def test_no_vig_probabilities_sum_to_one(self) -> None:
        probabilities = no_vig([0.52, 0.29, 0.24])
        self.assertAlmostEqual(sum(probabilities), 1.0, places=12)
        self.assertGreater(probabilities[0], probabilities[1])

    def test_poisson_inverse_recovers_rates(self) -> None:
        target = poisson_outcomes(1.84, 1.06)
        recovered = implied_goal_rates(*target[:3], target[3])
        self.assertAlmostEqual(recovered[0], 1.84, delta=0.06)
        self.assertAlmostEqual(recovered[1], 1.06, delta=0.06)

    def test_matchbook_parser_is_no_vig_and_goal_consistent(self) -> None:
        event = {
            "id": 42,
            "name": "Arsenal vs Coventry City",
            "start": "2026-08-22T14:00:00Z",
            "markets": [
                {
                    "name": "Match Odds",
                    "volume": 18_000,
                    "runners": [
                        runner("Arsenal", 1.43, 1.45),
                        runner("Draw", 5.1, 5.3),
                        runner("Coventry City", 8.2, 8.6),
                    ],
                },
                {
                    "name": "Total",
                    "handicap": 2.5,
                    "runners": [
                        runner("OVER 2.5", 1.74, 1.77),
                        runner("UNDER 2.5", 2.20, 2.24),
                    ],
                },
            ],
        }
        parsed = parse_matchbook_event(event)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        total = (
            parsed["homeProbability"]
            + parsed["drawProbability"]
            + parsed["awayProbability"]
        )
        self.assertAlmostEqual(total, 1.0, places=10)
        self.assertGreater(parsed["homeExpectedGoals"], parsed["awayExpectedGoals"])
        self.assertGreater(parsed["quality"], 0.5)


if __name__ == "__main__":
    unittest.main()
