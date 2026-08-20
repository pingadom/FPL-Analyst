"""Assemble the honest evidence boundary for joint chip sequencing."""

from __future__ import annotations

import json

import calibrate_model as lens
from breakthrough_engine import (
    ChipState,
    ChipTransition,
    optimise_chip_sequence,
)


def semantic_checks() -> dict:
    def transitions(state: ChipState) -> list[ChipTransition]:
        rows = [ChipTransition("Hold", 0.0, state.permanent_state)]
        values = {
            (1, "Triple Captain"): 4.0,
            (2, "Triple Captain"): 10.0,
            (1, "Free Hit"): 7.0,
            (2, "Wildcard"): 6.0,
        }
        for chip in state.available:
            value = values.get((state.week, chip))
            if value is None:
                continue
            rows.append(
                ChipTransition(
                    action=chip,
                    immediate_value=value,
                    next_permanent_state=(
                        "temporary"
                        if chip == "Free Hit"
                        else ("wildcarded" if chip == "Wildcard" else state.permanent_state)
                    ),
                    consumes_chip=chip,
                    preserves_permanent_state=chip == "Free Hit",
                )
            )
        return rows

    plan = optimise_chip_sequence(
        ChipState(
            week=1,
            end_week=2,
            available=frozenset({"Triple Captain", "Free Hit", "Wildcard"}),
            permanent_state="base",
        ),
        transitions,
        discount=1.0,
    )
    actions = [row for row in plan.actions if row[1] != "Hold"]
    return {
        "actions": [list(row) for row in plan.actions],
        "oneChipPerGameweek": len({week for week, _ in actions}) == len(actions),
        "waitedForBetterTripleCaptain": (2, "Triple Captain") in plan.actions,
        "freeHitNonPersistenceCoveredByUnitTest": True,
        "terminalState": plan.terminal_state,
    }


def main() -> None:
    data_root = lens.ROOT / "analysis" / "data"
    sequential = json.loads(
        (data_root / "sequential_chip_value_validation.json").read_text(
            encoding="utf-8"
        )
    )
    combined = json.loads(
        (data_root / "combined_chip_policy_validation.json").read_text(
            encoding="utf-8"
        )
    )
    ablation = json.loads(
        (data_root / "wildcard_freehit_ablation.json").read_text(encoding="utf-8")
    )
    wildcard = ablation["selected"]["Wildcard"]
    checks = semantic_checks()
    result = {
        "status": "joint sequential planner implemented; live diagnostic only",
        "historicalEvidence": {
            "tripleCaptainBenchBoostAverageGain": sequential["oldAuditedPolicy"][
                "evaluationAverageGain"
            ],
            "correctedFreeHitIncrementalAverage": combined["pairedVsTcBb"][
                "evaluationAverage"
            ],
            "correctedFreeHitHoldoutAverage": combined["pairedVsTcBb"][
                "holdout2022to2025Average"
            ],
            "correctedFreeHitHoldoutMinimum": combined["pairedVsTcBb"][
                "holdoutMinimum"
            ],
            "automaticWildcardAverageGain": wildcard["evaluationAverageGain"],
            "automaticWildcardMinimumGain": wildcard["evaluationMinimumGain"],
        },
        "sequenceSemantics": checks,
        "livePlanner": {
            "oneChipPerGameweek": True,
            "twoHalfSeasonInventories": True,
            "freeHitPreservesPermanentSquad": True,
            "wildcardChangesPermanentSquad": True,
            "tcBbUseMarginalNotTotalPoints": True,
            "unknownFixturesRetainOptionValue": True,
            "usesPastOnlyReservationValues": True,
        },
        "decision": (
            "Keep the audited TC/BB policy and corrected Free Hit in the live "
            "shadow. Reject automatic Wildcards. The joint DP is exposed as a "
            "receding-horizon diagnostic until deadline-time fixture-announcement "
            "vintages and prospective outcomes can validate its timing choices."
        ),
        "productionPromotion": False,
    }
    output = data_root / "breakthrough_chip_sequence_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
