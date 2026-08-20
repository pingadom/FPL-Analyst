"""Publish compact, evidence-labelled performance progress for the dashboard."""

from __future__ import annotations

import json
from statistics import mean

from prospective_common import APP_DATA, ROOT, atomic_json


def main() -> None:
    decision = json.loads((ROOT / "analysis" / "data" / "decision_regret_audit.json").read_text(encoding="utf-8"))
    listwise = json.loads((ROOT / "analysis" / "data" / "listwise_ranker_validation.json").read_text(encoding="utf-8"))
    hybrid = json.loads((ROOT / "analysis" / "data" / "hybrid_decision_validation.json").read_text(encoding="utf-8"))
    stack_path = ROOT / "analysis" / "data" / "championship_stack_validation.json"
    stack = json.loads(stack_path.read_text(encoding="utf-8")) if stack_path.exists() else {"variants": {}}
    defence = json.loads((ROOT / "analysis" / "data" / "team_defence_residual_audit.json").read_text(encoding="utf-8"))
    role_rows = decision["variants"]["lens7_role"]
    weekly_ceiling = round(mean(row["unlimitedWeeklyRebuild"] for row in role_rows), 1)
    target = round(mean(row["target"] for row in role_rows), 1)
    control = hybrid["forecastModels"]["stable"]
    hybrid_best = hybrid["forecastModels"][hybrid["bestForecastCombination"]]
    variants = stack.get("variants", {})
    captain = variants.get("hybridCaptain50", hybrid_best)
    chips = variants.get("hybridCaptain50LegacyChips")
    final_path = ROOT / "analysis" / "data" / "final_breakthrough_validation_v3.json"
    final = json.loads(final_path.read_text(encoding="utf-8")) if final_path.exists() else None
    if final:
        stages = final["stages"]
        control_average = stages["originalRouteControl"]["average"]
        hybrid_average = stages["previousForecastV2"]["average"]
        captain_average = stages["fullyIntegrated"]["average"]
        legacy_chip_average = stages["newCaptainNoChips"]["average"]
        target = stages["fullyIntegrated"]["top500Pace"]
        control_minimum = stages["originalRouteControl"]["minimum"]
        hybrid_minimum = stages["previousForecastV2"]["minimum"]
        target_hits = stages["fullyIntegrated"]["top500Hits"]
        experiments = [
            {
                "name": "Large captain surface",
                "decision": "shadow",
                "average": stages["newCaptainOldChips"]["average"],
                "delta": round(
                    stages["newCaptainOldChips"]["average"] - control_average, 1
                ),
                "detail": "1,655 screened configurations and 12 exact recursive finalists selected the 80/20 dynamic captain.",
            },
            {
                "name": "Recursive chip retuning",
                "decision": "shadow",
                "average": captain_average,
                "delta": round(
                    captain_average - stages["newCaptainNoChips"]["average"], 1
                ),
                "detail": "The 756-policy BB/TC/FH winner improved every evaluation season.",
            },
            {
                "name": "Automatic Wildcard",
                "decision": "rejected",
                "average": None,
                "delta": -15.8,
                "detail": "All six exact h10 Wildcard variants failed; the best lost 15.8 points per season.",
            },
            {
                "name": "Rollout action-value learner",
                "decision": "rejected",
                "average": 2177.5,
                "delta": -13.1,
                "detail": "70,247 realised rollout packages improved prediction error but worsened recursive decisions.",
            },
        ]
        bottleneck = (
            "The carried-squad forecast and transfer path remains the largest loss. "
            "Recent-season regime changes and unavailable exact deadline-vintage market data "
            "account for most of the remaining 84.8-point gap."
        )
    else:
        control_average = control["average"]
        hybrid_average = hybrid_best["average"]
        captain_average = captain["average"]
        legacy_chip_average = chips["average"] if chips else None
        control_minimum = control["minimum"]
        hybrid_minimum = hybrid_best["minimum"]
        target_hits = captain["targetHits"]
        experiments = [
            {
                "name": "Listwise transfer horizon",
                "decision": "shadow",
                "average": listwise["models"]["listPlan25"]["average"],
                "delta": round(listwise["models"]["listPlan25"]["average"] - listwise["models"]["stable"]["average"], 1),
                "detail": "Largest standalone gain; prioritises persistent six-week ordering.",
            },
            {
                "name": "Immediate + horizon fusion",
                "decision": "shadow",
                "average": hybrid_best["average"],
                "delta": round(hybrid_best["average"] - control["average"], 1),
                "detail": "Frontier next-GW ordering and listwise transfer planning are complementary.",
            },
            {
                "name": "Captain listwise rank",
                "decision": "shadow",
                "average": captain["average"],
                "delta": round(captain["average"] - hybrid_best["average"], 1),
                "detail": "Captain-only rerank; does not alter the transfer model.",
            },
            {
                "name": "Extra big-team defender boost",
                "decision": "rejected",
                "average": None,
                "delta": 0,
                "detail": defence["reason"],
            },
            {
                "name": "Percentile relevance labels",
                "decision": "rejected",
                "average": 2128.5,
                "delta": round(2128.5 - control["average"], 1),
                "detail": "Helped immediate ordering but destroyed the six-week planning gain.",
            },
        ]
        bottleneck = "The carried-squad transfer path remains the largest attainable loss. Unlimited weekly rebuilds nearly reach the pace line, while legal recursive management cannot instantly reach each fresh optimum."
    result = {
        "schemaVersion": 1,
        "status": "research-only",
        "controlAverage": control_average,
        "hybridAverage": hybrid_average,
        "captainStackAverage": captain_average,
        "legacyChipStackAverage": legacy_chip_average,
        "weeklyRebuildCeiling": weekly_ceiling,
        "top500Pace": target,
        "stackLift": round(captain_average - control_average, 1),
        "remainingGap": round(target - captain_average, 1),
        "gapClosedPercent": round(100 * (captain_average - control_average) / max(target - control_average, 0.1), 1),
        "controlMinimum": control_minimum,
        "hybridMinimum": hybrid_minimum,
        "targetHits": target_hits,
        "experiments": experiments,
        "bottleneck": bottleneck,
        "governance": "All new results have been exposed to the historical evaluation seasons. They may change the pre-GW1 shadow challenger, but cannot promote the production model.",
    }
    atomic_json(APP_DATA / "performance-progress.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
