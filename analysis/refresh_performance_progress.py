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
    result = {
        "schemaVersion": 1,
        "status": "research-only",
        "controlAverage": control["average"],
        "hybridAverage": hybrid_best["average"],
        "captainStackAverage": captain["average"],
        "legacyChipStackAverage": chips["average"] if chips else None,
        "weeklyRebuildCeiling": weekly_ceiling,
        "top500Pace": target,
        "stackLift": round(captain["average"] - control["average"], 1),
        "remainingGap": round(target - captain["average"], 1),
        "gapClosedPercent": round(100 * (captain["average"] - control["average"]) / max(target - control["average"], 0.1), 1),
        "controlMinimum": control["minimum"],
        "hybridMinimum": hybrid_best["minimum"],
        "targetHits": captain["targetHits"],
        "experiments": [
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
        ],
        "bottleneck": "The carried-squad transfer path remains the largest attainable loss. Unlimited weekly rebuilds nearly reach the pace line, while legal recursive management cannot instantly reach each fresh optimum.",
        "governance": "All new results have been exposed to the historical evaluation seasons. They may change the pre-GW1 shadow challenger, but cannot promote the production model.",
    }
    atomic_json(APP_DATA / "performance-progress.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
