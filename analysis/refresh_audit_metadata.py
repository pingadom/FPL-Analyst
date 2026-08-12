"""Refresh audit-only metadata without rerunning live or historical forecasts."""

from __future__ import annotations

import json
import math

import calibrate_model as lens


def main() -> None:
    result = json.loads(lens.OUTPUT.read_text(encoding="utf-8-sig"))
    benchmark_payload = json.loads(
        (lens.ROOT / "analysis" / "data" / "historical_rank_benchmarks.json").read_text(
            encoding="utf-8"
        )
    )
    benchmarks = {row["season"]: row for row in benchmark_payload["seasons"]}
    local_ranks: list[int] = []
    censored = 0
    for row in result["backtest"]:
        benchmark = benchmarks[row["season"]]
        margin = int(row["points"] - benchmark["points"])
        local = abs(margin) <= 50
        estimated = (
            max(
                1,
                int(
                    round(
                        500_000
                        * math.exp(float(benchmark["logRankSlope"]) * margin)
                    )
                ),
            )
            if local
            else None
        )
        row["estimatedRank"] = estimated
        row["estimatedRankInterval"] = (
            sorted(
                [
                    max(
                        1,
                        int(
                            round(
                                500_000
                                * math.exp(
                                    float(benchmark["logRankSlope"])
                                    * (row["points"] - cutoff)
                                )
                            )
                        ),
                    )
                    for cutoff in (benchmark["p05"], benchmark["p95"])
                ]
            )
            if local
            else None
        )
        row["rankEstimateLocal"] = local
        row["estimatedBand"] = (
            "Above top-500k cutoff"
            if margin >= 0
            else "Near top-500k cutoff"
            if margin >= -50
            else "Below locally calibrated rank range"
        )
        if estimated is None:
            censored += 1
        else:
            local_ranks.append(estimated)
    result["rankTarget"]["averageEstimatedRank"] = (
        round(sum(local_ranks) / len(local_ranks) / 1000) * 1000
        if local_ranks and censored == 0
        else None
    )
    result["rankTarget"]["rankEstimateCoverage"] = len(local_ranks)
    result["rankTarget"]["rankEstimateCensoredSeasons"] = censored
    result["rankTarget"]["method"] = (
        "Empirical cutoff estimate from a deterministic sample of 5,000 public "
        "official FPL manager histories. A local log(rank)-points fit and nearest "
        "observed score boundary reconstruct each cutoff; its interval also allows "
        "for ties and survivorship. Probability uses 4,000 four-GW block-bootstrap "
        "model seasons. Rank is withheld outside a 50-point local calibration "
        "window rather than extrapolating the cutoff curve into unsupported ranks."
    )
    audit_path = lens.ROOT / "analysis" / "data" / "audited_policy_validation.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    average = float(audit["average"])
    hits = int(audit["targetHits"])
    margin = float(audit["averageMargin"])
    promoted = bool(hits >= 6 and margin >= 0)
    target_average = round(
        sum(item["top500Target"] for item in result["backtest"])
        / len(result["backtest"]),
        1,
    )
    research_average = round(
        sum(item["points"] for item in result["backtest"])
        / len(result["backtest"]),
        1,
    )
    result["championGovernance"] = {
        "decisionChampion": "Lens 7.0" if promoted else "Research baseline",
        "decisionChallenger": "Frozen audited policy",
        "decisionPromoted": promoted,
        "reason": (
            "Promoted: the frozen pre-2018 policy cleared the estimated top-500k line in at least six of eight seasons with a non-negative average margin."
            if promoted
            else "Research-only: the frozen pre-2018 audit has not demonstrated consistent top-500k performance."
        ),
        "incumbentAveragePoints": target_average,
        "challengerAveragePoints": round(average, 1),
        "incumbentTop500Hits": 6,
        "challengerTop500Hits": hits,
        "playerLayerPromoted": promoted,
        "incumbentPlayerMae": None,
        "challengerPlayerMae": result["calibrationDiagnostics"]["mae"],
        "promotionRule": "Promotion requires at least 6/8 top-500k cutoff hits and a non-negative average cutoff margin under a policy frozen on 2016/17 and 2017/18; later searches remain diagnostics.",
    }
    result["frozenAudit"] = {
        "available": True,
        "selection": audit["selection"],
        "averagePoints": round(average, 1),
        "top500Hits": hits,
        "averageMargin": round(margin, 1),
        "minimumPoints": int(audit["minimum"]),
        "averageChipDelta": float(audit["averageChipDelta"]),
        "researchSearchAverage": research_average,
        "method": "The promotion benchmark is selected only on 2016/17 and 2017/18. The broader recursive search is shown for research transparency but cannot promote itself after exposure to later-season results.",
    }
    lens.OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Refreshed audit metadata in {lens.OUTPUT.relative_to(lens.ROOT)}")


if __name__ == "__main__":
    main()
