"""Large causal chip-threshold screen with exact recursive finalists."""

from __future__ import annotations

import itertools
import json

import numpy as np

import calibrate_model as lens
from action_specific_tenure_v2 import build_surfaces
from captain_fixture_history_validation import add_fixture_history
from captain_route_consensus_validation import selected_consensus_metric, weekly_percentile
from dynamic_match_model_v2 import build_dynamic_history
from forecast_layer_v2 import captain_availability_score, dynamic_route_score
from freehit_value_validation import causal_predictions, opportunity_frame
from frontier_ranker_validation import STRATEGY
from multiscale_horizon_validation import add_targets
from probabilistic_minutes_validation import causal_predictions as minute_predictions
from probabilistic_minutes_validation import season_summary
from wildcard_freehit_ablation import champion_forecasts


BB_THRESHOLDS = tuple(float(value) for value in range(8, 17))
TC_THRESHOLDS = tuple(float(value) for value in range(10, 23, 2))
FH_THRESHOLDS = (0.0, 3.0, 6.0, 9.0)
FH_RISK = (0.0, 0.35, 0.70)
FINALISTS = 8


def selected_captain(data, immediate, frozen_captain, minute):
    route = selected_consensus_metric(data, immediate, frozen_captain)
    dynamic, _ = dynamic_route_score(data, immediate, 0.70)
    dynamic, _ = captain_availability_score(data, dynamic, minute, 0.50)
    return 0.80 * route + 0.20 * weekly_percentile(data, dynamic)


def sequential_chip_gain(frame, thresholds: dict[str, float]) -> tuple[float, list[dict]]:
    """Causal first-crossing screen: no future signal is inspected."""
    total = 0.0
    choices = []
    season = str(frame["season"].iloc[0])
    halves = ((-999, 19), (20, 999)) if season == "2025-26" else ((-999, 999),)
    for lower, upper in halves:
        available = {"Bench Boost", "Triple Captain", "Free Hit"}
        local = frame[(frame["gw"] >= lower) & (frame["gw"] <= upper)].sort_values("gw")
        for row in local.itertuples():
            candidates = []
            values = {
                "Bench Boost": (float(row.bbSignal), float(row.actualBenchBoostGain)),
                "Triple Captain": (float(row.tcSignal), float(row.actualTripleCaptainGain)),
                "Free Hit": (float(row.fhSignal), float(row.actualFreeHitGain)),
            }
            for chip in available:
                signal, actual = values[chip]
                excess = signal - thresholds[chip]
                if excess >= 0:
                    candidates.append((excess, signal, chip, actual))
            if not candidates:
                continue
            _, signal, chip, actual = max(candidates)
            available.remove(chip)
            total += actual
            choices.append(
                {
                    "gw": int(row.gw),
                    "chip": chip,
                    "signal": round(float(signal), 3),
                    "actualGain": round(float(actual), 1),
                }
            )
    return total, choices


def paired(totals, baseline, seasons) -> dict:
    summary = season_summary(totals, seasons)
    delta = totals[2:] - baseline[2:]
    development = delta[:-2]
    return {
        **summary,
        "averageGain": round(float(delta.mean()), 1),
        "developmentGain": round(float(development.mean()), 1),
        "developmentStability": round(
            float(development.mean() - 0.20 * development.std()), 3
        ),
        "holdoutGain": round(float(delta[-2:].mean()), 1),
        "minimumGain": int(delta.min()),
        "positiveSeasons": int((delta > 0).sum()),
        "negativeSeasons": int((delta < 0).sum()),
        "seasonGains": delta.astype(int).tolist(),
    }


def main() -> None:
    dynamic, _ = build_dynamic_history()
    data = add_fixture_history(add_targets(dynamic.reset_index(drop=True)))
    immediate, plan, frozen_captain = champion_forecasts(data)
    minute = minute_predictions(data)
    captain = selected_captain(data, immediate, frozen_captain, minute)
    seasons = list(dict.fromkeys(data["season"].tolist()))
    free_hit_squads = lens.precompute_fresh_squads(data, immediate)
    collector_policy = lens.ChipPolicy(
        1e6,
        1e6,
        1e6,
        1e6,
        0.0,
        10,
        28,
        ("Free Hit", "Bench Boost", "Triple Captain"),
    )
    print("Collecting selected-path chip opportunities", flush=True)
    no_chip, collector_stats = lens.simulate_candidate(
        data,
        immediate,
        STRATEGY,
        chip_policy=collector_policy,
        free_hit_squads=free_hit_squads,
        plan_scores=plan,
        captain_scores=captain,
    )
    frame = opportunity_frame(collector_stats, seasons)
    fh_prediction, fh_scale, fit_audit = causal_predictions(frame)
    frame["fhBase"] = fh_prediction - frame["permanentTransferValueForegone"].to_numpy(float)
    frame["bbSignal"] = (
        frame["predictedBenchBoostGain"].to_numpy(float)
        + 0.15 * frame["benchDoubleCount"].to_numpy(float)
    )
    frame["tcSignal"] = (
        frame["predictedTripleCaptainGain"].to_numpy(float)
        * frame["captainFixtureCount"].clip(lower=1).to_numpy(float)
    )

    screening = []
    evaluation_seasons = list(lens.EVALUATION_SEASONS)
    for bench, triple, free_hit, risk in itertools.product(
        BB_THRESHOLDS, TC_THRESHOLDS, FH_THRESHOLDS, FH_RISK
    ):
        frame["fhSignal"] = frame["fhBase"] - risk * fh_scale
        gains = []
        choices = []
        for season in evaluation_seasons:
            gain, season_choices = sequential_chip_gain(
                frame[frame["season"].eq(season)],
                {
                    "Bench Boost": bench,
                    "Triple Captain": triple,
                    "Free Hit": free_hit,
                },
            )
            gains.append(gain)
            choices.append(season_choices)
        development = np.asarray(gains[:-2], float)
        screening.append(
            {
                "benchThreshold": bench,
                "tripleThreshold": triple,
                "freeHitThreshold": free_hit,
                "freeHitRisk": risk,
                "screenDevelopmentGain": round(float(development.mean()), 3),
                "screenDevelopmentStability": round(
                    float(development.mean() - 0.20 * development.std()), 3
                ),
                "screenHoldoutGain": round(float(np.mean(gains[-2:])), 3),
                "screenMinimumGain": round(float(min(gains)), 1),
                "screenSeasonGains": [round(float(value), 1) for value in gains],
                "screenChoices": choices,
            }
        )
    finalists = sorted(
        screening,
        key=lambda row: (
            row["screenDevelopmentStability"],
            row["screenDevelopmentGain"],
            row["screenMinimumGain"],
        ),
        reverse=True,
    )[:FINALISTS]
    exact = []
    for index, finalist in enumerate(finalists, start=1):
        print(f"Exact chip finalist {index}/{len(finalists)}", flush=True)
        frame["fhSignal"] = frame["fhBase"] - finalist["freeHitRisk"] * fh_scale
        overrides = {
            (str(row.season), int(row.gw), "Free Hit"): float(row.fhSignal)
            for row in frame.itertuples()
        }
        policy = lens.ChipPolicy(
            1e6,
            finalist["freeHitThreshold"],
            finalist["benchThreshold"],
            finalist["tripleThreshold"],
            0.0,
            10,
            28,
            ("Free Hit", "Bench Boost", "Triple Captain"),
        )
        totals, stats = lens.simulate_candidate(
            data,
            immediate,
            STRATEGY,
            chip_policy=policy,
            free_hit_squads=free_hit_squads,
            plan_scores=plan,
            captain_scores=captain,
            chip_value_overrides=overrides,
        )
        exact.append(
            {
                **{key: value for key, value in finalist.items() if key != "screenChoices"},
                **paired(totals, no_chip, seasons),
                "exactChoices": [row["chips"] for row in stats[2:]],
            }
        )
    selected = max(
        exact,
        key=lambda row: (
            row["developmentStability"],
            row["developmentGain"],
            row["minimumGain"],
        ),
    )

    # Wildcard is evaluated only after the TC/BB/FH policy is frozen by
    # development evidence. Its fresh squad uses the explicit h10 surface.
    surfaces, _ = build_surfaces(data, immediate)
    wildcard_squads = lens.precompute_fresh_squads(data, surfaces["wildcard"])
    frame["fhSignal"] = frame["fhBase"] - selected["freeHitRisk"] * fh_scale
    selected_overrides = {
        (str(row.season), int(row.gw), "Free Hit"): float(row.fhSignal)
        for row in frame.itertuples()
    }
    no_wc_policy = lens.ChipPolicy(
        1e6,
        selected["freeHitThreshold"],
        selected["benchThreshold"],
        selected["tripleThreshold"],
        0.0,
        10,
        28,
        ("Free Hit", "Bench Boost", "Triple Captain"),
    )
    selected_totals, _ = lens.simulate_candidate(
        data,
        immediate,
        STRATEGY,
        chip_policy=no_wc_policy,
        free_hit_squads=free_hit_squads,
        plan_scores=plan,
        captain_scores=captain,
        chip_value_overrides=selected_overrides,
    )
    wildcard_rows = []
    for gap, first, second in (
        (40.0, 6, 20),
        (60.0, 6, 20),
        (80.0, 6, 20),
        (40.0, 10, 28),
        (60.0, 10, 28),
        (80.0, 10, 28),
    ):
        print(f"Exact Wildcard gap={gap:g}, windows={first}/{second}", flush=True)
        policy = lens.ChipPolicy(
            gap,
            selected["freeHitThreshold"],
            selected["benchThreshold"],
            selected["tripleThreshold"],
            0.55,
            first,
            second,
            ("Wildcard", "Free Hit", "Bench Boost", "Triple Captain"),
        )
        totals, stats = lens.simulate_candidate(
            data,
            immediate,
            STRATEGY,
            chip_policy=policy,
            fresh_squads=wildcard_squads,
            free_hit_squads=free_hit_squads,
            plan_scores=plan,
            captain_scores=captain,
            chip_value_overrides=selected_overrides,
        )
        wildcard_rows.append(
            {
                "wildcardGap": gap,
                "firstWildcardMinGw": first,
                "secondWildcardMinGw": second,
                **paired(totals, selected_totals, seasons),
                "choices": [row["chips"] for row in stats[2:]],
            }
        )
    selected_wc = max(
        wildcard_rows,
        key=lambda row: (
            row["developmentStability"],
            row["developmentGain"],
            row["minimumGain"],
        ),
    )
    wc_gate = {
        "developmentPositive": selected_wc["developmentGain"] > 0,
        "holdoutNonNegative": selected_wc["holdoutGain"] >= 0,
        "minimumAtLeastMinusTen": selected_wc["minimumGain"] >= -10,
        "positiveSeasonsAtLeastFour": selected_wc["positiveSeasons"] >= 4,
    }
    chip_gate = {
        "developmentPositive": selected["developmentGain"] > 0,
        "holdoutNonNegative": selected["holdoutGain"] >= 0,
        "minimumAtLeastMinusTen": selected["minimumGain"] >= -10,
        "positiveSeasonsAtLeastFive": selected["positiveSeasons"] >= 5,
    }
    result = {
        "schemaVersion": 1,
        "status": "large causal chip policy search",
        "screenedPolicies": len(screening),
        "exactFinalists": len(exact),
        "freeHitFitAudit": fit_audit,
        "noChipBaseline": season_summary(no_chip, seasons),
        "selectedByDevelopmentOnly": selected,
        "chipGate": chip_gate,
        "chipPolicyPassed": all(chip_gate.values()),
        "wildcardSelectedByDevelopmentOnly": selected_wc,
        "wildcardGate": wc_gate,
        "wildcardPassed": all(wc_gate.values()),
        "wildcardExperiments": wildcard_rows,
        "exactPolicies": exact,
        "screening": screening,
    }
    output = lens.ROOT / "analysis" / "data" / "chip_surface_search_v2.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"screened": len(screening), "selected": selected, "chipGate": chip_gate, "wildcard": selected_wc, "wildcardGate": wc_gate}, indent=2))


if __name__ == "__main__":
    main()
