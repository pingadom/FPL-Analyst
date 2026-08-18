"""Interaction audit: agreed transfer-action challenger plus audited chips."""

from __future__ import annotations

import json

import numpy as np

import calibrate_model as lens
from bench_efficiency_validation import variant_summary
from freehit_value_validation import causal_predictions, opportunity_frame
from frontier_ranker_validation import STRATEGY
from multiscale_horizon_validation import add_targets
from transfer_action_ranker_validation import agreed_action_plan
from wildcard_freehit_ablation import champion_forecasts


def main() -> None:
    original, _ = lens.load_or_build_prepared_history()
    data = add_targets(original.reset_index(drop=True))
    scores, champion_plan, captain = champion_forecasts(data)
    action_plan = agreed_action_plan(data, champion_plan, 0.05)
    seasons = list(dict.fromkeys(data["season"].tolist()))
    free_hit_squads = lens.precompute_fresh_squads(data, scores)

    collector_policy = lens.ChipPolicy(1e6, 1e6, 1e6, 1e6, 0.0, 10, 28, ("Free Hit",))
    print("Collecting action-path Free Hit counterfactuals", flush=True)
    _, collector_stats = lens.simulate_candidate(
        data,
        scores,
        STRATEGY,
        chip_policy=collector_policy,
        free_hit_squads=free_hit_squads,
        plan_scores=action_plan,
        captain_scores=captain,
    )
    frame = opportunity_frame(collector_stats, seasons)
    prediction, _, fit_audit = causal_predictions(frame)
    adjusted = prediction - frame["permanentTransferValueForegone"].to_numpy(float)
    overrides = {
        (str(row.season), int(row.gw), "Free Hit"): float(adjusted[index])
        for index, row in frame.iterrows()
    }

    configurations = {
        "action-no-chips": None,
        "action-tc-bb": lens.AUDITED_CHAMPION_CHIP_POLICY,
        "action-tc-bb-fh": lens.ChipPolicy(
            1e6,
            3.0,
            lens.AUDITED_CHAMPION_CHIP_POLICY.bench_score,
            lens.AUDITED_CHAMPION_CHIP_POLICY.triple_score,
            0.0,
            10,
            28,
            ("Free Hit", "Bench Boost", "Triple Captain"),
        ),
    }
    rows = []
    totals_by_name: dict[str, np.ndarray] = {}
    for name, policy in configurations.items():
        print(f"Running {name}", flush=True)
        totals, stats = lens.simulate_candidate(
            data,
            scores,
            STRATEGY,
            chip_policy=policy,
            free_hit_squads=free_hit_squads if policy is not None else None,
            plan_scores=action_plan,
            captain_scores=captain,
            chip_value_overrides=overrides if name == "action-tc-bb-fh" else None,
            tracked_player_name="Salah",
        )
        totals_by_name[name] = totals
        rows.append({"name": name, "summary": variant_summary(totals, stats, seasons)})

    frozen_payload = json.loads(
        (lens.ROOT / "analysis" / "data" / "combined_chip_policy_validation.json").read_text(encoding="utf-8")
    )
    frozen_average = float(frozen_payload["challenger"]["evaluationAveragePoints"])
    combined = next(row for row in rows if row["name"] == "action-tc-bb-fh")
    no_chips = totals_by_name["action-no-chips"]
    combined_totals = totals_by_name["action-tc-bb-fh"]
    result = {
        "status": "paired action/chip interaction audit",
        "method": "The 5% two-band-consensus transfer action plan is held fixed while audited TC/BB and causal corrected Free Hit policies are added.",
        "freeHitFitAudit": fit_audit,
        "frozenChampionWithChipsAverage": frozen_average,
        "actionCombinedAverage": combined["summary"]["average"],
        "gainVsFrozenChampion": round(combined["summary"]["average"] - frozen_average, 1),
        "chipGainOnActionPath": round(float((combined_totals[2:] - no_chips[2:]).mean()), 1),
        "pairedChipGain": [
            {
                "season": seasons[index].replace("-", "/"),
                "gain": round(float(combined_totals[index] - no_chips[index]), 1),
            }
            for index in range(2, len(seasons))
        ],
        "experiments": rows,
    }
    output = lens.ROOT / "analysis" / "data" / "action_chip_combination_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "frozenChampionWithChipsAverage": frozen_average,
                "actionCombinedAverage": result["actionCombinedAverage"],
                "gainVsFrozenChampion": result["gainVsFrozenChampion"],
                "chipGainOnActionPath": result["chipGainOnActionPath"],
                "pairedChipGain": result["pairedChipGain"],
                "experiments": [
                    {
                        "name": row["name"],
                        "average": row["summary"]["average"],
                        "minimum": row["summary"]["minimum"],
                    }
                    for row in rows
                ],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
