"""Audit package predictions at the optimiser boundary where decisions occur."""

from __future__ import annotations

import json

import numpy as np

import calibrate_model as lens
from decision_focused_horizon_validation import target_and_maturity
from multiscale_horizon_validation import add_targets
from package_action_value_validation import (
    FeatureBuilder,
    causal_fits,
    checkpoint_for_event,
    collect_packages,
)
from probabilistic_component_challenger import causal_route_predictions
from transfer_action_ranker_validation import agreed_action_plan
from wildcard_freehit_ablation import champion_forecasts


def main() -> None:
    original, _ = lens.load_or_build_prepared_history()
    data = add_targets(original.reset_index(drop=True))
    scores, champion_plan, captain = champion_forecasts(data)
    component, _ = causal_route_predictions(data, scores)
    action_plan = agreed_action_plan(data, champion_plan, 0.05)
    target, maturity = target_and_maturity(data)
    builder = FeatureBuilder(
        data, scores, champion_plan, action_plan, component, target, maturity
    )
    metadata, features = collect_packages(
        data, scores, champion_plan, captain, builder
    )
    fits, _ = causal_fits(metadata, features)

    probabilities = []
    actions = []
    values = []
    rows = []
    for index, row in metadata.iterrows():
        if (
            not bool(row["chosenByChampion"])
            or str(row["season"]) not in lens.EVALUATION_SEASONS
            or bool(row["goalkeeperPackage"])
        ):
            continue
        fitted = fits.get(
            (int(row["seasonOrder"]), checkpoint_for_event(int(row["event"])))
        )
        if fitted is None:
            continue
        probability = float(
            np.mean(
                [
                    classifier.predict_probability(features[index])
                    for classifier in fitted.classifiers
                ]
            )
        )
        action = bool(float(row["targetGain"]) > 0)
        probabilities.append(probability)
        actions.append(action)
        values.append(float(row["targetGain"]))
        rows.append(
            {
                "season": str(row["season"]).replace("-", "/"),
                "gw": int(row["gw"]),
                "moves": int(row["moves"]),
                "predictedGain": round(float(row["predictedGain"]), 3),
                "adaptiveTargetGain": round(float(row["targetGain"]), 3),
                "positiveProbability": round(probability, 4),
            }
        )

    probability_values = np.asarray(probabilities)
    action_values = np.asarray(actions, dtype=bool)
    target_values = np.asarray(values)
    all_chosen = metadata[metadata["chosenByChampion"]]
    by_moves = []
    for moves, frame in metadata.groupby("moves", sort=True):
        by_moves.append(
            {
                "moves": int(moves),
                "candidates": len(frame),
                "meanPredictedGain": round(float(frame["predictedGain"].mean()), 3),
                "meanAdaptiveTargetGain": round(float(frame["targetGain"].mean()), 3),
                "meanOvervaluation": round(
                    float((frame["predictedGain"] - frame["targetGain"]).mean()), 3
                ),
            }
        )
    result = {
        "status": "package optimiser-boundary diagnostic",
        "method": "Evaluate causal action probabilities only on non-goalkeeper packages the frozen champion actually selected, rather than on the full generated frontier.",
        "frontierPackages": len(metadata),
        "championChosenPackagesAllSeasons": len(all_chosen),
        "evaluationBoundary": {
            "rows": len(rows),
            "positiveRate": round(float(action_values.mean()), 4),
            "alwaysPositiveAccuracy": round(float(action_values.mean()), 4),
            "classifierAccuracy50": round(
                float(np.mean((probability_values >= 0.50) == action_values)), 4
            ),
            "brier": round(
                float(np.mean((probability_values - action_values) ** 2)), 4
            ),
            "probabilityValueCorrelation": round(
                float(np.corrcoef(probability_values, target_values)[0, 1]), 4
            ),
        },
        "byMoveCount": by_moves,
        "decisions": rows,
    }
    output = lens.ROOT / "analysis" / "data" / "package_boundary_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "decisions"}, indent=2))


if __name__ == "__main__":
    main()
