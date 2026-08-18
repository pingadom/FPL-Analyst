"""Fit the final past-only Ridge calibration used as a live FH cross-check."""

from __future__ import annotations

import json

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import calibrate_model as lens
from freehit_value_validation import FEATURES, opportunity_frame
from frontier_ranker_validation import STRATEGY
from multiscale_horizon_validation import add_targets
from wildcard_freehit_ablation import champion_forecasts


def main() -> None:
    original, _ = lens.load_or_build_prepared_history()
    data = add_targets(original.reset_index(drop=True))
    scores, plan_scores, captain_scores = champion_forecasts(data)
    seasons = list(dict.fromkeys(data["season"].tolist()))
    free_hit_squads = lens.precompute_fresh_squads(data, scores)
    collector = lens.ChipPolicy(
        1e6, 1e6, 1e6, 1e6, 0.0, 10, 28, ("Free Hit",)
    )
    print("Collecting past Free Hit opportunities", flush=True)
    _, stats = lens.simulate_candidate(
        data,
        scores,
        STRATEGY,
        chip_policy=collector,
        free_hit_squads=free_hit_squads,
        plan_scores=plan_scores,
        captain_scores=captain_scores,
    )
    frame = opportunity_frame(stats, seasons)
    matrix = frame[list(FEATURES)].astype(float).to_numpy()
    target = frame["actualFreeHitGain"].to_numpy(float)
    most_recent = int(frame["seasonOrder"].max())
    age = most_recent - frame["seasonOrder"].to_numpy(int)
    weights = np.power(0.88, np.maximum(age, 0))
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    model = Ridge(alpha=45.0)
    model.fit(scaled, target, sample_weight=weights)
    fitted = model.predict(scaled)
    robust_scale = max(
        4.0, 1.4826 * float(np.median(np.abs(target - fitted)))
    )
    causal_path = lens.ROOT / "analysis" / "data" / "freehit_value_validation.json"
    causal = json.loads(causal_path.read_text(encoding="utf-8"))
    result = {
        "status": "past-only live Free Hit cross-check; prospective shadow",
        "trainedThrough": seasons[-1].replace("-", "/"),
        "trainingSeasons": [season.replace("-", "/") for season in seasons],
        "trainingRows": int(len(frame)),
        "features": list(FEATURES),
        "featureMean": scaler.mean_.round(10).tolist(),
        "featureScale": scaler.scale_.round(10).tolist(),
        "coefficients": model.coef_.round(10).tolist(),
        "intercept": round(float(model.intercept_), 10),
        "robustResidualScale": round(robust_scale, 4),
        "activationThreshold": 3.0,
        "causalValidation": causal["forecastAudit"],
        "warning": (
            "The final fit uses all completed seasons only for the next prospective "
            "season. Historical performance claims come from expanding-window "
            "causal predictions, not this in-sample fit."
        ),
    }
    output = lens.ROOT / "analysis" / "data" / "freehit_live_calibration.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "trainingRows": result["trainingRows"],
                "residualScale": result["robustResidualScale"],
                "causalValidation": result["causalValidation"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
