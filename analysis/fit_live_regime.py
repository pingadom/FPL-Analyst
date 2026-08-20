"""Fit the validated breakout detector and score the current deadline pool.

The score is intentionally diagnostic.  It is not added to expected points;
the action layer may use it as a review flag after the prospective record has
shown that doing so improves decisions.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

import calibrate_model as lens
from multiscale_horizon_validation import add_targets
from wildcard_freehit_ablation import champion_forecasts


OUTPUT = lens.ROOT / "app" / "data" / "regime-scores.json"
MODEL_PARAMS = {
    "learning_rate": 0.045,
    "max_iter": 110,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 80,
    "l2_regularization": 4.0,
    "random_state": 20260820,
}


def historical_features(data) -> np.ndarray:
    prior_role = data.groupby(["season", "player_key"], sort=False)[
        "player_role"
    ].shift(1)
    role_changed = (
        prior_role.notna()
        & (prior_role.astype(str) != data["player_role"].astype(str))
    ).to_numpy(float)
    recent_rate = data["recent_underlying_raw"].fillna(0).to_numpy(float)
    long_rate = data["long_underlying_raw"].fillna(0).to_numpy(float)
    recent_minutes = data["expected_minutes"].fillna(0).to_numpy(float)
    long_minutes = data["per_fixture_minutes"].fillna(0).to_numpy(float)
    observations = data["observations"].fillna(0).to_numpy(float)
    return np.nan_to_num(
        np.column_stack(
            [
                (recent_rate - long_rate) / np.maximum(np.abs(long_rate), 0.5),
                (recent_minutes - long_minutes) / 30.0,
                data["team_regime_shift"].fillna(0).to_numpy(float),
                role_changed,
                data["start_probability"].fillna(0).to_numpy(float),
                data["play_probability"].fillna(0).to_numpy(float),
                data["transfer_pressure_rank"].fillna(0.5).to_numpy(float),
                data["price_rise_probability"].fillna(0).to_numpy(float),
                data["recent_raw"].fillna(0).to_numpy(float) / 10.0,
                data["long_raw"].fillna(0).to_numpy(float) / 10.0,
                np.log1p(observations) / 6.0,
            ]
        ),
        nan=0.0,
        posinf=3.0,
        neginf=-3.0,
    )


def current_features(players: list[dict]) -> np.ndarray:
    rows = []
    for player in players:
        research = player.get("researchFeatures", {})
        minutes = player.get("minutesModel", {})
        recent_rate = float(research.get("recent_underlying_raw", 0.0))
        long_rate = float(research.get("long_underlying_raw", 0.0))
        expected_minutes = float(research.get("expected_minutes", player.get("expectedMinutes", 0)))
        # The live export does not retain the historical per-fixture minute
        # baseline.  Long-run expected minutes is reconstructed from the
        # current play probability and the empirical start/bench mixture.
        play = float(research.get("play_probability", minutes.get("playProbability", 0)))
        if play > 1:
            play /= 100.0
        start = float(research.get("start_probability", minutes.get("startProbability", 0)))
        if start > 1:
            start /= 100.0
        long_minutes = float(minutes.get("minutesIfStart", 82)) * start + float(
            minutes.get("minutesIfBench", 18)
        ) * max(0.0, play - start)
        role_changed = float(
            any("role" in str(flag).lower() for flag in player.get("riskFlags", []))
        )
        rows.append(
            [
                (recent_rate - long_rate) / max(abs(long_rate), 0.5),
                (expected_minutes - long_minutes) / 30.0,
                float(research.get("team_regime_shift", 0.0)),
                role_changed,
                start,
                play,
                float(research.get("transfer_pressure_rank", 0.5)),
                float(research.get("price_rise_probability", 0.0)),
                float(research.get("recent_raw", 0.0)) / 10.0,
                float(research.get("long_raw", 0.0)) / 10.0,
                np.log1p(float(research.get("observations", 0.0))) / 6.0,
            ]
        )
    return np.nan_to_num(np.asarray(rows, dtype=float), nan=0.0, posinf=3.0, neginf=-3.0)


def main() -> None:
    original, _ = lens.load_or_build_prepared_history()
    data = add_targets(original.reset_index(drop=True))
    _, plan, _ = champion_forecasts(data)
    groups = [data["season"], data["GW"], data["position_id"]]
    future_rank = data["target_h3"].groupby(groups).rank(pct=True).to_numpy(float)
    plan_rank = data.assign(_plan=plan)["_plan"].groupby(groups).rank(pct=True).to_numpy(float)
    target = (future_rank >= 0.90) & (plan_rank < 0.75)
    eligible = data["fixture_count"].to_numpy(int) > 0
    features = historical_features(data)
    positive = max(1, int(target[eligible].sum()))
    negative = max(1, int(eligible.sum() - positive))
    weights = np.where(
        target[eligible],
        0.5 * eligible.sum() / positive,
        0.5 * eligible.sum() / negative,
    )
    model = HistGradientBoostingClassifier(**MODEL_PARAMS)
    model.fit(features[eligible], target[eligible], sample_weight=weights)

    players = json.loads(
        (lens.ROOT / "app" / "data" / "current-players.json").read_text(
            encoding="utf-8"
        )
    )
    probabilities = model.predict_proba(current_features(players))[:, 1]
    validation_path = (
        lens.ROOT
        / "analysis"
        / "data"
        / "breakthrough_premium_regime_validation.json"
    )
    validation = (
        json.loads(validation_path.read_text(encoding="utf-8"))
        if validation_path.exists()
        else {}
    )
    payload = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "status": "diagnostic-only",
        "trainingRows": int(eligible.sum()),
        "trainingBreakouts": int(positive),
        "walkForwardAuc": validation.get("regime", {}).get("causalLearnedAuc"),
        "players": [
            {
                "id": int(player["id"]),
                "name": player["name"],
                "probability": round(float(probability), 6),
            }
            for player, probability in zip(players, probabilities)
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(OUTPUT.relative_to(lens.ROOT)),
                "players": len(players),
                "walkForwardAuc": payload["walkForwardAuc"],
                "top": sorted(payload["players"], key=lambda row: row["probability"], reverse=True)[:8],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
