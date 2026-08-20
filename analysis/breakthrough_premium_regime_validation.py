"""Validate regime-change evidence and audit premium-access failures."""

from __future__ import annotations

import json

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

import calibrate_model as lens
from breakthrough_engine import regime_change_probability
from multiscale_horizon_validation import add_targets
from wildcard_freehit_ablation import champion_forecasts


def percentile(values, groups) -> np.ndarray:
    return values.groupby(groups).rank(pct=True).to_numpy(float)


def main() -> None:
    original, _ = lens.load_or_build_prepared_history()
    data = add_targets(original.reset_index(drop=True))
    immediate, plan, _ = champion_forecasts(data)
    prior_role = data.groupby(["season", "player_key"], sort=False)[
        "player_role"
    ].shift(1)
    role_changed = (
        prior_role.notna() & (prior_role.astype(str) != data["player_role"].astype(str))
    ).to_numpy(bool)
    recent_rate = data["recent_underlying_raw"].fillna(0).to_numpy(float)
    long_rate = data["long_underlying_raw"].fillna(0).to_numpy(float)
    recent_minutes = data["expected_minutes"].fillna(0).to_numpy(float)
    long_minutes = data["per_fixture_minutes"].fillna(0).to_numpy(float)
    team_shift = data["team_regime_shift"].fillna(0).to_numpy(float)
    observations = data["observations"].fillna(0).to_numpy(int)
    signal = np.asarray(
        [
            regime_change_probability(
                recent_rate[index],
                long_rate[index],
                recent_minutes[index],
                long_minutes[index],
                team_shift[index],
                bool(role_changed[index]),
                int(observations[index]),
            )
            for index in range(len(data))
        ],
        dtype=float,
    )
    groups = [data["season"], data["GW"], data["position_id"]]
    future_rank = percentile(data["target_h3"], groups)
    plan_rank = percentile(
        data.assign(_plan=plan)["_plan"],
        groups,
    )
    breakout = (future_rank >= 0.90) & (plan_rank < 0.75)
    evaluation = data["season"].isin(lens.EVALUATION_SEASONS).to_numpy(bool)
    eligible = evaluation & (data["fixture_count"].to_numpy(int) > 0)
    baseline_signal = np.clip(
        0.5
        + 0.15
        * np.divide(
            recent_rate - long_rate,
            np.maximum(np.abs(long_rate), 0.5),
        ),
        0.0,
        1.0,
    )
    feature_names = (
        "recentLongUnderlyingDelta",
        "expectedMinutesDelta",
        "teamRegimeShift",
        "roleChanged",
        "startProbability",
        "playProbability",
        "transferPressure",
        "priceRiseProbability",
        "recentPoints",
        "longPoints",
        "observations",
    )
    features = np.column_stack(
        [
            np.divide(
                recent_rate - long_rate,
                np.maximum(np.abs(long_rate), 0.5),
            ),
            (recent_minutes - long_minutes) / 30.0,
            team_shift,
            role_changed.astype(float),
            data["start_probability"].fillna(0).to_numpy(float),
            data["play_probability"].fillna(0).to_numpy(float),
            data["transfer_pressure_rank"].fillna(0.5).to_numpy(float),
            data["price_rise_probability"].fillna(0).to_numpy(float),
            data["recent_raw"].fillna(0).to_numpy(float) / 10.0,
            data["long_raw"].fillna(0).to_numpy(float) / 10.0,
            np.log1p(observations) / 6.0,
        ]
    )
    features = np.nan_to_num(features, nan=0.0, posinf=3.0, neginf=-3.0)
    learned_signal = np.full(len(data), 0.5, dtype=float)
    orders = data["season_order"].to_numpy(int)
    fit_audit = []
    for order in sorted(set(orders[evaluation])):
        train = (orders < order) & (data["fixture_count"].to_numpy(int) > 0)
        test = (orders == order) & eligible
        if train.sum() < 1000 or not test.any():
            continue
        positive = max(1, int(breakout[train].sum()))
        negative = max(1, int(train.sum() - positive))
        weights = np.where(
            breakout[train],
            0.5 * train.sum() / positive,
            0.5 * train.sum() / negative,
        )
        model = HistGradientBoostingClassifier(
            learning_rate=0.045,
            max_iter=110,
            max_leaf_nodes=15,
            min_samples_leaf=80,
            l2_regularization=4.0,
            random_state=20260820,
        )
        model.fit(features[train], breakout[train], sample_weight=weights)
        learned_signal[test] = model.predict_proba(features[test])[:, 1]
        fit_audit.append(
            {
                "seasonOrder": int(order),
                "trainingRows": int(train.sum()),
                "trainingBreakouts": int(positive),
                "testRows": int(test.sum()),
            }
        )
    regime_auc = roc_auc_score(breakout[eligible], signal[eligible])
    baseline_auc = roc_auc_score(breakout[eligible], baseline_signal[eligible])
    learned_auc = roc_auc_score(breakout[eligible], learned_signal[eligible])

    premium = json.loads(
        (lens.ROOT / "analysis" / "data" / "premium_asset_audit.json").read_text(
            encoding="utf-8"
        )
    )
    access_failures = [
        {
            "season": row["season"],
            "asset": row["asset"],
            "forecastRank": row["averagePositionForecastRank"],
            "squadRate": row["squadRate"],
            "seasonPoints": row["seasonPoints"],
        }
        for row in premium["assets"]
        if float(row["averagePositionForecastRank"]) <= 3.0
        and float(row["squadRate"]) < 10.0
        and float(row["seasonPoints"]) >= 180
    ]
    result = {
        "status": "regime and premium-access diagnostics implemented",
        "method": (
            "Bayesian-shrunk role/minutes/underlying/team change probability, "
            "evaluated against previously under-ranked three-event top-decile "
            "returns. Premium failures require a strong forecast rank but low "
            "recursive ownership; no named-player bonus is introduced."
        ),
        "regime": {
            "rows": int(eligible.sum()),
            "breakoutRate": round(float(breakout[eligible].mean()), 5),
            "baselineAuc": round(float(baseline_auc), 5),
            "regimeSignalAuc": round(float(regime_auc), 5),
            "aucDelta": round(float(regime_auc - baseline_auc), 5),
            "causalLearnedAuc": round(float(learned_auc), 5),
            "causalLearnedAucDelta": round(float(learned_auc - baseline_auc), 5),
            "featureNames": list(feature_names),
            "fitAudit": fit_audit,
            "passedDiagnosticGate": bool(learned_auc > baseline_auc),
        },
        "premiumAccessFailures": access_failures,
        "premiumAccessFailureCount": len(access_failures),
        "decision": (
            "Expose only the causal learned probability and package-level "
            "captaincy option to the action model. The hand-built regime score "
            "is rejected and must not be added directly to player points."
        ),
        "productionPromotion": False,
    }
    output = lens.ROOT / "analysis" / "data" / "breakthrough_premium_regime_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
